import os
import random
import numpy as np
from typing import Optional
from datetime import timedelta
from decimal import Decimal, getcontext

from h_adminsim.task import OutpatientTask
from h_adminsim.task.opfv_task import *
from h_adminsim.task.opfu_task import *
from h_adminsim.registry import SCHEDULE_STATUS
from h_adminsim.task.fhir_manager import FHIRManager
from h_adminsim.tools import DataConverter
from h_adminsim.environment.hospital import HospitalEnvironment
from h_adminsim.utils.filesys_utils import json_load, json_save_fast, get_files
from h_adminsim.utils.common_utils import compare_iso_time, get_iso_time, str_to_datetime



class Simulator:
    def __init__(self,
                 task: dict,
                 simulation_start_day_before: float = 3,
                 fhir_integration: bool = False,
                 fhir_url: Optional[str] = None,
                 fhir_max_connection_retries: int = 5,
                 random_seed: int = 9999):
        
        # Initialize
        getcontext().prec = 10
        self.simulation_start_day_before = simulation_start_day_before
        self.fhir_integration = fhir_integration
        self.fhir_url = fhir_url if self.fhir_integration else None
        self.fhir_max_connection_retries = fhir_max_connection_retries
        self.task = self._init_task(task)
        self.random_seed = random_seed


    def __env_setup(self, random_seed: int, resume: bool):
        """
        Initialize environment-level random seeds.

        Args:
            random_seed (int): Random seed.
            resume (bool): Whether resumed simulation or not.
        """
        random.seed(random_seed)
        np.random.seed(random_seed)

        if self.fhir_integration and not resume:
            fhir_manager = FHIRManager(self.fhir_url)
            appointment_entries = fhir_manager.read_all('Appointment')
            patient_entries = fhir_manager.read_all('Patient')
            fhir_manager.delete_all(appointment_entries, verbose=False)
            fhir_manager.delete_all(patient_entries, verbose=False)



    def _init_task(self, task: dict) -> dict[str, OutpatientTask]:
        """
        Initialize the task queue for first-visit outpatient workflow.

        Args:
            task (dict): Task dictionary.
        """
        assert len(task), \
            colorstr("red", "At least one of `first_visit_intake`, `first_visit_scheduliing`, or `follow_up_visit_scheduling` must be provided.")

        if 'first_visit_intake' in task:
            assert isinstance(task['first_visit_intake'], OutpatientFirstIntake), \
                colorstr("red", "Wrong type of `first_visit_intake` task.")
        if 'first_visit_scheduling' in task:
            assert isinstance(task['first_visit_scheduling'], OutpatientFirstScheduling), \
                colorstr("red", "Wrong type of `first_visit_scheduling` task.")
            assert self.fhir_integration == task['first_visit_scheduling'].fhir_integration, \
                colorstr("red", "FHIR integration setting mismatch between Simulator and `first_visit_scheduling` task.")
        if 'follow_up_visit_scheduling' in task:
            assert isinstance(task['follow_up_visit_scheduling'], OutpatientFollowUpScheduling), \
                colorstr("red", "Wrong type of `follow_up_visit_scheduling` task.")
            assert self.fhir_integration == task['follow_up_visit_scheduling'].fhir_integration, \
                colorstr("red", "FHIR integration setting mismatch between Simulator and `follow_up_visit_scheduling` task.")
            
        if 'first_visit_intake' in task and 'follow_up_visit_scheduling' in task:
            assert 'first_visit_scheduling' in task, colorstr('red', '`first_visit_scheduling` simulation must be required.')
        
        return task
    
    
    def shuffle_data(self, data: dict):
        """
        Shuffle the agent test data and (when applicable) split follow-ups
        into a pending pool that will be activated by the run loop only after
        each patient's first-visit appointment time has passed.

        Args:
            data (dict): An agent test data to simulate a hospital environmnet.
        """
        has_followup = 'follow_up_visit_scheduling' in self.task
        has_first = any(t in self.task for t in ['first_visit_intake', 'first_visit_scheduling'])

        # Only follow-up visit case
        if has_followup and not has_first:
            followups = [x for x in data['agent_data'] if x[0]['visit_type'] == 'follow_up_visit']
            random.shuffle(followups)
            data['agent_data'] = followups
            data['_pending_followups'] = {}
            data['_ready_followups'] = list(followups)
        
        # Only first-visit case
        elif has_first and not has_followup:
            data['agent_data'] = list(filter(lambda x: x[0]['visit_type'] == 'first_visit', data['agent_data']))
            random.shuffle(data['agent_data'])
            data['_pending_followups'] = {}
            data['_ready_followups'] = []
        
        # First-visit and follow-up viist cases
        else:
            first_list = [x for x in data['agent_data'] if x[0]['visit_type'] == 'first_visit']
            followup_list = [x for x in data['agent_data'] if x[0]['visit_type'] == 'follow_up_visit']
            random.shuffle(first_list)
            data['agent_data'] = first_list
            data['_pending_followups'] = {gt['patient']: (gt, td) for gt, td in followup_list}
            data['_ready_followups'] = []


    @staticmethod
    def _promote_eligible_followups(pending: dict,
                                    ready: list,
                                    environment: HospitalEnvironment,
                                    processed_first_patients: set):
        """
        Promote pending follow-up scheduling tasks into the ready pool once
        each patient's first-visit appointment has finished, and drop those
        whose first-visit will never produce a usable appointment.
        The pool mutations are in-place. Safe to call when `pending` is empty.

        Args:
            pending (dict): Map of `patient_name -> (gt, test_data)` for follow-ups waiting on their corresponding first-visit.
            ready (list): Pool of `(gt, test_data)` follow-ups eligible for immediate scheduling. Newly-eligible entries are appended.
            environment (HospitalEnvironment): Provides `patient_schedules`, `current_time`, and `_utc_offset` used to evaluate eligibility.
            processed_first_patients (set): Patient names whose first-visit entry has already been popped from the first-visit queue. 
                                            Used to detect first-visit scheduling failures (popped, but no resulting `patient_schedules` entry).
        """
        if not pending:
            return

        scheduled_by_patient = {}
        for s in environment.patient_schedules:
            if s.get('visit_type') != 'first_visit':
                continue
            scheduled_by_patient.setdefault(s.get('patient'), []).append(s)

        for name in list(pending.keys()):
            entries = scheduled_by_patient.get(name, [])

            # Promote: first-visit appointment has finished
            if any(s.get('status') == SCHEDULE_STATUS['completed'] for s in entries):
                ready.append(pending.pop(name))
                continue

            # In flight: scheduled or in_progress, awaiting time advance
            if any(s.get('status') != SCHEDULE_STATUS['cancelled'] for s in entries):
                continue

            # Drop: all entries cancelled, or first-visit was processed but produced no schedule (error occurred case)
            if entries or name in processed_first_patients:
                pending.pop(name)


    @staticmethod
    def _advance_for_pending(pending: dict, environment: HospitalEnvironment) -> bool:
        """
        Advance `environment.current_time` past the earliest still-pending first-visit's end time so its follow-up can be promoted on the next iteration. 
        Capped at the simulation end (`_END_HOUR` on `_END_DATE`); only moves time forward.

        Args:
            pending (dict): Map of `patient_name -> (gt, test_data)` for follow-ups waiting on their first-visit.
            environment (HospitalEnvironment): Provides `patient_schedules`, `current_time`, `_utc_offset`, and `_END_HOUR`/`_END_DATE`. 
                                               Mutated in-place via `update_current_time(new_time=...)`.

        Returns:
            bool: True if `current_time` actually advanced (caller should retry promotion). 
                  False if no. candidate falls within the simulation window or `current_time` is already at the cap (caller should break).
        """
        if not pending:
            return False

        sim_end_iso = get_iso_time(environment._END_HOUR, environment._END_DATE, environment._utc_offset)
        pending_names = set(pending.keys())
        candidate_ends = []
        for s in environment.patient_schedules:
            if s['visit_type'] == 'first_visit' and s['patient'] in pending_names and s['status'] in [SCHEDULE_STATUS['scheduled'], SCHEDULE_STATUS['in_progress']]:
                end_iso = get_iso_time(s['schedule'][-1], date=s['date'], utc_offset=environment._utc_offset)
                if not compare_iso_time(end_iso, sim_end_iso):
                    candidate_ends.append(end_iso)
        
        if not candidate_ends:
            return False

        target_dt = str_to_datetime(min(candidate_ends)) + timedelta(seconds=1)
        new_iso = target_dt.isoformat(timespec='seconds')
        if compare_iso_time(new_iso, sim_end_iso):
            new_iso = sim_end_iso
        if not compare_iso_time(new_iso, environment.current_time):
            return False

        environment.update_current_time(new_time=new_iso)
        return True


    @staticmethod
    def _seed_virtual_first_visit(gt: dict,
                                  agent_simulation_data: dict,
                                  environment: HospitalEnvironment) -> bool:
        """
        Seed a synthetic first-visit for follow-up-only mode. Picks the earliest
        free aligned slot in the doctor's calendar from `_START_DATE` forward,
        then advances `current_time` past the seed end if needed. No-op if a
        first-visit entry already exists for the patient.

        Args:
            gt (dict): Ground-truth follow-up data (`patient`, `attending_physician`, ...).
            agent_simulation_data (dict): Static simulation data; mutated to mark the chosen slot in `doctor[<name>]['schedule']`.
            environment (HospitalEnvironment): Mutated in-place — appends to `patient_schedules`,
                                               increments `booking_num`, advances `current_time`.

        Returns:
            bool: True if a first-visit is available (existing entry or new seed placed).
                  False if the doctor's calendar has no free slot in the simulation window — caller should skip this patient.
        """
        # Follow-up patients with a prior first visit
        name = gt['patient']
        if any(s.get('patient') == name and s.get('visit_type') == 'first_visit'
               for s in environment.patient_schedules):
            return True

        # Follow-up patients without a prior first visit (only follow-up visit case)
        doctor = gt['attending_physician']
        doctor_info = agent_simulation_data['doctor'][doctor]
        department = doctor_info['department']
        time_unit = float(Decimal('1') / Decimal(str(doctor_info['capacity_per_hour'])))

        start_hour = environment._START_HOUR
        end_hour = environment._END_HOUR

        past_date, past_slot = None, None
        cur_dt = str_to_datetime(environment._START_DATE)
        end_dt = str_to_datetime(environment._END_DATE)
        while cur_dt <= end_dt:
            date = cur_dt.strftime('%Y-%m-%d')
            existing = doctor_info['schedule'].get(date, [])
            h = start_hour
            while True:
                next_h = float(Decimal(str(h)) + Decimal(str(time_unit)))
                if next_h > end_hour + 1e-9:
                    break
                slot = [h, next_h]
                if not any(slot[0] < e[1] - 1e-9 and e[0] < slot[1] - 1e-9 for e in existing):
                    past_date, past_slot = date, slot
                    break
                h = next_h
            if past_slot is not None:
                break
            cur_dt += timedelta(days=1)

        if past_slot is None:
            log(
                f"No free slot for virtual first-visit "
                f"(patient={name}, doctor={doctor}) within simulation window; "
                f"skipping follow-up scheduling for this patient.",
                level='warning',
            )
            return False

        virtual = {
            'visit_type': 'first_visit',
            'patient': name,
            'attending_physician': doctor,
            'department': department,
            'date': past_date,
            'schedule': past_slot,
            'patient_intention': None,
            'preference': None,
            'preferred_doctor': None,
            'valid_from': None,
            'test': None,
            'last_updated_time': environment.current_time,
            'waiting_order': -1,
            'status': SCHEDULE_STATUS['completed'],
        }
        environment.patient_schedules.append(virtual)
        environment.booking_num[doctor] += 1
        doctor_info['schedule'].setdefault(past_date, []).append(past_slot)
        doctor_info['schedule'][past_date].sort()

        # Build and upload FHIR Patient + Appointment for the virtual seed when integrated.
        if environment.fhir_manager is not None:
            metadata = agent_simulation_data.get('metadata')
            department_data = agent_simulation_data.get('department')
            fhir_patient = DataConverter.data_to_patient({
                'metadata': metadata,
                'department': department_data,
                'patient': {
                    name: [{
                        'department': department,
                        'gender': gt['gender'],
                        'telecom': gt['telecom'],
                        'birthDate': gt['birthDate'],
                        'identifier': gt['identifier'],
                        'address': gt['address'],
                    }]
                }
            })[0]
            fhir_appointment = DataConverter.data_to_appointment({
                'metadata': metadata,
                'department': department_data,
                'patient': {
                    name: [{
                        'visit_type': 'first_visit',
                        'department': department,
                        'attending_physician': doctor,
                        'date': past_date,
                        'schedule': past_slot,
                    }]
                }
            })[0]
            environment.update_fhir({'Patient': fhir_patient, 'Appointment': fhir_appointment})

        # Advance current_time past the virtual first-visit so it is "completed".
        end_iso = get_iso_time(past_slot[-1], date=past_date, utc_offset=environment._utc_offset)
        if compare_iso_time(end_iso, environment.current_time):
            new_iso = (str_to_datetime(end_iso) + timedelta(seconds=1)).isoformat(timespec='seconds')
            environment.update_current_time(new_time=new_iso)
        else:
            environment.update_patient_status()

        return True


    @staticmethod
    def resume_results(agent_simulation_data: dict, results_path: str) -> Tuple[dict, dict, set]:
        """
        Resume a previously saved simulation by aligning agent results.

        Args:
            agent_simulation_data (dict): Static agent test data for a simulation.
            results_path (str): Path to the JSON file containing the saved simulation results.

        Returns:
            Tuple[dict, int]:
                - dict: Schedule updated static agent test data.
                - dict: Previously saved agent results.
                - set: A dictionary containing patients that have already been processed for each task.
        """
        # Load previous results
        agent_results = json_load(results_path)

        # Get patients that have already been processed for each task
        done_patients = dict()
        for task_name, result in agent_results.items():
            if task_name == 'first_visit_intake':
                done_patients[task_name] = {done['patient']['name'] for done in result['gt']}
            elif task_name == 'first_visit_scheduling':
                done_patients[task_name] = set()
                for done in result['gt']:
                    try:
                        done_patients[task_name].add(done['patient'])
                    except (KeyError, TypeError):
                        continue
        
        # Updated doctor schedules based on the resumed results
        if 'first_visit_scheduling' in agent_results:
            fixed_schedule = agent_simulation_data['doctor']
            statuses = [x for y in agent_results['first_visit_scheduling']['status'] for x in (y if isinstance(y, list) or isinstance(y, tuple) else [y])]
            preds = [x for y in agent_results['first_visit_scheduling']['pred'] for x in (y if isinstance(y, list) or isinstance(y, tuple) else [y])]
            for status, pred in zip(statuses, preds):
                if status and 'status' in pred and pred['status'] != SCHEDULE_STATUS['cancelled']:
                    fixed_schedule[pred['attending_physician']]['schedule'][pred['date']].append(pred['schedule'])
                    fixed_schedule[pred['attending_physician']]['schedule'][pred['date']].sort()
        
        for task, patients in done_patients.items():
            log(f"{task:<20}: {colorstr(len(patients))} patient(s) resumed")
        
        return agent_simulation_data, agent_results, done_patients

    
    def run(self,
            simulation_data_path: str,
            output_dir: str,
            resume: bool = False,
            verbose: bool = False):
        """
        Run the agent-based hospital administrative simulation.

        Args:
            simulation_data_path (str): Path to a JSON file or directory containing agent simulation input data.
            output_dir (str): Directory to store simulation results.
            resume (bool, optional): Whether to resume a previous simulation if result files exist. Defaults to False.
            verbose (bool, optional): Whether to print detailed logs during task execution. Defaults to False.

        Raises:
            Exception: Propagates any errors encountered during simulation or result saving.
        """
        # Initialize environment
        self.__env_setup(self.random_seed, resume)
        
        # Load agent simulation data
        is_file = os.path.isfile(simulation_data_path)
        agent_simulation_data_files = [simulation_data_path] if is_file else get_files(simulation_data_path, ext='json')

        try:
            os.makedirs(output_dir, exist_ok=True)

            # Data per hospital
            for path in agent_simulation_data_files:
                agent_simulation_data = json_load(path)
                agent_results, done_patients = dict(), dict()
                self.shuffle_data(agent_simulation_data)
                environment = HospitalEnvironment(
                    agent_simulation_data,
                    self.fhir_url,
                    self.fhir_max_connection_retries,
                    self.simulation_start_day_before
                )
                basename = os.path.splitext(os.path.basename(path))[0]
                save_path = os.path.join(output_dir, f'{basename}_result.json')
                log(f'{basename} simulation started..', color=True)

                # Resume the results and the virtual hospital environment
                if resume and os.path.exists(save_path):
                    agent_simulation_data, agent_results, done_patients = Simulator.resume_results(agent_simulation_data, save_path)
                    environment.resume(agent_results)

                # Data per patient
                index = 0
                first_queue = [x for x in agent_simulation_data['agent_data'] if x[0]['visit_type'] == 'first_visit']
                pending = dict(agent_simulation_data.get('_pending_followups', {}))
                ready = list(agent_simulation_data.get('_ready_followups', []))
                processed_first_patients = set()

                while first_queue or ready or pending:
                    Simulator._promote_eligible_followups(pending, ready, environment, processed_first_patients)

                    if not first_queue and not ready:
                        if pending and Simulator._advance_for_pending(pending, environment):
                            continue
                        break

                    n_first, n_ready = len(first_queue), len(ready)
                    pick_followup = bool(ready) and (n_first == 0 or random.random() < n_ready / (n_first + n_ready))

                    if pick_followup:
                        gt, test_data = ready.pop(random.randrange(len(ready)))
                    else:
                        gt, test_data = first_queue.pop(0)
                        processed_first_patients.add(gt['patient'])

                    # Make a task queue
                    task_queue = list()
                    if test_data['visit_type'] == 'first_visit':
                        if 'first_visit_intake' in self.task:
                            task_queue.append(self.task['first_visit_intake'])
                        if 'first_visit_scheduling' in self.task:
                            task_queue.append(self.task['first_visit_scheduling'])
                    elif test_data['visit_type'] == 'follow_up_visit':
                        if 'follow_up_visit_scheduling' in self.task:
                            task_queue.append(self.task['follow_up_visit_scheduling'])

                    for task in task_queue:
                        if task.name in done_patients and gt['patient'] in done_patients[task.name]:
                            continue

                        # Seed a virtual past first-visit when running follow-up only
                        if task.name == 'follow_up_visit_scheduling':
                            if not Simulator._seed_virtual_first_visit(gt, agent_simulation_data, environment):
                                continue

                        # Simulation results
                        result = task((gt, test_data), agent_simulation_data, agent_results, environment, verbose)

                        # Append a single result
                        agent_results.setdefault(task.name, init_result_dict())
                        for k in result:
                            agent_results[task.name][k] += result[k]
                    
                    # Index for debugging
                    index += 1

                # Logging the results
                for task_name, result in agent_results.items():
                    if task_name == 'first_visit_intake':
                        statuses = [all(s.values()) for s in result['status']]
                        status_codes = ['/'.join(s.values()) for s in result['status_code']]
                    else:
                        statuses = result['status']
                        status_codes = result['status_code']
                    correctness = [x for y in statuses for x in (y if isinstance(y, list) or isinstance(y, tuple) else [y])]
                    status_code = [x for y in status_codes for x in (y if isinstance(y, list) or isinstance(y, tuple) else [y])]
                    accuracy = sum(correctness) / len(correctness)
                    log(f'{basename} - {task_name} task results..', color=True)
                    log(f'   - accuracy: {accuracy:.3f}, length: {len(correctness)}, status_code: {status_code}')

                json_save_fast(save_path, agent_results)
            
            log(f"Agent completed the tasks successfully", color=True)
        
        except Exception as e:
            if len(agent_results):
                json_save_fast(save_path, agent_results)
            log(f"Error occured while execute the tasks: {e}", level='error')
            raise
