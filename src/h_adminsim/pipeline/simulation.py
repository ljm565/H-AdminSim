import os
import random
import numpy as np
from typing import Optional

from h_adminsim.task.agent_task import *
from h_adminsim.task.fhir_manager import FHIRManager
from h_adminsim.environment.hospital import HospitalEnvironment
from h_adminsim.utils.filesys_utils import json_load, json_save_fast, get_files



class Simulator:
    def __init__(self,
                 intake_task: Optional[OutpatientFirstIntake] = None,
                 scheduling_task: Optional[OutpatientFirstScheduling] = None,
                 simulation_start_day_before: float = 3,
                 fhir_integration: bool = False,
                 fhir_url: Optional[str] = None,
                 fhir_max_connection_retries: int = 5,
                 random_seed: int = 9999):
        
        # Initialize
        self.simulation_start_day_before = simulation_start_day_before
        self.fhir_integration = fhir_integration
        self.fhir_url = fhir_url if self.fhir_integration else None
        self.fhir_max_connection_retries = fhir_max_connection_retries
        self.task_queue, self.task_list = self._init_task(intake_task, scheduling_task)
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



    def _init_task(self, 
                   intake_task: Optional[OutpatientFirstIntake] = None,
                   scheduling_task: Optional[OutpatientFirstScheduling] = None) -> Tuple[list[FirstVisitOutpatientTask], list[str]]:
        """
        Initialize the task queue for first-visit outpatient workflow.

        Args:
            intake_task (Optional[OutpatientFirstIntake], optional): Intake task instance to include in the queue. Defaults to None.
            scheduling_task (Optional[OutpatientFirstScheduling], optional): Scheduling task instance to include in the queue. Defaults to None.

        Returns:
            Tuple[list[FirstVisitOutpatientTask], list[str]]:
                A tuple containing:
                    - the ordered list of task objects
                    - the list of task names in execution order
        """
        task_queue, task_list = list(), list()
        assert intake_task != None or scheduling_task != None, \
            log("At least one of 'intake_task' or 'scheduling_task' must be provided (both cannot be None).", level='error')

        if intake_task != None:
            task_queue.append(intake_task)
            task_list.append(intake_task.name)
        if scheduling_task != None:
            task_queue.append(scheduling_task)
            task_list.append(scheduling_task.name)
        
        return task_queue, task_list
    
    
    @staticmethod
    def shuffle_data(data: dict):
        """
        Shuffle the agent test data by the schedule start time.

        Args:
            data (dict): An agent test data to simulate a hospital environmnet.
        """
        random.shuffle(data['agent_data'])        # In-place logic


    @staticmethod
    def resume_results(agent_simulation_data: dict, results_path: str, d_results_path: str) -> Tuple[dict, dict, dict, set]:
        """
        Resume a previously saved simulation by aligning agent results.

        Args:
            agent_simulation_data (dict): Static agent test data for a simulation.
            results_path (str): Path to the JSON file containing the saved simulation results.
            d_results_path (str): Path to the JSON file containing the saved dialog results.

        Returns:
            Tuple[dict, int]:
                - dict: Schedule updated static agent test data.
                - dict: Previously saved agent results.
                - dict: Previously saved dialog results.
                - set: A dictionary containing patients that have already been processed for each task.
        """
        # Load previous results
        agent_results = json_load(results_path)
        dialog_results = json_load(d_results_path) if os.path.exists(d_results_path) else dict()

        # Get patients that have already been processed for each task
        done_patients = dict()
        for task_name, result in agent_results.items():
            if task_name == 'intake':
                done_patients[task_name] = {done['patient']['name'] for done in result['gt']}
            elif task_name == 'schedule':
                done_patients[task_name] = set()
                for done in result['gt']:
                    try:
                        done_patients[task_name].add(done['patient'])
                    except (KeyError, TypeError):
                        continue
        
        # Updated doctor schedules based on the resumed results
        if 'schedule' in agent_results:
            fixed_schedule = agent_simulation_data['doctor']
            statuses = [x for y in agent_results['schedule']['status'] for x in (y if isinstance(y, list) or isinstance(y, tuple) else [y])]
            preds = [x for y in agent_results['schedule']['pred'] for x in (y if isinstance(y, list) or isinstance(y, tuple) else [y])]
            for status, pred in zip(statuses, preds):
                if status and 'status' in pred and pred['status'] != 'cancelled':
                    fixed_schedule[pred['attending_physician']]['schedule'][pred['date']].append(pred['schedule'])
                    fixed_schedule[pred['attending_physician']]['schedule'][pred['date']].sort()
        
        return agent_simulation_data, agent_results, dialog_results, done_patients

    
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
                agent_results, done_patients, dialog_results = dict(), dict(), dict()
                Simulator.shuffle_data(agent_simulation_data)
                environment = HospitalEnvironment(
                    agent_simulation_data,
                    self.fhir_url,
                    self.fhir_max_connection_retries,
                    self.simulation_start_day_before
                )
                basename = os.path.splitext(os.path.basename(path))[0]
                save_path = os.path.join(output_dir, f'{basename}_result.json')
                d_save_path = os.path.join(output_dir, f'{basename}_dialog.json')
                log(f'{basename} simulation started..', color=True)

                # Resume the results and the virtual hospital environment
                if resume and os.path.exists(save_path):
                    agent_simulation_data, agent_results, dialog_results, done_patients = Simulator.resume_results(agent_simulation_data, save_path, d_save_path)
                    environment.resume(agent_results)

                # Data per patient
                for j, (gt, test_data) in enumerate(agent_simulation_data['agent_data']):
                    for task in self.task_queue:
                        if task.name in done_patients and gt['patient'] in done_patients[task.name]:
                            continue

                        result = task((gt, test_data), agent_simulation_data, agent_results, environment, verbose)
                        dialogs = result.pop('dialog')

                        # Append a single result 
                        agent_results.setdefault(task.name, {'gt': [], 'pred': [], 'status': [], 'status_code': [], 'trial': [], 'dialog': []})
                        for k in result:
                            agent_results[task.name][k] += result[k]
                        
                        if task.name == 'intake':
                            dialog_results[gt['patient']] = dialogs[0]
                        else:
                            agent_results[task.name]['dialog'] += dialogs

                # Logging the results
                for task_name, result in agent_results.items():
                    correctness = [x for y in result['status'] for x in (y if isinstance(y, list) or isinstance(y, tuple) else [y])]
                    status_code = [x for y in result['status_code'] for x in (y if isinstance(y, list) or isinstance(y, tuple) else [y])]
                    accuracy = sum(correctness) / len(correctness)
                    log(f'{basename} - {task_name} task results..', color=True)
                    log(f'   - accuracy: {accuracy:.3f}, length: {len(correctness)}, status_code: {status_code}')

                json_save_fast(save_path, agent_results)
                if 'intake' in self.task_list:
                    json_save_fast(d_save_path, dialog_results)
            
            log(f"Agent completed the tasks successfully", color=True)
        
        except Exception as e:
            if len(agent_results):
                json_save_fast(save_path, agent_results)
                if 'intake' in self.task_list:
                    json_save_fast(d_save_path, dialog_results)
            log("Error occured while execute the tasks.", level='error')
            raise e
