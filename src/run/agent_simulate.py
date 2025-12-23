import os
import sys
import random
import numpy as np
from sconf import Config
from typing import Tuple
from argparse import ArgumentParser
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from h_adminsim import AdminStaffAgent, SupervisorAgent
from h_adminsim.task.fhir_manager import FHIRManager
from h_adminsim.task.agent_task import *
from h_adminsim.environment.hospital import HospitalEnvironment
from h_adminsim.utils import log
from h_adminsim.utils.filesys_utils import json_load, json_save_fast, yaml_save, get_files



def env_setup(config, is_continue):
    random.seed(config.seed)
    np.random.seed(config.seed)

    # Delete Patient and Appointment resources when starting a simulation
    if config.integration_with_fhir and not is_continue:
        fhir_manager = FHIRManager(config.fhir_url)
        appointment_entries = fhir_manager.read_all('Appointment')
        patient_entries = fhir_manager.read_all('Patient')
        fhir_manager.delete_all(appointment_entries, verbose=False)
        fhir_manager.delete_all(patient_entries, verbose=False)


def load_config(config_path):
    config = Config(config_path)
    return config


def shuffle_agent_test_data(agent_test_data: dict):
    """
    Shuffle the agent test data by the schedule start time.

    Args:
        agent_test_data (dict): An agent test data to simulate a hospital environmnet.
    """
    random.shuffle(agent_test_data['agent_data'])        # In-place logic


def resume_results(agent_test_data: dict, results_path: str, d_results_path: str) -> Tuple[dict, dict, dict, set]:
    """
    Resume a previously saved simulation by aligning agent results.

    Args:
        agent_test_data (dict): Static agent test data for a simulation.
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
                except KeyError:
                    continue
    
    # Updated doctor schedules based on the resumed results
    if 'schedule' in agent_results:
        fixed_schedule = agent_test_data['doctor']
        for status, pred in zip(agent_results['schedule']['status'], agent_results['schedule']['pred']):
            if status and pred['status'] != 'cancelled':
                fixed_schedule[pred['attending_physician']]['schedule'][pred['date']].append(pred['schedule'])
                fixed_schedule[pred['attending_physician']]['schedule'][pred['date']].sort()
    
    return agent_test_data, agent_results, dialog_results, done_patients


def main(args):
    # Init config
    config = load_config(args.config)
    config.yaml_file = args.config
    
    # Init environment
    env_setup(config, args.resume)

    # Initialize tasks
    queue = list()
    if 'intake' in args.type:
        use_vllm = False if any(m in config.supervisor_model.lower() for m in ['gpt', 'gemini']) else True
        supervisor_agent = SupervisorAgent(
            target_task='first_outpatient_intake',
            model=config.supervisor_model,
            use_vllm=use_vllm,
            vllm_endpoint = config.vllm_url if use_vllm else None
        )
        queue.append(OutpatientFirstIntake(
            patient_model=config.task_model,
            admin_staff_model=config.task_model,
            supervisor_agent=supervisor_agent if config.outpatient_intake.use_supervisor else None,
            intake_max_inference=config.intake_max_inference,
            admin_staff_last_task_user_prompt_path=config.outpatient_intake.staff_task_user_prompt,
        ))
        # queue.append(OutpatientIntake(config))
    if 'schedule' in args.type:
        use_sup_vllm = False if any(m in config.supervisor_model.lower() for m in ['gpt', 'gemini']) else True
        supervisor_agent = SupervisorAgent(
            target_task='first_outpatient_scheduling',
            model=config.supervisor_model,
            use_vllm=use_sup_vllm,
            vllm_endpoint = config.vllm_url if use_sup_vllm else None
        )
        use_admin_vllm = False if any(m in config.task_model.lower() for m in ['gpt', 'gemini']) else True
        admin_staff_agent = AdminStaffAgent(
            target_task='first_outpatient_scheduling',
            model=config.task_model,
            use_vllm=use_admin_vllm,
            vllm_endpoint = config.vllm_url if use_admin_vllm else None
        )
        queue.append(OutpatientFirstScheduling(
            scheduling_strategy=config.schedule_task.scheduling_strategy,
            admin_staff_agent=admin_staff_agent,
            supervisor_agent=supervisor_agent if config.schedule_task.use_supervisor else None,
            schedule_cancellation_prob=config.schedule_cancellation_prob,
            request_early_schedule_prob=config.request_early_schedule_prob,
            max_feedback_number=config.schedule_task.max_feedback_number,
            fhir_integration=config.integration_with_fhir
        ))

    # Initialize agent test data
    is_file = os.path.isfile(config.agent_test_data)
    agent_test_data_files = [config.agent_test_data] if is_file else get_files(config.agent_test_data, ext='json')
    all_agent_test_data = [json_load(path) for path in agent_test_data_files]   # one agent test data per hospital

    # Execute agent tasks
    try:
        os.makedirs(args.output_dir, exist_ok=True)
        yaml_save(os.path.join(args.output_dir, 'args.yaml'), config)
        
        # Data per hospital
        for i, agent_test_data in enumerate(all_agent_test_data):
            agent_results, done_patients, dialog_results = dict(), dict(), dict()
            shuffle_agent_test_data(agent_test_data)
            environment = HospitalEnvironment(
                agent_test_data, 
                config.fhir_url, 
                config.fhir_max_connection_retries,
                config.booking_days_before_simulation
            )
            basename = os.path.splitext(os.path.basename(agent_test_data_files[i]))[0]
            save_path = os.path.join(args.output_dir, f'{basename}_result.json')
            d_save_path = os.path.join(args.output_dir, f'{basename}_dialog.json')
            log(f'{basename} simulation started..', color=True)
            
            # Resume the results and the virtual hospital environment
            if args.resume and os.path.exists(save_path):
                agent_test_data, agent_results, dialog_results, done_patients = resume_results(agent_test_data, save_path, d_save_path)
                environment.resume(agent_results)

            # Data per patient
            for j, (gt, test_data) in enumerate(agent_test_data['agent_data']):
                for task in queue:
                    if task.name in done_patients and gt['patient'] in done_patients[task.name]:
                        continue

                    result = task((gt, test_data), agent_test_data, agent_results, environment, args.verbose)
                    dialogs = result.pop('dialog')

                    # Append a single result 
                    agent_results.setdefault(task.name, {'gt': [], 'pred': [], 'status': [], 'status_code': [], 'trial': [], 'feedback': []})
                    for k in result:
                        agent_results[task.name][k] += result[k]
                    
                    if task.name == 'intake':
                        dialog_results[gt['patient']] = dialogs[0]
            
            # Logging the results
            for task_name, result in agent_results.items():
                correctness = result['status']
                status_code = result['status_code']
                accuracy = sum(correctness) / len(correctness)
                log(f'{basename} - {task_name} task results..', color=True)
                log(f'   - accuracy: {accuracy:.3f}, length: {len(correctness)}, status_code: {status_code}')
            
            json_save_fast(save_path, agent_results)
            if 'intake' in args.type:
                json_save_fast(d_save_path, dialog_results)
            
        log(f"Agent completed the tasks successfully", color=True)
    
    except Exception as e:
        if len(agent_results):
            json_save_fast(save_path, agent_results)
            if 'intake' in args.type:
                json_save_fast(d_save_path, dialog_results)
        log("Error occured while execute the tasks.", level='error')
        raise e
    


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True, help='Path to the configuration file')
    parser.add_argument('-t', '--type', type=str, required=True, nargs='+', choices=['intake', 'schedule'], help='Task types you want to execute (you can specify multiple)')
    parser.add_argument('-o', '--output_dir', type=str, required=True, help='Path to save agent test results')
    parser.add_argument('--resume', action='store_true', required=False, help='Continue the stopped processing')
    parser.add_argument('--verbose', action='store_true', required=False, help='Whether logging the each result or not')
    args = parser.parse_args()

    main(args)
