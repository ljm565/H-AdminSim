import os
import sys
import random
import numpy as np
from sconf import Config
from argparse import ArgumentParser
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from tasks import *
from registry.environment import HospitalEnvironment
from utils import log
from utils.filesys_utils import json_load, json_save_fast, yaml_save, get_files



def env_setup(config, is_continue):
    random.seed(config.seed)
    np.random.seed(config.seed)

    # Delete Patient and Appointment resources when starting a simulation
    if config.integration_with_fhir and not is_continue:
        fhir_manager = FHIRManager(config)
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
    

def main(args):
    # Init config
    config = load_config(args.config)
    config.yaml_file = args.config
    
    # Init environment
    env_setup(config, args.continuing)

    # Initialize tasks
    queue = list()
    if 'intake' in args.type:
        queue.append(OutpatientIntake(config))
    if 'schedule' in args.type:
        queue.append(AssignSchedule(config))

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
            agent_results, done_length = dict(), 0
            shuffle_agent_test_data(agent_test_data)
            environment = HospitalEnvironment(config, agent_test_data)
            basename = os.path.splitext(os.path.basename(agent_test_data_files[i]))[0]
            save_path = os.path.join(args.output_dir, f'{basename}_result.json')
            log(f'{basename} simulation started..', color=True)
            
            # Skip if the result already exits
            if args.skip_saved_file and os.path.exists(save_path):
                continue

            if args.continuing and os.path.exists(save_path):
                agent_results = json_load(save_path)
                key = list(agent_results.keys())[0]
                done_length = len(agent_results[key]['gt'])

                if 'schedule' in agent_results:
                    fixed_schedule = agent_test_data['doctor']
                    for status, pred in zip(agent_results['schedule']['status'], agent_results['schedule']['pred']):
                        if status:
                            fixed_schedule[pred['attending_physician']]['schedule'][pred['date']].append(pred['schedule'])
                            fixed_schedule[pred['attending_physician']]['schedule'][pred['date']].sort()
            
            # Data per patient
            for j, (gt, test_data) in enumerate(agent_test_data['agent_data']):
                if args.continuing and j < done_length:
                    continue

                for task in queue:
                    result = task((gt, test_data), agent_test_data, agent_results, environment)
                    
                    # Append a single result 
                    agent_results.setdefault(task.name, {'gt': [], 'pred': [], 'status': [], 'status_code': []})
                    for k in result:
                        agent_results[task.name][k] += result[k]
                                  
            # Logging the results
            for task_name, result in agent_results.items():
                correctness = result['status']
                status_code = result['status_code']
                accuracy = sum(correctness) / len(correctness)
                log(f'{basename} - {task_name} task results..', color=True)
                log(f'   - accuracy: {accuracy:.3f}, length: {len(correctness)}, status_code: {status_code}')
            
            json_save_fast(save_path, agent_results)
            
        log(f"Agent completed the tasks successfully", color=True)
    
    except Exception as e:
        json_save_fast(save_path, agent_results)
        log("Error occured while execute the tasks.", level='error')
        raise e
    


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True, help='Path to the configuration file')
    parser.add_argument('-t', '--type', type=str, required=True, nargs='+', choices=['intake', 'schedule'], help='Task types you want to execute (you can specify multiple)')
    parser.add_argument('-o', '--output_dir', type=str, required=True, help='Path to save agent test results')
    parser.add_argument('--skip_saved_file', action='store_true', required=False, help='Skip inference if results already exsist')
    parser.add_argument('--continuing', action='store_true', required=False, help='Continue the stopped processing')
    args = parser.parse_args()

    main(args)
