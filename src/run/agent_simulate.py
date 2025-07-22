import os
import sys
import random
import numpy as np
from sconf import Config
from copy import deepcopy
from argparse import ArgumentParser
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from tasks import *
from registry.environment import HospitalEnvironment
from utils import log
from utils.filesys_utils import json_load, json_save_fast, yaml_save, get_files



def env_setup(config):
    random.seed(config.seed)
    np.random.seed(config.seed)


def load_config(config_path):
    config = Config(config_path)
    return config


def ordering_agent_test_data(agent_test_data: dict):
    """
    Order the agent test data by the schedule start time.

    Args:
        agent_test_data (dict): An agent test data to simulate a hospital environmnet.
    """
    agent_data = agent_test_data['agent_data']
    agent_data.sort(key=lambda x: x[0]['schedule']['time'])     # In-place logic


def main(args):
    # Init config
    config = load_config(args.config)
    config.yaml_file = args.config
    
    # Init environment
    env_setup(config)

    # Initialize tasks
    queue = list()
    if 'department' in args.type:
        queue.append(AssignDepartment(config))
    if 'schedule' in args.type:
        queue.append(AssignSchedule(config))
    if 'fhir_resource' in args.type:
        queue.append(MakeFHIRResource(config))
    if 'fhir_api' in args.type:
        queue.append(MakeFHIRAPI(config))

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
            agent_results = dict()
            ordering_agent_test_data(agent_test_data)
            environment = HospitalEnvironment(agent_test_data)
            basename = os.path.splitext(os.path.basename(agent_test_data_files[i]))[0]
            save_path = os.path.join(args.output_dir, f'{basename}_result.json')
            
            # Skip if the result already exits
            if args.skip_saved_file and os.path.exists(save_path):
                continue
            
            # Data per patient
            for gt, test_data in agent_test_data['agent_data']:
                for task in queue:
                    result = task((gt, test_data), agent_test_data, agent_results, environment)
                    
                    # Append a single result 
                    agent_results.setdefault(task.name, {'gt': [], 'pred': [], 'status': [], 'status_code': []})
                    for k in result:
                        agent_results[task.name][k] += result[k]

                    if task.name == 'schedule':
                        idx = 0
                        for j, s in enumerate(agent_results[task.name]['status']):
                            if s:
                                agent_results[task.name]['pred'][j] = deepcopy(environment.patient_schedules[idx])
                                idx += 1
                                
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
        log("Error occured while execute the tasks.", level='error')
        raise e
    


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True, help='Path to the configuration file')
    parser.add_argument('-t', '--type', type=str, required=True, nargs='+', choices=['department', 'schedule', 'fhir_resource', 'fhir_api'], help='Task types you want to execute (you can specify multiple)')
    parser.add_argument('-o', '--output_dir', type=str, required=True, help='Path to save agent test results')
    parser.add_argument('-s', '--skip_saved_file', action='store_true', required=False, help='Skip inference if results already exit')
    args = parser.parse_args()

    main(args)
