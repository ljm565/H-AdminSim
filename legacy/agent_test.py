import os
import sys
from sconf import Config
from argparse import ArgumentParser
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from tasks import *
from utils import log
from utils.filesys_utils import json_load, json_save_fast, yaml_save, get_files



def env_setup(config):
    pass


def load_config(config_path):
    config = Config(config_path)
    return config


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
        for i, agent_test_data in enumerate(all_agent_test_data):
            agent_results = dict()
            basename = os.path.splitext(os.path.basename(agent_test_data_files[i]))[0]
            save_path = os.path.join(args.output_dir, f'{basename}_result.json')
            
            # Skip if the result already exits
            if args.skip_saved_file and os.path.exists(save_path):
                continue

            for task in queue:
                log(f'{basename} - {task.name} task started..', color=True)
                results = task(agent_test_data, agent_results)
                
                correctness = results['status']
                status_code = results['status_code']
                accuracy = sum(correctness) / len(correctness)
                log(f'Result - accuracy: {accuracy:.3f}, length: {len(correctness)}, status_code: {status_code}')
                
                agent_results[task.name] = results

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
