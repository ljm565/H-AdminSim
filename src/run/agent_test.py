import os
import sys
from sconf import Config
from argparse import ArgumentParser
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from tasks import *
from utils import log
from utils.filesys_utils import json_load, get_files



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
    agent_results = dict()
    if 'department' in args.type:
        queue.append(AssignDepartment(config))
        agent_results['department'] = list()
    if 'schedule' in args.type:
        queue.append(AssignSchedule(config))
        agent_results['schedule'] = list()
    if 'fhir_resource' in args.type:
        queue.append(MakeFHIRResource(config))
        agent_results['fhir_resource'] = list()
    if 'fhire_api' in args.type:
        queue.append(MakeFHIRAPI(config))
        agent_results['fhire_api'] = list()

    # Initialize agent test data
    is_file = os.path.isfile(config.agent_test_data)
    agent_test_data_files = [config.agent_test_data] if is_file else get_files(config.agent_test_data, ext='json')
    all_agent_test_data = [json_load(path) for path in agent_test_data_files]   # one agent test data per hospital

    # Execute agent tasks
    try:
        for _ in range(len(queue)):
            task = queue.pop(0)
            log(f'{task.name} task started..', color=True)
            for agent_test_data in all_agent_test_data:
                results = task(agent_test_data, agent_results)
                correctness = [gt == pred for gt, pred in zip(results['gt'], results['pred'])]
                accuracy = sum(correctness) / len(correctness)
                agent_results[task.name].append(results)
                log(f'Accuracy: {accuracy:.3f}, length: {len(correctness)}')

        log(f"Agent completed the tasks successfully", color=True)
    except Exception as e:
        log("Error occured while execute the tasks.", level='error')
        raise e
    

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True, help='Path to the configuration file')
    parser.add_argument('-t', '--type', type=str, required=True, nargs='+', choices=['department', 'schedule', 'fhir_resource', 'fhir_api'], help='Task types you want to execute (you can specify multiple)')
    args = parser.parse_args()

    main(args)
