import os
import sys
import random
import numpy as np
from sconf import Config
from argparse import ArgumentParser
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from tools import FHIRManager
from utils import log
from utils.random_utils import random_uuid
from utils.filesys_utils import json_load, get_files


def env_setup(config):
    random.seed(config.seed)
    np.random.seed(config.seed)


def load_config(config_path):
    config = Config(config_path)
    return config


# def main(args):
#     # Init config
#     config = load_config(args.config)
#     config.yaml_file = args.config
    
#     # Init environment
#     env_setup(config)

#     # Execute CRUD operation
#     fhir_manager = FHIRManager(config)
    
#     if args.mode == 'create':
#         is_file = os.path.isfile(config.create_data_path)
#         files = [config.create_data_path] if is_file else get_files(config.create_data_path, ext='json')
        
#         for file in files:
#             resource_data = json_load(file)
#             resource_type = resource_data.get('resourceType')
#             if 'id' not in resource_data:
#                 resource_data['id'] = random_uuid(args.is_develop)
            
#             response = fhir_manager.create(resource_type, resource_data)
#             if  200 <= response.status_code < 300:
#                 log(f"Created {resource_type} with ID {response.json().get('id')}")

#     elif args.mode == 'read':
#         if not args.id or not args.resource_type:
#             log("ID and resource type are required for read operation", level='error')
#             raise ValueError('ID and resource type are required for read operation')
    
#         response = fhir_manager.read(args.resource_type, args.id)
#         if  200 <= response.status_code < 300:
#             log(f"Read {args.resource_type} with ID {args.id}")

#     elif args.mode == 'update':
#         if not args.id:
#             log("ID is required for update operation", level='error')
#             raise ValueError('ID is required for update operation')

#         resource_data = json_load(args.update_data_path) if args.update_data_path else {}
#         resource_type = resource_data.get('resourceType')
#         resource_data['id'] = args.id
#         response = fhir_manager.update(resource_type, args.id, resource_data)
#         if  200 <= response.status_code < 300:
#             log(f"Updated {args.resource_type} with ID {args.id}")
        
#     elif args.mode == 'delete':
#         if not args.id or not args.resource_type:
#             log("ID and resource type are required for delete operation", level='error')
#             raise ValueError('ID and resource type are required for delete operation')
        
#         response = fhir_manager.delete(args.resource_type, args.id)
#         if  200 <= response.status_code < 300:
#             log(f"Deleted {args.resource_type} with ID {args.id}")
    


# if __name__ == '__main__':
#     parser = ArgumentParser()
#     parser.add_argument('-c', '--config', type=str, required=True, help='Path to the configuration file')
#     parser.add_argument('-m', '--mode', type=str, required=True, choices=['create', 'read', 'update', 'delete'], help='CRUD operation mode')
#     parser.add_argument('-d', '--is_develop', action='store_true', required=False, help='Enable development mode for controlled random UUID generation')
#     parser.add_argument('--id', type=str, required=False, help='Resource ID for read, update, or delete operations')
#     parser.add_argument('--resource_type', type=str, required=False, help='Resource type for read, update, or delete operations')
#     parser.add_argument('--update_data_path', type=str, required=False, help='Data path for update operation (JSON file)')
#     args = parser.parse_args()

#     main(args)
