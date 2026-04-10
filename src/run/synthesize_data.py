import os
import sys
import random
import numpy as np
from sconf import Config
from argparse import ArgumentParser
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from h_adminsim.pipeline import DataGenerator
from h_adminsim.utils import log



def env_setup(config):
    random.seed(config.seed)
    np.random.seed(config.seed)


def load_config(config_path):
    config = Config(config_path)
    return config


def main(args):
    # Init config
    config = load_config(args.config)
    config.yaml_file = args.config
    
    # Init environment
    env_setup(config)

    # Generate data
    data_generator = DataGenerator(config=config)
    output = data_generator.build(
        sanity_check=args.sanity_check,
        convert_to_fhir=args.convert_to_fhir,
        build_agent_data=True,
        department_info_path=args.department_info_path,
        symptom_file_path=args.disease_symptom_file_path,
    )

    

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True, help='Path to the configuration file')
    parser.add_argument('--sanity_check', action='store_true', required=False, help='Check whether generated data and Hospital object compatible')
    parser.add_argument('--convert_to_fhir', action='store_true', required=False, help='Whether convert generated data to FHIR or not')
    parser.add_argument('--department_info_path', type=str, required=False, default=None,
        help='Path to the custom department information file. Recommend to use together with --disease_symptom_file_path.')
    parser.add_argument('--disease_symptom_file_path', type=str, required=False, default=None,
        help='Path to the custom disease-symptom mapping file. Recommend to use together with --department_info_path.')

    args = parser.parse_args()

    main(args)
