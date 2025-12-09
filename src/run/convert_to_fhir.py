import os
import sys
from sconf import Config
from argparse import ArgumentParser
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from h_adminsim.tools import DataConverter
from h_adminsim.utils import log



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

    # Initialize data converter
    converter = DataConverter(config)
    try:
        output = converter(args.output_dir, args.sanity_check)
        log(f"Data conversion completed successfully", color=True)
    except Exception as e:
        log("Data conversion failed.", level='error')
        raise e
    

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True, help='Path to the configuration file')
    parser.add_argument('-o', '--output_dir', type=str, required=True, help='Path to save FHIR-converted data')
    parser.add_argument('--sanity_check', action='store_true', required=False, help='Check whether converted FHIR resources are unique')
    args = parser.parse_args()

    main(args)
