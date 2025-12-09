import os
import sys
import random
import numpy as np
from sconf import Config
from argparse import ArgumentParser
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from h_adminsim.tools import DataSynthesizer
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

    # Initialize data synthesizer
    data_synthesizer = DataSynthesizer(config)
    try:
        data, hospital_obj = data_synthesizer.synthesize(sanity_check=args.sanity_check)
        log(f"Data synthesis completed successfully", color=True)
    except Exception as e:
        log("Data synthesis failed.", level='error')
        raise e
    

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True, help='Path to the configuration file')
    parser.add_argument('--sanity_check', action='store_true', required=False, help='Check whether generated data and Hospital object compatible')
    args = parser.parse_args()

    main(args)
