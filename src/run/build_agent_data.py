import os
import sys
import random
import numpy as np
from sconf import Config
from argparse import ArgumentParser
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from tools import AgentTestBuilder
from utils import log



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

    # Initialize data converter
    builder = AgentTestBuilder(config)
    try:
        output = builder(args.output_dir)
        log(f"Agent data generation completed successfully", color=True)
    except Exception as e:
        log("Agent data generation failed.", level='error')
        raise e
    

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True, help='Path to the configuration file')
    parser.add_argument('-o', '--output_dir', type=str, required=True, help='Path to save agent test data')
    args = parser.parse_args()

    main(args)
