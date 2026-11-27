import os
import sys
import logging
import traceback
from sconf import Config
from typing import Optional
import patientsim.utils as pu
from argparse import ArgumentParser, Namespace
from concurrent.futures import ProcessPoolExecutor, as_completed
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from h_adminsim.pipeline.builder import build_simulator, prepare_run
from h_adminsim.utils import set_logging, LOGGING_NAME, log
from h_adminsim.utils.common_utils import replace_none_value
from h_adminsim.utils.filesys_utils import get_files



def bridge_patientsim() -> None:
    """
    Make the `patientsim` package log through the H-AdminSim logger.
    """
    main_logger = logging.getLogger(LOGGING_NAME)
    pu.LOGGER = main_logger
    pu.LOGGING_NAME = LOGGING_NAME


def init_worker_logging(output_dir: str, name: str) -> None:
    """
    Set up logging for a worker process, writing to ``<output_dir>/<name>_worker_<pid>.out``.

    Args:
        output_dir (str): Log directory.
        name (str): Log file prefix (model name).
    """
    pid = os.getpid()
    set_logging(
        LOGGING_NAME,
        verbose=True,
        log_file=os.path.join(output_dir, f"{name}_worker_{pid}.out")
    )
    bridge_patientsim()


def load_config(config_path: str) -> Config:
    """
    Load a YAML config, turning ``'None'`` strings into ``None``.

    Args:
        config_path (str): Path to the config file.

    Returns:
        Config: The loaded config.
    """
    config = Config(config_path)
    config = replace_none_value(config)
    return config


def simulate(config: Config, args: Namespace, single_file: Optional[str] = None) -> None:
    """
    Build the requested tasks and run the simulator on the agent data (one file, or the configured path).

    Args:
        config (Config): Simulation config.
        args (Namespace): Parsed CLI arguments.
        single_file (Optional[str], optional): Agent data file for this worker; None uses ``config.agent_test_data``. Defaults to None.
    """
    if args.logging_dir is not None:
        init_worker_logging(args.logging_dir, config.task_model.replace('/', '_'))

    # Build the tasks and run
    simulator = build_simulator(config, args.type, max_patients=args.max_patients)
    simulator.run(
        simulation_data_path=config.agent_test_data if single_file is None else single_file,
        output_dir=args.output_dir,
        resume=args.resume,
        verbose=args.verbose,
    )


def main(args: Namespace) -> None:
    """
    Prepare the output folder (config, pinned policies) and run the simulation, in parallel over agent data files.

    Args:
        args (Namespace): Parsed CLI arguments.

    Raises:
        RuntimeError: If any worker's simulation failed.
    """
    # Init config
    config = load_config(args.config)
    config.yaml_file = args.config
    prepare_run(config, args.type, args.output_dir)   # policies checked and copied into the run folder; args.yaml and workers use the copies
    
    # Multi-processing
    simulation_data_files = get_files(config.agent_test_data, ext='json')
    num_workers = min(getattr(args, "num_workers", os.cpu_count() or 1), len(simulation_data_files))
    if num_workers <= 1:
        bridge_patientsim()
        try:
            args.logging_dir = None
            simulate(config, args)
        except Exception as e:
            log(f'Exception occured: {e}', level='error')
            raise
    else:
        failures = list()
        if args.logging_dir is None:
            args.logging_dir = os.path.join(args.output_dir, 'logs')
            os.makedirs(args.logging_dir, exist_ok=True)
        with ProcessPoolExecutor(max_workers=num_workers) as ex:
            futures = {ex.submit(simulate, config, args, path): path for path in simulation_data_files}
            for fut in as_completed(futures):
                path = futures[fut]
                try:
                    fut.result()
                except Exception:
                    log(f"{path}\n{traceback.format_exc()}", level="error")
                    failures.append(path)
                    continue
            
        if failures:
            raise RuntimeError(f"{len(failures)} simulations failed: {failures}")



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True, help='Path to the configuration file')
    parser.add_argument('-t', '--type', type=str, required=True, nargs='+', choices=['first_visit_intake', 'first_visit_scheduling', 'follow_up_visit_scheduling'], help='Task types you want to execute (you can specify multiple)')
    parser.add_argument('-o', '--output_dir', type=str, required=True, help='Path to save agent test results')
    parser.add_argument('--resume', action='store_true', required=False, help='Continue the stopped processing')
    parser.add_argument('--verbose', action='store_true', required=False, help='Whether logging the each result or not')
    parser.add_argument('--num_workers', type=int, required=False, default=1, help='Whether execute the code with multi-processing or not')
    parser.add_argument('--logging_dir', type=str, required=False, default=None, help='Directory for log files (used only in multiprocessing mode)')
    parser.add_argument('--max_patients', type=int, required=False, default=None, help='Patients simulated per hospital (drawn after the seeded shuffle); default: all')
    args = parser.parse_args()

    main(args)
