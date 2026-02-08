import os
import sys
import logging
from sconf import Config
import patientsim.utils as pu
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor, as_completed
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from h_adminsim import SupervisorAgent
from h_adminsim.task.agent_task import *
from h_adminsim.pipeline import Simulator
from h_adminsim.utils import set_logging, LOGGING_NAME
from h_adminsim.utils.filesys_utils import yaml_save, get_files



def bridge_patientsim():
        main_logger = logging.getLogger(LOGGING_NAME)
        pu.LOGGER = main_logger
        pu.LOGGING_NAME = LOGGING_NAME


def init_worker_logging(output_dir: str, name: str):
    pid = os.getpid()
    set_logging(
        LOGGING_NAME,
        verbose=True,
        log_file=os.path.join(output_dir, f"{name}_worker_{pid}.out")
    )
    bridge_patientsim()


def load_config(config_path: str):
    config = Config(config_path)
    return config


def simulate(config, args, single_file=None):
    if args.logging_dir is not None:
        init_worker_logging(args.logging_dir, config.task_model.replace('/', '_'))

    # Initialize tasks
    intake_task, scheduling_task = None, None
    if 'intake' in args.type:
        use_vllm = False if any(m in config.supervisor_model.lower() for m in ['gpt', 'gemini']) else True
        supervisor_agent = SupervisorAgent(
            target_task='first_outpatient_intake',
            model=config.supervisor_model,
            use_vllm=use_vllm,
            vllm_endpoint = config.vllm_url if use_vllm else None
        )
        use_vllm = False if any(m in config.task_model.lower() for m in ['gpt', 'gemini']) else True
        intake_task = OutpatientFirstIntake(
            patient_model=config.task_model,
            admin_staff_model=config.task_model,
            supervisor_agent=supervisor_agent if config.outpatient_intake.use_supervisor else None,
            intake_max_inference=config.outpatient_intake.intake_max_inference,
            patient_vllm_endpoint=config.vllm_url if use_vllm else None,
            admin_staff_vllm_endpoint=config.vllm_url if use_vllm else None
        )
    if 'schedule' in args.type:
        use_vllm = False if any(m in config.task_model.lower() for m in ['gpt', 'gemini']) else True
        scheduling_task = OutpatientFirstScheduling(
            patient_model=config.task_model,
            admin_staff_model=config.task_model,
            schedule_cancellation_prob=config.schedule_cancellation_prob,
            request_early_schedule_prob=config.request_early_schedule_prob,
            fhir_integration=config.integration_with_fhir,
            scheduling_strategy=config.schedule_task.scheduling_strategy,
            patient_vllm_endpoint=config.vllm_url if use_vllm else None,
            admin_staff_vllm_endpoint=config.vllm_url if use_vllm else None
        )

    # Run simulations
    simulator = Simulator(
        intake_task=intake_task,
        scheduling_task=scheduling_task,
        simulation_start_day_before=config.booking_days_before_simulation,
        fhir_integration=config.integration_with_fhir,
        fhir_url=config.fhir_url,
        fhir_max_connection_retries=config.fhir_max_connection_retries,
        random_seed=config.seed,
    )

    simulator.run(
        simulation_data_path=config.agent_test_data if single_file is None else single_file,
        output_dir=args.output_dir,
        resume=args.resume,
        verbose=args.verbose,
    )


def main(args):
    # Init config
    config = load_config(args.config)
    config.yaml_file = args.config
    os.makedirs(args.output_dir, exist_ok=True)
    yaml_save(os.path.join(args.output_dir, 'args.yaml'), config)
    
    # Multi-processing
    simulation_data_files = get_files(config.agent_test_data, ext='json')
    num_workers = min(getattr(args, "num_workers", os.cpu_count() or 1), len(simulation_data_files))
    if num_workers <= 1:
        bridge_patientsim()
        try:
            args.logging_dir = None
            simulate(config, args)
        except:
            raise
    else:
        try:
            if args.logging_dir is None:
                args.logging_dir = os.path.join(args.output_dir, 'logs')
                os.makedirs(args.logging_dir, exist_ok=True)
            
            with ProcessPoolExecutor(max_workers=num_workers) as ex:
                futures = [
                    ex.submit(
                        simulate, 
                        config,
                        args,
                        path
                    ) for path in simulation_data_files
                ]

                for fut in as_completed(futures):
                    fut.result()
        except:
            raise



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True, help='Path to the configuration file')
    parser.add_argument('-t', '--type', type=str, required=True, nargs='+', choices=['intake', 'schedule'], help='Task types you want to execute (you can specify multiple)')
    parser.add_argument('-o', '--output_dir', type=str, required=True, help='Path to save agent test results')
    parser.add_argument('--resume', action='store_true', required=False, help='Continue the stopped processing')
    parser.add_argument('--verbose', action='store_true', required=False, help='Whether logging the each result or not')
    parser.add_argument('--num_workers', type=int, required=False, default=1, help='Whether execute the code with multi-processing or not')
    parser.add_argument('--logging_dir', type=str, required=False, default=None, help='Directory for log files (used only in multiprocessing mode)')
    args = parser.parse_args()

    main(args)
