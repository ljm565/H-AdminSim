import os
import sys
import random
import numpy as np
from sconf import Config
from argparse import ArgumentParser
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from h_adminsim import AdminStaffAgent, SupervisorAgent
from h_adminsim.task.agent_task import *
from h_adminsim.task.fhir_manager import FHIRManager
from h_adminsim.pipeline import DataGenerator, Simulator
from h_adminsim.utils import log
from h_adminsim.utils.filesys_utils import yaml_save




def env_setup(config, resume):
    random.seed(config.seed)
    np.random.seed(config.seed)

    # Delete Patient and Appointment resources when starting a simulation
    if config.integration_with_fhir and not resume:
        fhir_manager = FHIRManager(config.fhir_url)
        appointment_entries = fhir_manager.read_all('Appointment')
        patient_entries = fhir_manager.read_all('Patient')
        fhir_manager.delete_all(appointment_entries, verbose=False)
        fhir_manager.delete_all(patient_entries, verbose=False)


def load_config(config_path):
    config = Config(config_path)
    return config


def main(args):
    # Init config
    d_config = load_config(args.data_config) if args.data_config else None
    s_config = load_config(args.simulation_config)
    s_config.yaml_file = args.simulation_config
    

    # Init environment
    env_setup(s_config, args.resume)


    # Generate data for the simulation
    data_generator = DataGenerator() if not d_config else DataGenerator(config=d_config)
    output = data_generator.build(convert_to_fhir=True)

    if args.upload_data_to_fhir:
        data_generator.upload_to_fhir(
            fhir_data_dir=data_generator.save_dir / 'fhir_data',
            fhir_url=s_config.fhir_url
        )

    log('Data has been successfully generated!', color=True)


    # Simulation
    output_dir = data_generator.save_dir / 'simulation_results'
    s_config.agent_test_data = str(data_generator.save_dir / 'agent_data')
    os.makedirs(output_dir, exist_ok=True)
    yaml_save(os.path.join(output_dir, 'args.yaml'), s_config)
    
    intake_task, scheduling_task = None, None
    if 'intake' in args.type:
        use_vllm = False if any(m in s_config.supervisor_model.lower() for m in ['gpt', 'gemini']) else True
        supervisor_agent = SupervisorAgent(
            target_task='first_outpatient_intake',
            model=s_config.supervisor_model,
            use_vllm=use_vllm,
            vllm_endpoint = s_config.vllm_url if use_vllm else None
        )
        use_vllm = False if any(m in s_config.task_model.lower() for m in ['gpt', 'gemini']) else True
        intake_task = OutpatientFirstIntake(
            patient_model=s_config.task_model,
            admin_staff_model=s_config.task_model,
            supervisor_agent=supervisor_agent if s_config.outpatient_intake.use_supervisor else None,
            intake_max_inference=s_config.intake_max_inference,
            patient_vllm_endpoint=s_config.vllm_url if use_vllm else None,
            admin_staff_vllm_endpoint=s_config.vllm_url if use_vllm else None
        )

    if 'schedule' in args.type:
        use_vllm = False if any(m in s_config.supervisor_model.lower() for m in ['gpt', 'gemini']) else True
        supervisor_agent = SupervisorAgent(
            target_task='first_outpatient_scheduling',
            model=s_config.supervisor_model,
            use_vllm=use_vllm,
            vllm_endpoint = s_config.vllm_url if use_vllm else None
        )
        use_vllm = False if any(m in s_config.task_model.lower() for m in ['gpt', 'gemini']) else True
        scheduling_task = OutpatientFirstScheduling(
            patient_model=s_config.task_model,
            admin_staff_model=s_config.task_model,
            scheduling_strategy=s_config.schedule_task.scheduling_strategy,
            supervisor_agent=supervisor_agent if s_config.schedule_task.use_supervisor else None,
            schedule_cancellation_prob=s_config.schedule_cancellation_prob,
            request_early_schedule_prob=s_config.request_early_schedule_prob,
            max_feedback_number=s_config.schedule_task.max_feedback_number,
            fhir_integration=s_config.integration_with_fhir,
            patient_vllm_endpoint=s_config.vllm_url if use_vllm else None,
            admin_staff_vllm_endpoint=s_config.vllm_url if use_vllm else None
        )
    
    simulator = Simulator(
        intake_task=intake_task,
        scheduling_task=scheduling_task,
        simulation_start_day_before=s_config.booking_days_before_simulation,
        fhir_integration=s_config.integration_with_fhir,
        fhir_url=s_config.fhir_url,
        fhir_max_connection_retries=s_config.fhir_max_connection_retries,
        random_seed=s_config.seed
    )
    
    log('Simulation started!', color=True)
    simulator.run(
        simulation_data_path=s_config.agent_test_data,
        output_dir=output_dir,
        resume=args.resume,
        verbose=args.verbose
    )
    
    

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_config', type=str, required=False, default=None, help='Path to the data synthesis configuration file')
    parser.add_argument('--simulation_config', type=str, required=True, help='Path to the simulation configuration file')
    parser.add_argument('--type', type=str, required=True, nargs='+', choices=['intake', 'schedule'], help='Task types you want to execute (you can specify multiple)')
    parser.add_argument('--resume', action='store_true', required=False, help='Continue the stopped processing')
    parser.add_argument('--verbose', action='store_true', required=False, help='Whether logging the each result or not')
    parser.add_argument('--upload_data_to_fhir', action='store_true', required=False, help='Whether to upload synthetic data to FHIR')
    args = parser.parse_args()

    main(args)
