import os
import sys
import random
import numpy as np
from sconf import Config
from argparse import ArgumentParser
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from h_adminsim.agent import SupervisorAgent
from h_adminsim.task.opfv_task import OutpatientFirstIntake, OutpatientFirstScheduling
from h_adminsim.task.opfu_task import OutpatientFollowUpScheduling
from h_adminsim.task.fhir_manager import FHIRManager
from h_adminsim.pipeline import DataGenerator, Simulator, HospitalMAS
from h_adminsim.policy import pin_policy_config
from h_adminsim.utils import log
from h_adminsim.utils.mas_utils import init_mas_system
from h_adminsim.utils.common_utils import replace_none_value
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
    config = replace_none_value(config)
    return config


def main(args):
    # Init config
    d_config = load_config(args.data_config) if args.data_config else None
    s_config = load_config(args.simulation_config)
    s_config.yaml_file = args.simulation_config
    

    # Init environment
    env_setup(s_config, args.resume)


    # Generate data for the simulation
    data_synthesis_task = ['first_visit', 'follow_up_visit']
    data_generator = DataGenerator(task=data_synthesis_task) if not d_config \
        else DataGenerator(task=data_synthesis_task, config=d_config)
    output = data_generator.build(
        convert_to_fhir=True,
        department_info_path=args.department_info_path,
        symptom_file_path=args.disease_symptom_file_path,
    )

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
    pin_policy_config(s_config.schedule_task, output_dir)   # policy files copied into the run folder; args.yaml and workers use the copies
    yaml_save(os.path.join(output_dir, 'args.yaml'), s_config)
    
    # Initialize the admin staff multi-agent system (orchestrator + one staff agent per task)
    mas_structure = init_mas_system(
        orchestrator_model=s_config.task_model,
        orchestrator_vllm_endpoint=s_config.vllm_url,
        first_visit_intake_model=s_config.task_model if 'first_visit_intake' in args.type else None,
        first_visit_intake_vllm_endpoint=s_config.vllm_url if 'first_visit_intake' in args.type else None,
        first_visit_scheduling_model=s_config.task_model if 'first_visit_scheduling' in args.type else None,
        first_visit_scheduling_vllm_endpoint=s_config.vllm_url if 'first_visit_scheduling' in args.type else None,
        follow_up_visit_scheduling_model=s_config.task_model if 'follow_up_visit_scheduling' in args.type else None,
        follow_up_visit_scheduling_vllm_endpoint=s_config.vllm_url if 'follow_up_visit_scheduling' in args.type else None,
        intake_max_inference=s_config.outpatient_intake.intake_max_inference,
    )
    staff_mas = HospitalMAS(mas_structure)

    task = dict()
    if 'first_visit_intake' in args.type:
        use_vllm = False if any(m in s_config.supervisor_model.lower() for m in ['gpt', 'gemini']) else True
        supervisor_agent = SupervisorAgent(
            target_task='first_visit_intake',
            model=s_config.supervisor_model,
            use_vllm=use_vllm,
            vllm_endpoint=s_config.vllm_url if use_vllm else None,
            temperature=0 if not 'gpt-5' in s_config.supervisor_model.lower() else 1
        )
        use_vllm = False if any(m in s_config.task_model.lower() for m in ['gpt', 'gemini']) else True
        _task = OutpatientFirstIntake(
            patient_model=s_config.task_model,
            admin_staff_mas=staff_mas,
            supervisor_agent=supervisor_agent if s_config.outpatient_intake.use_supervisor else None,
            intake_max_inference=s_config.outpatient_intake.intake_max_inference,
            patient_vllm_endpoint=s_config.vllm_url if use_vllm else None,
        )
        task[_task.name] = _task
    if 'first_visit_scheduling' in args.type:
        use_vllm = False if any(m in s_config.task_model.lower() for m in ['gpt', 'gemini']) else True
        _task = OutpatientFirstScheduling(
            patient_model=s_config.task_model,
            admin_staff_mas=staff_mas,
            schedule_cancellation_prob=s_config.schedule_task.opfv_schedule_cancellation_prob,
            request_early_schedule_prob=s_config.schedule_task.opfv_request_early_schedule_prob,
            fhir_integration=s_config.integration_with_fhir,
            scheduling_strategy=s_config.schedule_task.scheduling_strategy,
            patient_vllm_endpoint=s_config.vllm_url if use_vllm else None,
        )
        task[_task.name] = _task
    if 'follow_up_visit_scheduling' in args.type:
        use_vllm = False if any(m in s_config.task_model.lower() for m in ['gpt', 'gemini']) else True
        _task = OutpatientFollowUpScheduling(
            patient_model=s_config.task_model,
            admin_staff_mas=staff_mas,
            schedule_cancellation_prob=s_config.schedule_task.opfu_schedule_cancellation_prob,
            request_early_schedule_prob=s_config.schedule_task.opfu_request_early_schedule_prob,
            fhir_integration=s_config.integration_with_fhir,
            scheduling_strategy=s_config.schedule_task.scheduling_strategy,
            patient_vllm_endpoint=s_config.vllm_url if use_vllm else None,
            policy_config=s_config.schedule_task.get('policy_config'),
        )
        task[_task.name] = _task

    # Run simulations
    simulator = Simulator(
        task=task,
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
    parser.add_argument('--type', type=str, required=True, nargs='+', choices=['first_visit_intake', 'first_visit_scheduling', 'follow_up_visit_scheduling'], help='Task types you want to execute (you can specify multiple)')
    parser.add_argument('--resume', action='store_true', required=False, help='Continue the stopped processing')
    parser.add_argument('--verbose', action='store_true', required=False, help='Whether logging the each result or not')
    parser.add_argument('--upload_data_to_fhir', action='store_true', required=False, help='Whether to upload synthetic data to FHIR')
    parser.add_argument('--department_info_path', type=str, required=False, default=None,
        help='Path to the custom department information file. Recommend to use together with --disease_symptom_file_path.')
    parser.add_argument('--disease_symptom_file_path', type=str, required=False, default=None,
        help='Path to the custom disease-symptom mapping file. Recommend to use together with --department_info_path.')
    args = parser.parse_args()

    main(args)
