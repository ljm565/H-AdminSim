import os
from typing import Optional

from sconf import Config

from h_adminsim.agent.supervisor import SupervisorAgent
from h_adminsim.pipeline.h_mas import HospitalMAS
from h_adminsim.pipeline.simulation import Simulator
from h_adminsim.policy.builder import load_policy_config, pin_policy_config
from h_adminsim.task.opfu_task import OutpatientFollowUpScheduling
from h_adminsim.task.opfv_task import OutpatientFirstIntake, OutpatientFirstScheduling
from h_adminsim.utils.filesys_utils import yaml_save
from h_adminsim.utils.mas_utils import init_mas_system
from h_adminsim.utils.policy_utils import check_policy_requirements



def _uses_vllm(model: str) -> bool:
    """
    Whether a model is served through vLLM (anything but the GPT and Gemini APIs).

    Args:
        model (str): Model name.

    Returns:
        bool: True unless the name contains ``gpt`` or ``gemini``.
    """
    return not any(m in model.lower() for m in ['gpt', 'gemini'])


def prepare_run(config: Config, task_types: list[str], output_dir: str) -> None:
    """
    Prepare a run's output folder: check the follow-up policies against the agent data, copy them into the folder
    (the config then points at the copies), and save the config as ``args.yaml``.

    Args:
        config (Config): Simulation config (as ``config/agent_simulate.yaml``); its policy list is rewritten.
        task_types (list[str]): Task types to run.
        output_dir (str): The run's output folder.
    """
    os.makedirs(output_dir, exist_ok=True)
    if 'follow_up_visit_scheduling' in task_types:   # policies are used by the follow-up scheduling task only
        check_policy_requirements(load_policy_config(config.schedule_task.get('policy_config')), config.agent_test_data)
        pin_policy_config(config.schedule_task, output_dir)
    yaml_save(os.path.join(output_dir, 'args.yaml'), config)


def build_simulator(config: Config, task_types: list[str], max_patients: Optional[int] = None) -> Simulator:
    """
    The simulator of a simulation config: the staff multi-agent system and the requested tasks.

    Args:
        config (Config): Simulation config (as ``config/agent_simulate.yaml``).
        task_types (list[str]): Task types to run: ``first_visit_intake``, ``first_visit_scheduling`` and/or
                                ``follow_up_visit_scheduling``.
        max_patients (Optional[int], optional): Patients simulated per hospital (see `Simulator`). Defaults to None.

    Returns:
        Simulator: The simulator; run it on ``config.agent_test_data``.
    """
    task = dict()
    mas_structure = init_mas_system(
        orchestrator_model=config.task_model,
        orchestrator_vllm_endpoint=config.vllm_url,
        first_visit_intake_model=config.task_model if 'first_visit_intake' in task_types else None,
        first_visit_intake_vllm_endpoint=config.vllm_url if 'first_visit_intake' in task_types else None,
        first_visit_scheduling_model=config.task_model if 'first_visit_scheduling' in task_types else None,
        first_visit_scheduling_vllm_endpoint=config.vllm_url if 'first_visit_scheduling' in task_types else None,
        follow_up_visit_scheduling_model=config.task_model if 'follow_up_visit_scheduling' in task_types else None,
        follow_up_visit_scheduling_vllm_endpoint=config.vllm_url if 'follow_up_visit_scheduling' in task_types else None,
        intake_max_inference=config.outpatient_intake.intake_max_inference,
    )
    staff_mas = HospitalMAS(mas_structure)
    patient_vllm_endpoint = config.vllm_url if _uses_vllm(config.task_model) else None
    if 'first_visit_intake' in task_types:
        supervisor_vllm = _uses_vllm(config.supervisor_model)
        supervisor_agent = SupervisorAgent(
            target_task='first_visit_intake',
            model=config.supervisor_model,
            use_vllm=supervisor_vllm,
            vllm_endpoint=config.vllm_url if supervisor_vllm else None,
            temperature=0 if not 'gpt-5' in config.supervisor_model.lower() else 1
        )
        _task = OutpatientFirstIntake(
            patient_model=config.task_model,
            admin_staff_mas=staff_mas,
            supervisor_agent=supervisor_agent if config.outpatient_intake.use_supervisor else None,
            intake_max_inference=config.outpatient_intake.intake_max_inference,
            patient_vllm_endpoint=patient_vllm_endpoint,
        )
        task[_task.name] = _task
    if 'first_visit_scheduling' in task_types:
        _task = OutpatientFirstScheduling(
            patient_model=config.task_model,
            admin_staff_mas=staff_mas,
            schedule_cancellation_prob=config.schedule_task.opfv_schedule_cancellation_prob,
            request_early_schedule_prob=config.schedule_task.opfv_request_early_schedule_prob,
            fhir_integration=config.integration_with_fhir,
            scheduling_strategy=config.schedule_task.scheduling_strategy,
            patient_vllm_endpoint=patient_vllm_endpoint,
        )
        task[_task.name] = _task
    if 'follow_up_visit_scheduling' in task_types:
        _task = OutpatientFollowUpScheduling(
            patient_model=config.task_model,
            admin_staff_mas=staff_mas,
            schedule_cancellation_prob=config.schedule_task.opfu_schedule_cancellation_prob,
            request_early_schedule_prob=config.schedule_task.opfu_request_early_schedule_prob,
            fhir_integration=config.integration_with_fhir,
            scheduling_strategy=config.schedule_task.scheduling_strategy,
            patient_vllm_endpoint=patient_vllm_endpoint,
            policy_config=config.schedule_task.get('policy_config'),
        )
        task[_task.name] = _task

    return Simulator(
        task=task,
        simulation_start_day_before=config.booking_days_before_simulation,
        fhir_integration=config.integration_with_fhir,
        fhir_url=config.fhir_url,
        fhir_max_connection_retries=config.fhir_max_connection_retries,
        random_seed=config.seed,
        max_patients=max_patients,
    )
