from typing import Optional

from h_adminsim.registry.phrases import AGENT_DESCRIPTION
from h_adminsim.agent.orchestrator import OrchestratorAgent
from h_adminsim.agent.intake_admin_staff import IntakeAdminStaffAgent
from h_adminsim.agent.scheduling_admin_staff import SchedulingAdminStaffAgent
from h_adminsim.utils import colorstr



def init_task_models(model: str, 
                     vllm_endpoint: Optional[str] = None) -> tuple[str, Optional[str], bool]:
    """
    Resolve a task model's vLLM settings: Gemini/GPT models use their API, any other model needs ``vllm_endpoint``.

    Args:
        model (str): The model name.
        vllm_endpoint (Optional[str], optional): The vLLM endpoint URL. Defaults to None.
    
    Returns:
        tuple[str, Optional[str], bool]: ``(model, vLLM endpoint or None for API models, use_vllm)``.
    """
    if any(keyword in model.lower() for keyword in ['gemini', 'gpt']):
        return model, None, False
    else:
        assert vllm_endpoint is not None, colorstr("red", 'vLLM endpoint must be provided for non-Gemini/GPT models.')
        return model, vllm_endpoint, True
    


def init_mas_system(orchestrator_model: str,
                    orchestrator_vllm_endpoint: Optional[str] = None,
                    first_visit_intake_model: Optional[str] = None,
                    first_visit_intake_vllm_endpoint: Optional[str] = None,
                    first_visit_scheduling_model: Optional[str] = None,
                    first_visit_scheduling_vllm_endpoint: Optional[str] = None,
                    follow_up_visit_scheduling_model: Optional[str] = None,
                    follow_up_visit_scheduling_vllm_endpoint: Optional[str] = None,
                    intake_max_inference: int = 5,) -> dict:
    """
    Build the multi-agent system: an orchestrator plus one sub-agent per task whose model is given.

    Args:
        orchestrator_model (str): Orchestrator model name.
        orchestrator_vllm_endpoint (Optional[str], optional): Orchestrator vLLM endpoint (non-API models). Defaults to None.
        first_visit_intake_model (Optional[str], optional): First-visit intake model; None omits the agent. Defaults to None.
        first_visit_intake_vllm_endpoint (Optional[str], optional): Its vLLM endpoint. Defaults to None.
        first_visit_scheduling_model (Optional[str], optional): First-visit scheduling model; None omits the agent. Defaults to None.
        first_visit_scheduling_vllm_endpoint (Optional[str], optional): Its vLLM endpoint. Defaults to None.
        follow_up_visit_scheduling_model (Optional[str], optional): Follow-up scheduling model; None omits the agent. Defaults to None.
        follow_up_visit_scheduling_vllm_endpoint (Optional[str], optional): Its vLLM endpoint. Defaults to None.
        intake_max_inference (int, optional): Maximum inferences for the intake agent. Defaults to 5.

    Returns:
        dict: ``{'orchestrator': {'agent', 'description', 'next_step', 'subagent': {task: {...}}}}``.
    """
    # Initialize orchestrator agent
    task_model, task_vllm_endpoint, task_use_vllm = init_task_models(
        orchestrator_model, 
        orchestrator_vllm_endpoint
    )
    orchestrator_agent = OrchestratorAgent(
        model=task_model,
        use_vllm=task_use_vllm,
        vllm_endpoint=task_vllm_endpoint,
        temperature=0 if not 'gpt-5' in task_model.lower() else 1
    )
    mas_structure = {
        'orchestrator': {
            'agent': orchestrator_agent,
            'description': AGENT_DESCRIPTION['orchestrator'],
            'next_step': None,
            'subagent': {}
        }
    }

    # Initialize sub-agents based on the specified task types
    if first_visit_intake_model is not None:
        task_model, task_vllm_endpoint, task_use_vllm = init_task_models(
            first_visit_intake_model,
            first_visit_intake_vllm_endpoint
        )
        fv_intake_agent = IntakeAdminStaffAgent(
            model=task_model,
            department_list=None,   # Injected per-hospital at call time via `set_departments`
            max_inferences=intake_max_inference,
            use_vllm=task_use_vllm,
            vllm_endpoint=task_vllm_endpoint,
            temperature=0 if not 'gpt-5' in task_model.lower() else 1
        )
        agent_info = {
            'agent': fv_intake_agent,
            'description': AGENT_DESCRIPTION['first_visit_intake'],
            'next_step': 'first_visit_scheduling',
            'subagent': {}
        }
        mas_structure['orchestrator']['subagent']['first_visit_intake'] = agent_info
    
    if first_visit_scheduling_model is not None:
        task_model, task_vllm_endpoint, task_use_vllm = init_task_models(
            first_visit_scheduling_model, 
            first_visit_scheduling_vllm_endpoint
        )
        fv_scheduling_agent = SchedulingAdminStaffAgent(
            target_task='first_visit_scheduling',
            model=task_model,
            use_vllm=task_use_vllm,
            vllm_endpoint=task_vllm_endpoint,
            temperature=0 if not 'gpt-5' in task_model.lower() else 1
        )
        agent_info = {
            'agent': fv_scheduling_agent,
            'description': AGENT_DESCRIPTION['first_visit_scheduling'],
            'next_step': None,
            'subagent': {}
        }
        mas_structure['orchestrator']['subagent']['first_visit_scheduling'] = agent_info
    
    if follow_up_visit_scheduling_model is not None:
        task_model, task_vllm_endpoint, task_use_vllm = init_task_models(
            follow_up_visit_scheduling_model, 
            follow_up_visit_scheduling_vllm_endpoint
        )
        fu_scheduling_agent = SchedulingAdminStaffAgent(
            target_task='follow_up_visit_scheduling',
            model=task_model,
            use_vllm=task_use_vllm,
            vllm_endpoint=task_vllm_endpoint,
            temperature=0 if not 'gpt-5' in task_model.lower() else 1
        )
        agent_info = {
            'agent': fu_scheduling_agent,
            'description': AGENT_DESCRIPTION['follow_up_visit_scheduling'],
            'next_step': None,
            'subagent': {}
        }
        mas_structure['orchestrator']['subagent']['follow_up_visit_scheduling'] = agent_info

    return mas_structure
