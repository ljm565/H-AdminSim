from typing import Optional, Tuple

from h_adminsim.agent import *
from h_adminsim.utils import colorstr



def init_task_models(model: str, 
                     vllm_endpoint: Optional[str] = None) -> Tuple[str, str, bool]:
    """
    Initialize the model for the task.

    Args:
        model (str): The model name.
        vllm_endpoint (Optional[str], optional): The VLLM endpoint URL. Defaults to None.
    
    Returns:
        Tuple[str, str, bool]: The model name, VLLM endpoint URL, vllm usage flag.
    """
    if any(keyword in model.lower() for keyword in ['gemini', 'gpt']):
        return model, None, False
    else:
        assert vllm_endpoint is not None, colorstr("red", 'vLLM endpoint must be provided for non-Gemini/GPT models.')
        return model, vllm_endpoint, True
    


def init_mas_system(orchestrator_model: str,
                    orchestrator_vllm_endpoint: Optional[str],
                    first_visit_intake_model: Optional[str] = None,
                    first_visit_intake_vllm_endpoint: Optional[str] = None,
                    first_visit_scheduling_model: Optional[str] = None,
                    first_visit_scheduling_vllm_endpoint: Optional[str] = None,
                    follow_up_visit_scheduling_model: Optional[str] = None,
                    follow_up_visit_scheduling_vllm_endpoint: Optional[str] = None,
                    intake_max_inference: int = 5,) -> dict:
    """
    Initialize the multi-agent system (MAS) structure with the specified models for the orchestrator and sub-agents.

    Args:
        orchestrator_model (str): The model name for the orchestrator agent.
        orchestrator_vllm_endpoint (Optional[str]): The vLLM endpoint URL for the orchestrator agent (if applicable).
        first_visit_intake_model (Optional[str], optional): The model name for the first visit intake agent. Defaults to None.
        first_visit_intake_vllm_endpoint (Optional[str], optional): The vLLM endpoint URL for the first visit intake agent (if applicable). Defaults to None.
        first_visit_scheduling_model (Optional[str], optional): The model name for the first visit scheduling agent. Defaults to None.
        first_visit_scheduling_vllm_endpoint (Optional[str], optional): The vLLM endpoint URL for the first visit scheduling agent (if applicable). Defaults to None.
        follow_up_visit_scheduling_model (Optional[str], optional): The model name for the follow-up visit scheduling agent. Defaults to None.
        follow_up_visit_scheduling_vllm_endpoint (Optional[str], optional): The vLLM endpoint URL for the follow-up visit scheduling agent (if applicable). Defaults to None.
        intake_max_inference (int, optional): The maximum number of inferences allowed for the intake agent. Defaults to 5.

    Returns:
        dict: A dictionary representing the initialized multi-agent system structure.
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
            'description': 'Oversees the entire patient intake and scheduling process, making high-level decisions and delegating tasks to sub-agents.',
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
            'description': 'Patient intake and department recommendation.',
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
            'description': 'Schedule a first-visit appointment.',
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
            'description': 'Schedule follow-up visit tests and appointments.',
            'subagent': {}
        }
        mas_structure['orchestrator']['subagent']['follow_up_visit_scheduling'] = agent_info

    return mas_structure
