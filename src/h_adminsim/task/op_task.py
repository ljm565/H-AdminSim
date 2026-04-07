from typing import Tuple, Optional

from h_adminsim.utils import log



class OutpatientTask:
    def __init__(self):
        self.reset_token_data()

    
    def reset_token_data(self):
        self.token_stats = {
            'simulation_n': 0,
            'patient_token': {'input':[], 'output': [], 'reasoning': []}, 
            'admin_staff_token': {'input': [], 'output': [], 'reasoning': []}, 
            'supervisor_token': {'input':[], 'output': [], 'reasoning': []}
        }


    def save_token_data(self, 
                        patient_token: Optional[dict] = None, 
                        admin_staff_token: Optional[dict] = None, 
                        supervisor_token: Optional[dict] = None):
        """
        Save the API token usage data

        Args:
            patient_token (Optional[dict], optional): Patient token information. Defaults to None.
            admin_staff_token (Optional[dict], optional): Administration staff token information. Defaults to None.
            supervisor_token (Optional[dict], optional): Supervisor token information. Defaults to None.
        """
        self.token_stats['simulation_n'] += 1
        if patient_token:
            self.token_stats['patient_token']['input'].extend(patient_token['prompt_tokens'])
            self.token_stats['patient_token']['output'].extend(patient_token['completion_tokens'])
            if 'reasoning_tokens' in patient_token:
                self.token_stats['patient_token']['reasoning'].extend(patient_token['reasoning_tokens'])

        if admin_staff_token:
            self.token_stats['admin_staff_token']['input'].extend(admin_staff_token['prompt_tokens'])
            self.token_stats['admin_staff_token']['output'].extend(admin_staff_token['completion_tokens'])
            if 'reasoning_tokens' in admin_staff_token:
                self.token_stats['admin_staff_token']['reasoning'].extend(admin_staff_token['reasoning_tokens'])

        if supervisor_token:
            self.token_stats['supervisor_token']['input'].extend(supervisor_token['prompt_tokens'])
            self.token_stats['supervisor_token']['output'].extend(supervisor_token['completion_tokens'])
            if 'reasoning_tokens' in supervisor_token:
                self.token_stats['supervisor_token']['reasoning'].extend(supervisor_token['reasoning_tokens'])
    
    
    def _init_task_models(self, model: str, vllm_endpoint: Optional[str] = None) -> Tuple[str, str, bool]:
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
            assert vllm_endpoint is not None, log('VLLM endpoint must be provided for non-Gemini/GPT models.', 'error')
            return model, vllm_endpoint, True