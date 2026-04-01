import time
from typing import Optional
from patientsim.patient import PatientAgent
from patientsim.checker import CheckerAgent
from patientsim.utils.common_utils import detect_op_termination

from h_adminsim import IntakeAdminStaffAgent
from h_adminsim.utils import log, colorstr
from h_adminsim.utils.common_utils import preprocess_utterance




class OPIntakeSimulation:
    def __init__(self, 
                 patient_agent: PatientAgent,
                 admin_staff_agent: IntakeAdminStaffAgent,
                 checker_agent: Optional[CheckerAgent] = None,
                 max_inferences: int = 5):

        # Initialize simulation parameters
        self.patient_agent = patient_agent
        self.admin_staff_agent = admin_staff_agent
        self.checker_agent = checker_agent
        self.max_inferences = max_inferences
        self.current_inference = 0  # Current inference index
        self._sanity_check()


    def _sanity_check(self):
        """
        Verify and synchronize the maximum number of inference rounds 
        between the Administration Staff agent and the OP simulation.

        If the configured values do not match, a warning is logged and 
        the Administration Staff agent's configuration is updated to align with the 
        OP simulation. The system prompt is also rebuilt accordingly.
        """
        if not self.admin_staff_agent.max_inferences == self.max_inferences:
            log("The maximum number of inferences between the Administration Staff agent and the OP simulation does not match.", level="warning")
            log(f"The simulation will start with the value ({self.max_inferences}) configured in the OP simulation, \
                and the Administration Staff agent system prompt will be updated accordingly.", level="warning")
            self.admin_staff_agent.max_inferences = self.max_inferences
            self.admin_staff_agent.build_prompt()

        if self.checker_agent:
            assert self.checker_agent.visit_type == self.patient_agent.visit_type, \
                log(colorstr("red", f"The visit type between the Checker agent ({self.checker_agent.visit_type}) and the Patient agent ({self.patient_agent.visit_type}) must be the same."))


    def _init_agents(self, verbose: bool = True) -> None:
        """
        Reset the conversation histories and token usage records of both the Patient and Doctor agents.

        Args:
            verbose (bool, optional): Whether to print verbose output. Defaults to True.
        """
        self.patient_agent.reset_history(verbose=verbose)
        self.admin_staff_agent.reset_history(verbose=verbose)


    def simulate(self, 
                 verbose: bool = True, 
                 patient_kwargs: dict = {},
                 staff_kwargs: dict = {},
                 **kwargs) -> list[dict]:
        """
        Run a full conversation simulation between the Administration Staff and Patient agents in the outpatient setting.

        The simulation alternates turns between the Administration Staff and Patient until
        the maximum number of inference rounds is reached or early termination
        is detected. During the final turn, the Administration Staff agent should assign the department to the patient.

        Args:
            verbose (bool, optional): Whether to print verbose output. Defaults to True.
            patient_kwargs (dict, optional): Additional keyword arguments for the Patient agent. Defaults to {}.
            staff_kwargs (dict, optional): Additional keyword arguments for the Administration Staff agent. Defaults to {}.
        
        Returns:
            list[dict]: Dialogue history, where each dict contains:
                - "role" (str): "Staff" or "Patient"
                - "content" (str): Text content of the dialogue turn
        """
        # Initialize agents
        self._init_agents(verbose=verbose)
        
        if verbose:
            log(f"Patient prompt:\n{self.patient_agent.system_prompt}")
            log(f"Administration staff prompt:\n{self.admin_staff_agent.system_prompt}")

        # Start conversation
        staff_greet = self.admin_staff_agent.staff_greet
        dialog_history = [{"role": "Staff", "content": staff_greet}]
        role = f"{colorstr('blue', 'Staff')}   [0%]"
        log(f"{role:<23}: {staff_greet}")

        merged_patient_kwargs = {**patient_kwargs, **kwargs}
        merged_staff_kwargs = {**staff_kwargs, **kwargs}

        for inference_idx in range(self.max_inferences):
            progress = int(((inference_idx + 1) / self.max_inferences) * 100)

            # Obtain response from patient
            patient_response = self.patient_agent(
                user_prompt=dialog_history[-1]["content"],
                using_multi_turn=True,
                verbose=verbose,
                **merged_patient_kwargs
            )
            dialog_history.append({"role": "Patient", "content": patient_response})
            role = f"{colorstr('green', 'Patient')} [{progress}%]"
            log(f"{role:<23}: {patient_response}")

            # Obtain response from staff
            staff_response = self.admin_staff_agent(
                user_prompt=dialog_history[-1]["content"] + "\nThis is the final turn. Now, you must provide your top5 differential diagnosis." \
                    if inference_idx == self.max_inferences - 1 else dialog_history[-1]["content"],
                using_multi_turn=True,
                verbose=verbose,
                **merged_staff_kwargs
            )
            dialog_history.append({"role": "Staff", "content": staff_response})
            role = f"{colorstr('blue', 'Staff')}   [{progress}%]"
            log(f"{role:<23}: {staff_response}")

            # If early termination is detected, break the loop
            if detect_op_termination(staff_response):
                break

            elif self.checker_agent:
                # Check if the staff response contains termination cues
                termination_check = self.checker_agent(response=staff_response).strip().upper()

                # Check if the response indicates termination
                if termination_check == "Y":
                    log("Conversation termination detected by the checker agent.", level="warning")
                    break

        log("Simulation completed.", color=True)

        output = {
            "dialog_history": dialog_history,
            "patient_token_usage": self.patient_agent.client.token_usages,
            "admin_staff_token_usage": self.admin_staff_agent.client.token_usages,
        }
        return output
    

    def simulate_stream(self,
                        verbose: bool = True,
                        patient_kwargs: dict = {},
                        staff_kwargs: dict = {},
                        **kwargs):
        """
        Run a streaming conversation simulation, yielding each utterance as it is produced.

        Functionally equivalent to `simulate()`, but yields each formatted utterance
        immediately after it is generated rather than returning the full result at the end.
        Useful for real-time display or downstream streaming pipelines.

        Args:
            verbose (bool, optional): Whether to print verbose output. Defaults to True.
            patient_kwargs (dict, optional): Additional keyword arguments for the Patient agent. Defaults to {}.
            staff_kwargs (dict, optional): Additional keyword arguments for the Administration Staff agent. Defaults to {}.
            **kwargs: Additional keyword arguments passed to both agents.

        Yields:
            Tuple[str, str]: A 2-tuple of (role, content) where role is 'Staff' or 'Patient'
                             and content is the whitespace-normalized utterance text.
        """
        # Initialize agents
        self._init_agents(verbose=verbose)
        
        if verbose:
            log(f"Patient prompt:\n{self.patient_agent.system_prompt}")
            log(f"Administration staff prompt:\n{self.admin_staff_agent.system_prompt}")

        # Start conversation
        staff_greet = self.admin_staff_agent.staff_greet
        dialog_history = [{"role": "Staff", "content": staff_greet}]
        role = f"{colorstr('blue', 'Staff')}   [0%]"
        log(f"{role:<23}: {staff_greet}")
        yield 'Staff', preprocess_utterance(staff_greet)

        merged_patient_kwargs = {**patient_kwargs, **kwargs}
        merged_staff_kwargs = {**staff_kwargs, **kwargs}

        for inference_idx in range(self.max_inferences):
            progress = int(((inference_idx + 1) / self.max_inferences) * 100)

            # Obtain response from patient
            patient_response = self.patient_agent(
                user_prompt=dialog_history[-1]["content"],
                using_multi_turn=True,
                verbose=verbose,
                **merged_patient_kwargs
            )
            dialog_history.append({"role": "Patient", "content": patient_response})
            role = f"{colorstr('green', 'Patient')} [{progress}%]"
            log(f"{role:<23}: {patient_response}")
            yield 'Patient', preprocess_utterance(patient_response)

            # Obtain response from staff
            staff_response = self.admin_staff_agent(
                user_prompt=dialog_history[-1]["content"] + "\nThis is the final turn. Now, you must provide your top5 differential diagnosis." \
                    if inference_idx == self.max_inferences - 1 else dialog_history[-1]["content"],
                using_multi_turn=True,
                verbose=verbose,
                **merged_staff_kwargs
            )
            dialog_history.append({"role": "Staff", "content": staff_response})
            role = f"{colorstr('blue', 'Staff')}   [{progress}%]"
            log(f"{role:<23}: {staff_response}")
            yield 'Staff', preprocess_utterance(staff_response)

            # If early termination is detected, break the loop
            if detect_op_termination(staff_response):
                break

            elif self.checker_agent:
                # Check if the staff response contains termination cues
                termination_check = self.checker_agent(response=staff_response).strip().upper()

                # Check if the response indicates termination
                if termination_check == "Y": 
                    log("Conversation termination detected by the checker agent.", level="warning")
                    break

        log("Simulation completed.", color=True)
