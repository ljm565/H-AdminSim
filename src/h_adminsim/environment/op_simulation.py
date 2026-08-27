from typing import TYPE_CHECKING
from patientsim import PatientAgent

if TYPE_CHECKING:
    from h_adminsim.pipeline import HospitalMAS



class OPSimulation:
    """
    Common base for every outpatient dialogue simulation.

    Holds the two agents that every simulation drives — the Patient agent and the
    Administration Staff MAS — together with the reset logic they share. Task-specific
    state (dialogue histories, scheduling rules, prompts) belongs to the subclasses.
    """
    # Must be declared in subclasses to specify which dialogue histories to keep.
    HISTORY_KEYS: tuple[str, ...]
    
    def __init__(self,
                 patient_agent: PatientAgent,
                 admin_staff_mas: "HospitalMAS"):

        self.patient_agent = patient_agent
        self.admin_staff_mas = admin_staff_mas


    def _init_history(self):
        """
        Reset the dialogue histories.
        """
        self.dialog_history = {key: [] for key in self.HISTORY_KEYS}


    def _init_agents(self, verbose: bool = True) -> None:
        """
        Reset the conversation histories and token usage records of both the Patient and Doctor agents.

        Args:
            verbose (bool, optional): Whether to print verbose output. Defaults to True.
        """
        self.patient_agent.reset_history(verbose=verbose)
        self.admin_staff_mas.reset(verbose=verbose)
