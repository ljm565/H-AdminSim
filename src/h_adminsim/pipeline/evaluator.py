from h_adminsim.tools import Evaluator as BaseEvaluator
from h_adminsim.utils import log



class Evaluator(BaseEvaluator):
    def __init__(self, path: str):
        super().__init__(path)
        
    
    def evaluate(self, tasks: list[str]):
        """
        Evaluate the performance of an agent.

        Args:
            tasks (list[str]): A task list to evaluate.
        """
        if 'task' in tasks:
            self.task_evaluation()
            log('')

        if 'feedback' in tasks:
            self.supervisor_evaluation()
            log('')

        if 'rounds' in tasks:
            self.calculate_avg_rounds()
            log('')

        if 'department' in tasks:
            self.department_evaluation()
            log('')
        