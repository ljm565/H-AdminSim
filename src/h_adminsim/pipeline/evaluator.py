from typing import Optional

from h_adminsim.policy.evaluation import run_evaluations
from h_adminsim.tools.evaluator import FirstVisitEvaluator, FollowUpVisitEvaluator
from h_adminsim.utils import colorstr, log



class Evaluator:
    """
    Single entry point for evaluating simulation results.
    First-visit evaluations (prefixed `fv_`) read one results directory, while follow-up evaluations
    (prefixed `fu_`) compare several runs of the same data under different negotiation settings.
    """
    FV_TASKS = ('fv_task', 'fv_department', 'fv_rounds', 'fv_token', 'fv_human')
    FU_TASKS = ('fu_tau', 'fu_dashboard', 'fu_protection', 'fu_custom')


    def __init__(self,
                 fv_path: Optional[str] = None,
                 fu_hospital_side_path: Optional[str] = None,
                 fu_patient_side_path: Optional[str] = None,
                 fu_interpolated_paths: Optional[list[str]] = None,
                 fu_protection_paths: Optional[list[str]] = None,
                 fu_agent_data_path: Optional[str] = None,
                 fu_custom_paths: Optional[list[str]] = None,
                 fu_custom_configs: Optional[list[str]] = None,
                 model_name: Optional[str] = None):
        """
        Args:
            fv_path (Optional[str], optional): First-visit (intake and/or scheduling) results directory.
                                               Required for the `fv_*` evaluations. Defaults to None.
            fu_hospital_side_path (Optional[str], optional): Follow-up results directory of a `hospital-side` negotiation run.
                                                             Required for `fu_tau` and `fu_dashboard`. Defaults to None.
            fu_patient_side_path (Optional[str], optional): Follow-up results directory of a `patient-side` negotiation run.
                                                            Adds the patient-side pole to `fu_dashboard`. Defaults to None.
            fu_interpolated_paths (Optional[list[str]], optional): Follow-up results directories of `threshold` negotiation runs.
                                                                   The dashboard keeps them in the given order. Defaults to None.
            fu_protection_paths (Optional[list[str]], optional): Follow-up results directories compared by `fu_protection`,
                                                                 including runs without protection as baselines. Defaults to None.
            fu_agent_data_path (Optional[str], optional): Agent simulation data directory the follow-up runs used, read by `fu_protection`.
                                                          If None, `<results dir>/../agent_data` is used. Defaults to None.
            fu_custom_paths (Optional[list[str]], optional): Follow-up results directories evaluated by `fu_custom` with the
                                                             user-defined evaluations of each run's policy snapshot. Defaults to None.
            fu_custom_configs (Optional[list[str]], optional): Policy config files whose `evaluations` `fu_custom` runs on every
                                                               `fu_custom_paths` run instead of each run's snapshot. Defaults to None.
            model_name (Optional[str], optional): Model name. Required for `fv_token`, and shown in the `fu_dashboard`. Defaults to None.
        """
        self.fv_path = fv_path
        self.fu_hospital_side_path = fu_hospital_side_path
        self.fu_patient_side_path = fu_patient_side_path
        self.fu_interpolated_paths = fu_interpolated_paths
        self.fu_protection_paths = fu_protection_paths
        self.fu_agent_data_path = fu_agent_data_path
        self.fu_custom_paths = fu_custom_paths
        self.fu_custom_configs = fu_custom_configs
        self.model_name = model_name


    def _check_tasks(self, tasks: list[str]):
        """
        Check that the requested evaluations are known and that their inputs were given.

        Args:
            tasks (list[str]): Evaluations to run.

        Raises:
            ValueError: If an evaluation is unknown or its required input is missing.
        """
        unknown = set(tasks) - set(self.FV_TASKS) - set(self.FU_TASKS)
        if unknown:
            raise ValueError(colorstr("red", f"Unknown evaluation(s): {sorted(unknown)}. Choose from {list(self.FV_TASKS + self.FU_TASKS)}."))

        if any(t in self.FV_TASKS for t in tasks) and not self.fv_path:
            raise ValueError(colorstr("red", f"`fv_path` is required for first-visit evaluations {list(self.FV_TASKS)}."))
        if 'fv_token' in tasks and not self.model_name:
            raise ValueError(colorstr("red", "`model_name` is required for `fv_token`."))
        if any(t in ('fu_tau', 'fu_dashboard') for t in tasks) and not self.fu_hospital_side_path:
            raise ValueError(colorstr("red", "`fu_hospital_side_path` is required for `fu_tau` and `fu_dashboard`."))
        if 'fu_protection' in tasks and not self.fu_protection_paths:
            raise ValueError(colorstr("red", "`fu_protection_paths` is required for `fu_protection`."))
        if 'fu_custom' in tasks and not self.fu_custom_paths:
            raise ValueError(colorstr("red", "`fu_custom_paths` is required for `fu_custom`."))


    def evaluate(self, tasks: list[str], save: bool = False, tau_n: int = 4) -> dict:
        """
        Run the requested evaluations and log their results.

        Args:
            tasks (list[str]): Evaluations to run.
                - `fv_task`: Task accuracy of each first-visit task.
                - `fv_department`: Department recommendation accuracy of the intake task.
                - `fv_rounds`: Average number of intake dialogue rounds.
                - `fv_token`: API cost from the token usages (needs `model_name`).
                - `fv_human`: Human evaluation scores (`.txt` files in `fv_path`).
                - `fu_tau`: Interior τ seeds for the `threshold` runs, computed from the hospital-side run.
                - `fu_dashboard`: Negotiation dashboard over the hospital-side, interpolated, and patient-side runs.
                - `fu_protection`: Comparison of urgent capacity protection across `fu_protection_paths`.
                - `fu_custom`: User-defined evaluations (policy config `evaluations`) on `fu_custom_paths`.
            save (bool, optional): Whether to save the `fu_dashboard`, `fu_protection` and `fu_custom` results next to the results. Defaults to False.
            tau_n (int, optional): Number of interior τ seeds per preference for `fu_tau`. Defaults to 4.

        Returns:
            dict: Results of the evaluations that return values (`fu_tau`, `fu_dashboard`, `fu_protection`, `fu_custom`), keyed by evaluation name.
        """
        if isinstance(tasks, str):
            tasks = [tasks]
        self._check_tasks(tasks)
        outputs = dict()

        # First-visit evaluations
        if any(t in self.FV_TASKS for t in tasks):
            log('################### First-visit evaluation ###################')
            evaluator = FirstVisitEvaluator(self.fv_path, human_eval='fv_human' in tasks)

            if 'fv_task' in tasks:
                evaluator.task_evaluation()
                log('')

            if 'fv_token' in tasks:
                evaluator.token_cost(self.model_name.lower())
                log('')

            if 'fv_human' in tasks:
                evaluator.human_evaluation()
                log('')

            if 'fv_department' in tasks:
                evaluator.department_evaluation()
                log('')

            if 'fv_rounds' in tasks:
                evaluator.calculate_avg_rounds()
                log('')

        # Follow-up evaluations
        if any(t in ('fu_tau', 'fu_dashboard', 'fu_protection') for t in tasks):
            log('################### Follow-up evaluation ###################')
            evaluator = FollowUpVisitEvaluator(
                hospital_side_path=self.fu_hospital_side_path,
                model_name=self.model_name,
                patient_side_path=self.fu_patient_side_path,
                interpolated_paths=self.fu_interpolated_paths,
                protection_paths=self.fu_protection_paths,
                agent_data_path=self.fu_agent_data_path,
            )

            if 'fu_tau' in tasks:
                outputs['fu_tau'] = evaluator.interpolation_taus(tau_n)
                log('')

            if 'fu_dashboard' in tasks:
                outputs['fu_dashboard'] = evaluator.dashboard_data(save)
                log('')

            if 'fu_protection' in tasks:
                outputs['fu_protection'] = evaluator.protection_report(save)
                log('')

        # Follow-up: user-defined evaluations
        if 'fu_custom' in tasks:
            log('################### Custom follow-up evaluation ###################')
            outputs['fu_custom'] = run_evaluations(self.fu_custom_paths, configs=self.fu_custom_configs, save=save)
            log('')

        return outputs
