import os
import glob
import json
from typing import ClassVar, Iterable, Optional, Union
from dataclasses import dataclass, field, fields, is_dataclass

from h_adminsim.registry.errors import PolicyError
from h_adminsim.utils import log, colorstr
from h_adminsim.utils.common_utils import decode_non_finite
from h_adminsim.utils.filesys_utils import get_files, json_load, json_save_fast



TASK = 'follow_up_visit_scheduling'



def _flat(values: Iterable) -> list:
    """
    Flatten a result column, where one entry may hold several records (a booking and its follow-on events).

    Args:
        values (Iterable): The column's entries (a record or a list / tuple of records each).

    Returns:
        list: The records, in order.
    """
    return [x for y in values for x in (y if isinstance(y, (list, tuple)) else [y])]



@dataclass(frozen=True)
class RunData:
    """
    One follow-up simulation run, as an `Evaluation` sees it.

    Fields:
        path: The run's results directory.
        result_file: The run's ``*_result.json``.
        metadata: The run's ``_metadata`` (operating window, ``policies``, ``metrics``, ``evaluations``).
        records: Every follow-up result record in order, ``{'gt', 'pred', 'status', 'status_code'}``: the bookings and
            the cancellation / rescheduling events, successful or not.
    """
    path: str
    result_file: str
    metadata: dict
    records: list = field(default_factory=list)

    @property
    def bookings(self) -> list:
        """
        The successful follow-up bookings (the ``pred`` of every passed record that holds a patient booking).

        Each booking carries ``patient``, ``urgency``, ``preference``, ``test`` (booked slots with ``result_ready_at``),
        ``negotiation_metrics`` (built-in and user metrics, the decision, ``decided_by``; infinite values such as a free
        win's ``gpc`` are floats), ``restrictions``
        (``{policy id: [[test_code, date], ...]}`` blocked for this booking's tests) and ``capacity_protection``.

        Returns:
            list: The booking dicts.
        """
        return [r['pred'] for r in self.records
                if r['status'] is True and isinstance(r['pred'], dict) and 'patient' in r['pred']]


    @property
    def policies(self) -> list:
        """The run's policies (``_metadata.policies``)."""
        return list(self.metadata.get('policies') or [])


    @property
    def metrics(self) -> list:
        """The run's user-defined negotiation metrics (``_metadata.metrics``)."""
        return list(self.metadata.get('metrics') or [])


    @classmethod
    def load(cls, path: str) -> "RunData":
        """
        Read a run's results directory.

        Args:
            path (str): The run's results directory.

        Raises:
            FileNotFoundError: If no ``*_result.json`` is found under `path`.
            PolicyError: If the run has no follow-up results.

        Returns:
            RunData: The run.
        """
        from h_adminsim.utils.policy_utils import run_policies

        files = sorted(get_files(path, '_result.json'))
        if not files:
            raise FileNotFoundError(f"No '*_result.json' under {path}")
        data = json_load(files[0])
        metadata = data.get('_metadata') or {}
        run_policies(metadata, where=files[0])
        task = data.get(TASK)
        if task is None:
            raise PolicyError(colorstr("red", f"{files[0]}: no follow-up scheduling results to evaluate."))
        keys = ('gt', 'pred', 'status', 'status_code')
        columns = {k: _flat(task.get(k) or []) for k in keys}
        records = [dict(zip(keys, values)) for values in zip(*(columns[k] for k in keys))]
        for r in records:   # infinite metrics (a free win's gpc / ti) are saved as 'inf'; give evaluations floats
            pred = r['pred']
            if isinstance(pred, dict) and isinstance(pred.get('negotiation_metrics'), dict):
                r['pred'] = {**pred, 'negotiation_metrics': decode_non_finite(pred['negotiation_metrics'])}
        return cls(path=path, result_file=files[0], metadata=metadata, records=records)



class Evaluation:
    """
    Base class of a user-defined evaluation: turns one finished run (`RunData`) into a JSON-serializable dict.

    Evaluations read what the simulation recorded: each booking's ``negotiation_metrics`` (including user metrics and
    whatever a `review` policy put in its decision's ``record``), its ``restrictions`` per `restrict` policy, its booked
    tests, and the run metadata. Like a policy, an evaluation is a dataclass whose fields are its configuration.
    """
    api_version: ClassVar[int] = 1
    name: ClassVar[str] = ''

    # Set by the builder (see `Policy`)
    id = ''
    source = None
    source_sha256 = None


    def evaluate(self, run: RunData) -> dict:
        """
        Evaluate one run.

        Args:
            run (RunData): The run.

        Raises:
            NotImplementedError: If the subclass does not implement it.

        Returns:
            dict: JSON-serializable results.
        """
        raise NotImplementedError


    def params(self) -> dict:
        """
        The evaluation's configuration (its dataclass fields), as recorded in the run metadata.

        Returns:
            dict: ``{field: value}``.
        """
        if not is_dataclass(self):
            return {}
        return {f.name: getattr(self, f.name) for f in fields(self)}



def _run_snapshot_paths(run_dir: str) -> list:
    """
    The policy config snapshot of a run (``<run>/policy/NN_*.yaml``) in priority order.

    Args:
        run_dir (str): The run's results directory.

    Returns:
        list: The snapshot YAML paths (empty when the run has none).
    """
    return sorted(glob.glob(os.path.join(run_dir, 'policy', '*.yaml')))



def run_evaluations(run_dirs: list,
                    configs: Optional[Union[str, list]] = None,
                    save: bool = False) -> dict:
    """
    Run user-defined evaluations on finished runs.

    By default each run is evaluated with the evaluations listed in its own policy config snapshot (``<run>/policy/``),
    i.e. the evaluation code the run was set up with. Passing `configs` evaluates every run with the evaluations of
    those policy config files instead (e.g. an evaluation written after the runs finished).

    Args:
        run_dirs (list): Results directories of the runs.
        configs (Optional[Union[str, list]], optional): Policy config file(s) whose `evaluations` to use for every run.
                                                        Defaults to None (each run's own snapshot).
        save (bool, optional): Whether to save the report to ``custom_report.json`` in the parent folder of the first
                               run. Defaults to False.

    Raises:
        PolicyError: If a run has no evaluation to run, or an evaluation returns a non-JSON-serializable result.

    Returns:
        dict: ``{run directory: {evaluation id: result}}``.
    """
    from h_adminsim.policy.builder import load_policy_config

    shared = load_policy_config(configs).evaluations if configs is not None else None
    report = {}
    for run_dir in run_dirs:
        run = RunData.load(run_dir)
        if shared is not None:
            evaluations = shared
        else:
            snapshot = _run_snapshot_paths(run_dir)
            evaluations = load_policy_config(snapshot).evaluations if snapshot else []
        if not evaluations:
            raise PolicyError(colorstr("red", f"{run_dir}: no evaluation to run; list `evaluations` in the run's policy "
                                              f"config, or pass evaluation config files."))
        results = {}
        for evaluation in evaluations:
            result = evaluation.evaluate(run)
            try:
                json.dumps(result, allow_nan=True)
            except (TypeError, ValueError) as e:
                raise PolicyError(colorstr("red", f"Evaluation `{evaluation.id}` returned a non-JSON-serializable result "
                                                  f"for {run_dir}: {e}")) from e
            results[evaluation.id] = result
            log(f"{colorstr(os.path.basename(os.path.normpath(run_dir)))} - {evaluation.id}: {result}")
        report[run_dir] = results

    if save and run_dirs:
        save_path = os.path.join(os.path.dirname(os.path.normpath(run_dirs[0])), 'custom_report.json')
        json_save_fast(save_path, report)
        log(f"Custom evaluation report saved to {colorstr(save_path)}")
    return report
