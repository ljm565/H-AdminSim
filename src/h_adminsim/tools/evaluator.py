import os
import math
import numpy as np
from bisect import bisect_right
from decimal import Decimal, getcontext
from collections import Counter, defaultdict
from typing import Any, Iterator, Optional, Union

from h_adminsim.utils import log, colorstr
from h_adminsim.utils.common_utils import (
    get_iso_time,
    hours_between,
    str_to_datetime,
    encode_non_finite,
    convert_time_to_segment,
)
from h_adminsim.tools.negotiation_metrics import frontload_pairs
from h_adminsim.utils.policy_utils import run_policies, run_policy_summary
from h_adminsim.utils.image_preprocess_utils import draw_fail_donut_subplots
from h_adminsim.utils.filesys_utils import get_files, json_load, json_save_fast



class Evaluator:
    """
    Base evaluator: result loading and the accuracy / token-cost / supervisor / human evaluations shared by tasks.
    """
    def __init__(self, path: str, human_eval: bool = False):
        """
        Collect the run's `*_result.json` files (and human evaluation `.txt` files).

        Args:
            path (str): Results directory of a run.
            human_eval (bool, optional): If True, also collect the `.txt` human evaluation files. Defaults to False.
        """
        self.path = path
        self.files = get_files(self.path, '_result.json')
        if human_eval:
            self.human_eval_files = get_files(self.path, '.txt')

        self.model_pricing = {
            "gpt-5-nano": {
                "input": 0.05,    # $0.05 / 1M tokens
                "output": 0.40,   # $0.40 / 1M tokens (reasoning)
            },
            "gpt-5-mini": {
                "input": 0.25,    # $0.25 / 1M tokens
                "output": 2.00,   # $2.00 / 1M tokens (reasoning)
            },
            "gemini-2.5-flash": {
                "input": 0.30,    # $0.30 / 1M tokens
                "output": 2.50,   # $2.50 / 1M tokens (reasoning)
            },
        }


    def task_evaluation(self):
        """
        Log macro- and micro-wise accuracy and failure types per task, and draw `fails.png`.
        """
        aggregated_results = dict()
        for file in self.files:
            data = json_load(file)

            for task, value in data.items():
                if task.startswith('_'):    # Environment state a run saved, not a task's result
                    continue

                if not task in aggregated_results:
                    aggregated_results[task] = {'status': [], 'status_code': []}

                aggregated_results[task]['status'].append(value['status'])
                aggregated_results[task]['status_code'].append(value['status_code'])

        # Macro-wise evaluation
        log('--------------Macro-wise Evaluation--------------')
        for task, value in aggregated_results.items():
            statuses = [[all(s.values()) for s in single_s] for single_s in value['status']] if task == 'first_visit_intake' else value['status']
            accuracies = [sum(x if isinstance(x, bool) else sum(x) for x in status) / sum(1 if isinstance(x, bool) else len(x) for x in status) * 100 for status in statuses]
            avg_accuracy = sum(accuracies) / len(accuracies)
            stdv = round((sum((x - avg_accuracy) ** 2 for x in accuracies) / len(accuracies)) ** 0.5, 2) if len(accuracies) > 1 else 0.0
            log(f'{colorstr(task):<27} | average accuracy: {colorstr("green", f"{avg_accuracy:.2f}% ± {stdv}")}, files: {len(accuracies)}')
            log(f'    - Individual accuracies: {", ".join([colorstr("green", f"{acc:.2f}%") for acc in accuracies])}')


        # Micro-wise evaluation
        log('')
        log('--------------Micro-wise Evaluation--------------')
        fail_data_dict = dict()
        for task, value in aggregated_results.items():
            if task == 'first_visit_intake':
                # Statuses
                _status = [[all(s.values()) for s in single_s] for single_s in value['status']]
                _patient_status = [[s['patient'] for s in single_s] for single_s in value['status']]
                _staff_status = [[s['staff'] for s in single_s] for single_s in value['status']]
                status = [x for y in sum(_status, []) for x in (y if isinstance(y, list) or isinstance(y, tuple) else [y])]
                patient_status = [x for y in sum(_patient_status, []) for x in (y if isinstance(y, list) or isinstance(y, tuple) else [y])]
                staff_status = [x for y in sum(_staff_status, []) for x in (y if isinstance(y, list) or isinstance(y, tuple) else [y])]

                # Status codes
                _patient_status_code = [[sc['patient'] for sc in single_sc] for single_sc in value['status_code']]
                _staff_status_code = [[sc['staff'] for sc in single_sc] for single_sc in value['status_code']]
                patient_status_code = [x for y in sum(_patient_status_code, []) for x in (y if isinstance(y, list) or isinstance(y, tuple) else [y])]
                staff_status_code = [x for y in sum(_staff_status_code, []) for x in (y if isinstance(y, list) or isinstance(y, tuple) else [y])]

                # Fail cases
                patient_failed_cases = [c for s, c in zip(patient_status, patient_status_code) if not s and 'unexpected' not in c]
                failed_cases = [c for s, c in zip(staff_status, staff_status_code) if not s and 'unexpected' not in c]
                failed_case_l = len(failed_cases)
                patient_failed_case_l = len(patient_failed_cases)
            else:
                status = [x for y in sum(value['status'], []) for x in (y if isinstance(y, list) or isinstance(y, tuple) else [y])]
                status_code = [x for y in sum(value['status_code'], []) for x in (y if isinstance(y, list) or isinstance(y, tuple) else [y])]
                patient_failed_cases = []
                failed_cases = [c for s, c in zip(status, status_code) if not s and 'unexpected' not in c]
                failed_case_l = len(failed_cases)
                patient_failed_case_l = len(patient_failed_cases)

            accuracy = sum(status) / len(status) * 100
            error_rate = (failed_case_l / len(status)) * 100
            log(f'{colorstr(task):<27} | accuracy: {colorstr("green", f"{accuracy:.2f}%")}, length: {sum(status)} / {len(status)}')
            log(f'{f"{colorstr(task)} (staff)":<27} | Error   : {colorstr("red", f"{error_rate:.2f}%")}, length: {failed_case_l} / {len(status)}')

            if failed_cases:
                fail_summary = Counter(failed_cases)
                reschedule_fail_summary = Counter()

                for k, v in list(fail_summary.items()):
                    if k.startswith("reschedule:") and 'identify' not in k and 'unexpected' not in k:
                        norm_key = k.replace("reschedule:", "").strip()
                        fail_summary[norm_key] += v
                        reschedule_fail_summary[norm_key] += v
                        fail_summary.pop(k)

                for fail_type, count in fail_summary.items():
                    percent = (count / failed_case_l) * 100
                    reschedule_n = reschedule_fail_summary[fail_type] if fail_type in reschedule_fail_summary else 0
                    if reschedule_n:
                        log(f'    - Fail type {colorstr("red", fail_type):<30}: {count} (reschedule: {reschedule_n}) cases ({percent:.2f}%)')
                    else:
                        log(f'    - Fail type {colorstr("red", fail_type):<30}: {count} cases ({percent:.2f}%)')
                fail_data_dict[task] = failed_cases

            if patient_failed_cases:
                error_rate = (patient_failed_case_l / len(status)) * 100
                log(f'{f"{colorstr(task)} (patient)":<27} | Error   : {colorstr("red", f"{error_rate:.2f}%")}, length: {patient_failed_case_l} / {len(status)}')
                fail_summary = Counter(patient_failed_cases)
                for fail_type, count in fail_summary.items():
                    percent = (count / patient_failed_case_l) * 100
                    log(f'    - Fail type {colorstr("red", fail_type):<30}: {count} cases ({percent:.2f}%)')

        draw_fail_donut_subplots(fail_data_dict, os.path.join(self.path, 'fails.png'))


    def token_cost(self, model_name: str):
        """
        Log the average API cost and token usage per simulation for each task.

        Args:
            model_name (str): Priced model: 'gpt-5-nano', 'gpt-5-mini' or 'gemini-2.5-flash'.

        Raises:
            ValueError: If `model_name` has no pricing.
        """
        if model_name not in self.model_pricing:
            raise ValueError(f"Unknown model: {model_name}. Choose from {list(self.model_pricing.keys())}")

        aggregated_results = dict()
        for file in self.files:
            data = json_load(file)
            for task, value in data.items():
                if task.startswith('_'):    # Environment state a run saved, not a task's result
                    continue

                if task not in aggregated_results:
                    aggregated_results[task] = {'token': []}
                aggregated_results[task]['token'].extend(value.get('token', []))

        pricing = self.model_pricing[model_name]
        agent_keys = ["patient_token", "admin_staff_token", "supervisor_token"]

        log('')
        log('--------------Micro-wise Calculation--------------')
        for task, value in aggregated_results.items():
            sim_costs = []

            for sim in value['token']:
                if not len(sim):
                    continue

                sim_cost = {}
                sim_total = 0.0

                for agent_key in agent_keys:
                    agent_data = sim.get(agent_key, {})
                    input_tokens     = sum(agent_data.get("input", []))
                    output_tokens    = sum(agent_data.get("output", []))
                    reasoning_tokens = sum(agent_data.get("reasoning", []))
                    total_output_tokens = output_tokens + reasoning_tokens

                    agent_cost = (input_tokens / 1_000_000) * pricing["input"] \
                            + (total_output_tokens / 1_000_000) * pricing["output"]

                    sim_cost[agent_key] = agent_cost
                    sim_cost[f"{agent_key}_input"]     = input_tokens
                    sim_cost[f"{agent_key}_output"]    = output_tokens
                    sim_cost[f"{agent_key}_reasoning"] = reasoning_tokens
                    sim_total += agent_cost

                sim_cost["total"] = sim_total
                sim_costs.append(sim_cost)

            if not sim_costs:
                log(f"No token usage found for '{task}'; skipping token cost calculation.", level='warning')
                continue

            n = len(sim_costs)
            avg ={k: sum(s[k] for s in sim_costs) / n for k in agent_keys + ["total"]}

            log(f"{task} (n={n})", color=True)
            for agent_key in agent_keys:
                avg_cost = sum(s[agent_key]                      for s in sim_costs) / n
                avg_input = sum(s[f"{agent_key}_input"]           for s in sim_costs) / n
                avg_output = sum(s[f"{agent_key}_output"]          for s in sim_costs) / n
                avg_reasoning = sum(s[f"{agent_key}_reasoning"]       for s in sim_costs) / n

                log(f"{agent_key}")
                log(f"    price               : ${avg_cost:.6f}")
                log(f"    avg_input_tokens    : {avg_input:.1f}")
                log(f"    avg_output_tokens   : {avg_output:.1f}")
                log(f"    avg_reasoning_tokens: {avg_reasoning:.1f}")
                log("")

            log(f"total            : ${avg['total']:.6f}")


    def human_evaluation(self):
        """
        Log arena wins and average scores per model from the tab-separated human evaluation files.
        """
        scores = {'arena': dict(), 'score': dict()}
        all_lines = list()
        for file in self.human_eval_files:
            with open(file, 'r') as f:
                lines = f.readlines()
                all_lines.extend([line.strip() for line in lines if line.strip()])

        for line in all_lines:
            arena, score_a, score_b, model_a, model_b = line.split('\t')
            scores['arena'].setdefault(model_a, 0)
            scores['arena'].setdefault(model_b, 0)
            scores['score'].setdefault(model_a, [])
            scores['score'].setdefault(model_b, [])

            if arena == 'A':
                scores['arena'][model_a] += 1
            else:
                scores['arena'][model_b] += 1

            scores['score'][model_a].append(float(score_a))
            scores['score'][model_b].append(float(score_b))

        log('--------------Human Evaluation--------------')
        for model in scores['arena'].keys():
            arena_wins = scores['arena'][model]
            score_list = scores['score'][model]
            avg_score = sum(score_list) / len(score_list)
            stdv = round((sum((x - avg_score) ** 2 for x in score_list) / len(score_list)) ** 0.5, 2) if len(score_list) > 1 else 0.0
            log(f'{colorstr(model):<15} | Arena wins: {colorstr("green", str(arena_wins))}, Average score: {colorstr("green", f"{avg_score:.2f} ± {stdv}")}')



class FirstVisitEvaluator(Evaluator):
    """
    First-visit (outpatient intake + first-visit scheduling) evaluation.
    """
    def department_evaluation(self):
        """
        Log the department error rate of the intake results.
        """
        aggregated_results = {'first_visit_intake': {'gt': [], 'pred': [], 'status': []}}

        for file in self.files:
            data = json_load(file)
            if 'first_visit_intake' not in data:
                continue
            aggregated_results['first_visit_intake']['gt'].extend(data['first_visit_intake']['gt'])
            aggregated_results['first_visit_intake']['pred'].extend(data['first_visit_intake']['pred'])
            aggregated_results['first_visit_intake']['status'].extend(data['first_visit_intake']['status'])

        gt = aggregated_results['first_visit_intake']['gt']
        pred = aggregated_results['first_visit_intake']['pred']
        if not gt:
            log("No 'first_visit_intake' results found; skipping department evaluation.", level='warning')
            return

        status = [all(s.values()) for s in aggregated_results['first_visit_intake']['status']]
        total_n, dept_err_n = len(gt), 0
        for g, p, s in zip(gt, pred, status):
            if not s:
                gt_depts = g['department']
                pred_dept = p['department'][0]

                if pred_dept not in gt_depts:
                    dept_err_n += 1

        log('--------------Department Evaluation--------------')
        log(f'Error rate: {colorstr("red", f"{(dept_err_n/total_n)*100:.2f}%")}, length: {dept_err_n} / {total_n}')


    def calculate_avg_rounds(self):
        """
        Log the mean and std of staff rounds per intake dialogue.
        """
        counts = list()
        for file in self.files:
            data = json_load(file)
            if 'first_visit_intake' not in data:
                continue
            dialogs = data['first_visit_intake']['dialog']
            for dialog in dialogs:
                counts.append(dialog.count('Staff: ')-1)

        if not counts:
            log("No 'first_visit_intake' results found; skipping average rounds calculation.", level='warning')
            return

        mean, stdv = np.mean(counts), np.std(counts)
        log('-----------------Average Rounds-----------------')
        log(f'Average Rounds: {mean:.2f} ± {stdv:.2f}')


class NegotiationRecordMixin:
    """
    Stateless helpers reading the follow-up task's stored records (final predictions, ``negotiation_metrics``,
    ``capacity_protection``) from any list of result files, plus wait / cost statistics and test capacity.
    """
    TASK = 'follow_up_visit_scheduling'
    METRIC_KEYS = ('preference', 'gpc', 'tcl', 'ti', 'G', 'R', 'U', 'U_pref', 'U_thr',
                   'negotiation_action', 'negotiation_outcome', 'negotiation_rounds')
    # Unit of `_protection_cost`'s `native` cost per scheduling objective (the quantity that objective minimizes)
    PROTECTION_COST_UNITS = {'throughput_max': 'result_delay_h', 'visit_min': 'extra_visit_days', 'stay_min': 'extra_idle_wait_h'}


    @classmethod
    def _booking_rows(cls, files: list[str]) -> Iterator[tuple]:
        """
        Yield `(gt, final pred, final status)` of every follow-up row in `files`.

        Args:
            files (list[str]): Result files.

        Returns:
            Iterator[tuple]: The rows; `gt` is a dict for a regular booking, a string for system rows.
        """
        for file in files:
            value = json_load(file).get(cls.TASK) or {}
            for gt, pred, status in zip(value.get('gt', []), value.get('pred', []), value.get('status', [])):
                yield gt, cls._final_pred(pred), cls._final_pred(status)


    @staticmethod
    def _final_pred(pred: Any) -> Any:
        """
        Last attempt of a prediction (or status) stored as a list of retries.

        Args:
            pred (Any): A single value, or a list of retries.

        Returns:
            Any: The last item (None for an empty list), or `pred` itself if not a list.
        """
        if isinstance(pred, list):
            return pred[-1] if pred else None
        return pred


    @classmethod
    def _iter_preds(cls, files: list[str]) -> Iterator[dict]:
        """
        Yield each row's final prediction dict across `files` (non-dict predictions skipped).

        Args:
            files (list[str]): Result files.

        Returns:
            Iterator[dict]: The final predictions.
        """
        for _, fp, _ in cls._booking_rows(files):
            if isinstance(fp, dict):
                yield fp


    @classmethod
    def _iter_metrics(cls, files: list[str]) -> Iterator[dict]:
        """
        Yield each prediction's `negotiation_metrics`, with stored `'inf'` / `'-inf'` strings restored to floats.

        Args:
            files (list[str]): Result files.

        Returns:
            Iterator[dict]: The metric dicts (None values stay None).
        """
        restore = {'inf': float('inf'), '-inf': float('-inf')}
        for fp in cls._iter_preds(files):
            m = fp.get('negotiation_metrics')
            if isinstance(m, dict):
                yield {k: (restore[v] if isinstance(v, str) and v in restore else v) for k, v in m.items()}


    @staticmethod
    def _is_inf(x: Any) -> bool:
        """
        Whether `x` is an infinite number (a free win's GPC / trigger index).

        Args:
            x (Any): Value to test.

        Returns:
            bool: True for an infinite int/float.
        """
        return isinstance(x, (int, float)) and math.isinf(x)

    
    @staticmethod
    def _mean(xs: list) -> float:
        """
        Mean of the non-None values.

        Args:
            xs (list): Values (None entries ignored).

        Returns:
            float: The mean, or 0.0 if there are no values.
        """
        xs = [x for x in xs if x is not None]
        return sum(xs) / len(xs) if xs else 0.0
    

    @classmethod
    def _ti(cls, m: dict) -> float:
        """
        τ-invariant trigger index ``GPC·TCL`` recomputed from the metrics (the stored ``ti`` is scaled by τ).

        Args:
            m (dict): Negotiation metrics; ``gpc`` and ``tcl`` are required (KeyError otherwise).

        Returns:
            float: ``gpc * tcl``, or inf for a free win.
        """
        gpc, tcl = m['gpc'], m['tcl']
        if cls._is_inf(gpc):
            return gpc
        return gpc * tcl


    @classmethod
    def _metric_row(cls, m: dict) -> dict:
        """
        Flatten a `negotiation_metrics` dict to the dashboard fields plus derived flags.

        Args:
            m (dict): Negotiation metrics.

        Returns:
            dict: `METRIC_KEYS` values with τ-invariant `ti`, plus `gpc_inf` and `auto`.
        """
        row = {k: m.get(k) for k in cls.METRIC_KEYS}
        row['ti'] = cls._ti(m)                        # τ-invariant GPC·TCL (not the scaled stored ti)
        row['gpc_inf'] = cls._is_inf(m.get('gpc'))
        row['auto'] = (m.get('negotiation_action') == 'auto')
        return row
    

    @staticmethod
    def _wait_stats(hours: list) -> dict:
        """
        Count / median / mean / p90 of waiting hours.

        Args:
            hours (list): Waiting hours.

        Returns:
            dict: `n`, `median_h`, `mean_h`, `p90_h` (stats None when empty).
        """
        if not hours:
            return {'n': 0, 'median_h': None, 'mean_h': None, 'p90_h': None}
        return {
            'n': len(hours), 
            'median_h': round(float(np.median(hours)), 1), 
            'mean_h': round(float(np.mean(hours)), 1),
            'p90_h': round(float(np.percentile(hours, 90)), 1)
        }
    

    @staticmethod
    def _cost_stats(costs: list) -> dict:
        """
        Counts of worse/better costs and their mean/median statistics.

        Args:
            costs (list): Per-booking costs (worse > 1e-6, better < -1e-6).

        Returns:
            dict: `n`, `worse`, `better`, `mean_all`, `mean_worse`, `median_worse` (stats None when undefined).
        """
        worse = [c for c in costs if c > 1e-6]
        return {
            'n': len(costs), 'worse': len(worse), 'better': sum(1 for c in costs if c < -1e-6),
            'mean_all': round(float(np.mean(costs)), 2) if costs else None,
            'mean_worse': round(float(np.mean(worse)), 2) if worse else None,
            'median_worse': round(float(np.median(worse)), 2) if worse else None,
        }
    

    @classmethod
    def _cohort_stats(cls, cohort: list[dict]) -> dict:
        """
        Outcome counts, friction and R / G / U statistics of a run's targeted patients.

        R / G / U are per group: `targeted` (all targets), `switched` (accepted / forced / auto), `unswitched` (kept).

        Args:
            cohort (list[dict]): Rows from `eligible_cohort`.

        Returns:
            dict: The statistics (group-keyed dicts for R / G / U and per-preference counts).
        """
        targets = [d for d in cohort if d['is_target']]
        accepted = [d for d in targets if d['negotiation_outcome'] == 'accepted']
        forced = [d for d in targets if d['negotiation_outcome'] == 'forced']
        kept = [d for d in targets if d['negotiation_outcome'] == 'kept']    # negotiated but the patient refused
        auto = [d for d in targets if d['negotiation_outcome'] == 'auto']
        concluded = accepted + forced + kept                                  # every negotiation that ran (auto excluded)
        switched = accepted + forced + auto                                   # booked the throughput_max schedule, so R is realized
        rounds = [d['negotiation_rounds'] or 0 for d in targets]

        groups = {'targeted': targets, 'switched': switched, 'unswitched': kept}   # unswitched = targeted - switched
        mean_of = lambda key, pref=None: {
            name: round(cls._mean([d[key] for d in rows if pref is None or d['preference'] == pref]), 3)
            for name, rows in groups.items()
        }
        sum_of = lambda key, pref=None: {
            name: round(sum(d[key] for d in rows if (pref is None or d['preference'] == pref) and d[key] is not None), 3)
            for name, rows in groups.items()
        }
        count_of = lambda pref: {name: sum(1 for d in rows if d['preference'] == pref) for name, rows in groups.items()}
        return {
            'n_eligible': len(cohort),
            'n_eligible_visit': sum(1 for d in cohort if d['preference'] == 'visit_min'),
            'n_eligible_stay': sum(1 for d in cohort if d['preference'] == 'stay_min'),
            'n_target': len(targets),
            'accepted': len(accepted),
            'forced': len(forced),
            'kept': len(kept),
            'auto': len(auto),
            'concluded': len(concluded),
            'acceptance_rate': round(len(accepted) / len(concluded) * 100, 2) if concluded else None,
            'switched': len(switched),
            'unswitched': len(kept),
            'n_visit': count_of('visit_min'),
            'n_stay': count_of('stay_min'),
            'sum_R': {name: round(sum(d['R'] for d in rows), 2) for name, rows in groups.items()},
            'mean_R': mean_of('R'),
            'sum_R_visit': sum_of('R', 'visit_min'),
            'mean_R_visit': mean_of('R', 'visit_min'),
            'sum_R_stay': sum_of('R', 'stay_min'),
            'mean_R_stay': mean_of('R', 'stay_min'),
            'sum_G_visit': sum_of('G', 'visit_min'),        # days (visit_min's concession unit)
            'mean_G_visit': mean_of('G', 'visit_min'),
            'sum_G_stay': sum_of('G', 'stay_min'),          # hours (stay_min's concession unit)
            'mean_G_stay': mean_of('G', 'stay_min'),
            'mean_U': mean_of('U'),
            'mean_U_pref': mean_of('U_pref'),
            'mean_U_thr': mean_of('U_thr'),
            'friction_total': sum(rounds),
            'friction_mean': round(cls._mean(rounds), 2),
            'rounds_dist': {str(k): v for k, v in sorted(Counter(rounds).items())},   # str keys for JSON
        }


    @classmethod
    def _device_bookings(cls, files: list[str]) -> dict[str, list[tuple[str, float, float]]]:
        """
        Every test booking in `files`, per device.

        Args:
            files (list[str]): Result files.

        Returns:
            dict[str, list[tuple[str, float, float]]]: `{device: [(date, start hour, end hour), ...]}`.
        """
        book = {}
        for fp in cls._iter_preds(files):
            for t in (fp.get('test') or []):
                dev, date, sch = t.get('device'), t.get('date'), t.get('schedule')
                if not (dev and date and sch):
                    continue
                book.setdefault(dev, []).append((date, float(sch[0]), float(sch[1])))
        return book

    
    @classmethod
    def eligible_cohort(cls, files: Union[list[str], str]) -> list[dict]:
        """
        Negotiation-eligible patients (GPC > 0, free wins included), whatever the run's policy did.

        `is_target` marks a `negotiate`/`auto` action with an executed outcome (not 'none', e.g. on the reasoning fallback).

        Args:
            files (Union[list[str], str]): Result file(s).

        Returns:
            list[dict]: `_metric_row` rows with `is_target`.
        """
        if isinstance(files, str):
            files = [files]
        rows = []
        for m in cls._iter_metrics(files):
            if m['gpc'] > 0:                          # inf (free win) included
                row = cls._metric_row(m)
                row['is_target'] = m['negotiation_action'] in ('negotiate', 'auto') \
                                   and m.get('negotiation_outcome') != 'none'
                rows.append(row)
        return rows
    

    @staticmethod
    def _protection_cost(pred: dict, tests: list, record: dict) -> Optional[dict]:
        """
        Cost of the protection to one booking vs. its unprotected counterfactual, on the tests both placed.

        Args:
            pred (dict): The booked prediction (`idle_waiting_time` read for `stay_min`).
            tests (list): The booking's live tests.
            record (dict): The booking's `capacity_protection` record (`objective`, `unprotected`).

        Returns:
            Optional[dict]: `objective`, `native` (in the objective's unit), `result_delay_h`, `extra_visit_days`; None if
                            not comparable (no common test, or `stay_min` with different test sets).
        """
        unprotected = record['unprotected']
        if not unprotected.get('all_results_ready_at'):
            return None
        objective = record['objective']

        # The schedule actually booked under the protection vs the one the same objective would have booked without it
        protected_test_schedule = {t['code']: t for t in tests}
        unprotected_test_schedule = unprotected['tests']
        common = set(protected_test_schedule) & set(unprotected_test_schedule)
        if not common:
            return None
        complete = common == set(protected_test_schedule) == set(unprotected_test_schedule)
        protected_ready = max((protected_test_schedule[c]['result_ready_at'] for c in common), key=str_to_datetime)
        unprotected_ready = max((unprotected_test_schedule[c]['result_ready_at'] for c in common), key=str_to_datetime)
        extra_days = len({protected_test_schedule[c]['date'] for c in common}) \
                        - len({unprotected_test_schedule[c]['date'] for c in common})

        cost = {
            'objective': objective,
            'result_delay_h': hours_between(protected_ready, unprotected_ready),
            'extra_visit_days': extra_days
        }
        if objective == 'stay_min':
            # Idle wait depends on every same-day neighbour, so it is only comparable on the full test set
            if not complete:
                return None
            cost['native'] = pred['idle_waiting_time'] - unprotected['idle_waiting_time']
        else:
            cost['native'] = cost['extra_visit_days'] if objective == 'visit_min' else cost['result_delay_h']
        return cost
    

    @staticmethod
    def _test_capacity(start_hour: float,
                       end_hour: float,
                       result_file: str,
                       agent_data_path: Optional[str] = None) -> dict:
        """
        Free test hours before any follow-up booking, pooled over each test's devices (closed days excluded).

        Args:
            start_hour (float): Opening hour of the operating window.
            end_hour (float): Closing hour of the operating window.
            result_file (str): The run's result file (`<dataset>/<run>/<hospital>_agent_result.json`).
            agent_data_path (Optional[str], optional): Agent data directory; `<dataset>/agent_data` if None. Defaults to None.

        Returns:
            dict: `{(test_code, date): Decimal free hours}`.

        Raises:
            FileNotFoundError: If the run's hospital data is not found.
        """
        if agent_data_path is None:
            agent_data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(result_file))), 'agent_data')
        hospital_id = os.path.basename(result_file).split('_agent_result')[0]
        source = os.path.join(agent_data_path, f'{hospital_id}_agent.json')
        if not os.path.isfile(source):
            raise FileNotFoundError(f"No hospital data for {result_file}: {source}")

        day_hours = Decimal(str(end_hour)) - Decimal(str(start_hour))
        capacity = defaultdict(Decimal)
        for tests in (json_load(source).get('test') or {}).values():
            for test in tests:
                for info in (test.get('devices') or {}).values():
                    for date, intervals in (info.get('schedule') or {}).items():
                        busy = sum((Decimal(str(e)) - Decimal(str(s)) for s, e in intervals), Decimal(0))
                        if busy < day_hours:
                            capacity[(test['code'], date)] += day_hours - busy
        return capacity



class FollowUpVisitEvaluator(NegotiationRecordMixin, Evaluator):
    """
    Follow-up-visit (OPFU test scheduling + negotiation) evaluation.
    """
    def __init__(self,
                 hospital_side_path: Optional[str] = None,
                 model_name: Optional[str] = None,
                 patient_side_path: Optional[str] = None,
                 interpolated_paths: Optional[list[str]] = None,
                 protection_paths: Optional[list[str]] = None,
                 agent_data_path: Optional[str] = None):
        """
        Load every run's result file and metadata and check the runs are compatible.

        Args:
            hospital_side_path (Optional[str], optional): Hospital-side (τ=0) results directory; needed for the dashboard and τ seeds. Defaults to None.
            model_name (Optional[str], optional): Model label for the dashboard meta block. Defaults to None.
            patient_side_path (Optional[str], optional): Patient-side (τ=∞) results directory (the ΔU baseline). Defaults to None.
            interpolated_paths (Optional[list[str]], optional): Interpolated results directories, in order. Defaults to None.
            protection_paths (Optional[list[str]], optional): Results directories for the capacity protection report. Defaults to None.
            agent_data_path (Optional[str], optional): Agent data directory for test capacity; each run's `<dataset>/agent_data` if None. Defaults to None.

        Raises:
            ValueError: If neither `hospital_side_path` nor `protection_paths` is given.
        """
        getcontext().prec = 10

        if not hospital_side_path and not protection_paths:
            raise ValueError('FollowUpVisitEvaluator needs `hospital_side_path` or `protection_paths`.')

        # The base evaluator's single run (`self.path` / `self.files`): the hospital side, else the first protection run
        super().__init__(hospital_side_path or protection_paths[0], False)

        # If hospital-side path is provided
        if hospital_side_path:
            self.hospital_file, _hospital_md = self._read_data(hospital_side_path)
            mds = [_hospital_md]
        else:
            self.hospital_file, mds = None, []

        # If patient-side path is provided
        if patient_side_path:
            self.patient_file, _patient_md = self._read_data(patient_side_path)
            mds.append(_patient_md)
        else:
            self.patient_file, _patient_md = None, None

        # Interpolated runs, kept in the given order (hospital side -> patient side)
        self.interpolated_files, self.interpolated_mds = [], []
        for path in interpolated_paths or []:
            interpolated_file, interpolated_md = self._read_data(path)
            self.interpolated_files.append(interpolated_file)
            self.interpolated_mds.append(interpolated_md)
        mds.extend(self.interpolated_mds)

        # Every run in dashboard order: hospital side -> interpolated runs as given -> patient side (when provided)
        self.all_files = ([self.hospital_file] if self.hospital_file else []) + self.interpolated_files \
                         + ([self.patient_file] if self.patient_file else [])

        # Protection runs, in the given order (`protection_report` groups them by mode and ratio itself)
        self.protection_files, self.protection_mds = [], []
        for path in protection_paths or []:
            protection_file, protection_md = self._read_data(path)
            self.protection_files.append(protection_file)
            self.protection_mds.append(protection_md)
        mds.extend(self.protection_mds)

        # Sanity check: all runs (dashboard and protection) must share the same operating window and time_unit
        self._sanity_check(*mds)

        # Store the operating window and time_unit from the metadata (identical across runs after the check)
        self.start_hour = float(mds[0]['start_hour'])
        self.end_hour = float(mds[0]['end_hour'])
        self.time_unit = float(mds[0]['time_unit'])
        self.model_name = model_name
        self.agent_data_path = agent_data_path

        log(f"Run metadata: window {self.start_hour}-{self.end_hour} (time_unit {self.time_unit})")


    @staticmethod
    def _read_data(path: str) -> tuple[str, dict]:
        """
        Find a run's result file and read its `_metadata` (pre-plugin results are rejected by `run_policies`).

        Args:
            path (str): Path to a run's results directory.

        Returns:
            tuple[str, dict]: The result file path and the `_metadata` dict.
        
        Raises:
            FileNotFoundError: If no `*_result.json` file is found under `path`.
            ValueError: If the `_metadata` block is missing in the result file.
        """
        result_files = sorted(get_files(path, '_result.json'))
        
        if not result_files:
            raise FileNotFoundError(f"No '*_result.json' under {path}")
        
        if len(result_files) > 1:
            log(f"Warning: multiple '*_result.json' files under {path}; using the first one: {result_files[0]}", level='warning')
        
        result_file = result_files[0]
        md = json_load(result_file).get('_metadata')
        
        if not md:
            raise ValueError(f"No '_metadata' in {result_file} — re-run the simulation or backfill it.")

        # Pre-plugin results (only the old `policy` block) must be converted first
        run_policies(md, where=result_file)
        
        return result_file, md
    

    @staticmethod
    def _sanity_check(*args) -> None:
        """
        Check every `_metadata` key except `policies`, `metrics` and `evaluations` is equal across runs.

        Args:
            *args (dict): `_metadata` dicts of the runs to compare.

        Raises:
            ValueError: If any non-policy metadata value differs between runs.
        """
        if len(args) < 2:
            return

        keys = sorted(set().union(*args) - {'policies', 'metrics', 'evaluations'})
        mismatches = {k: [md.get(k) for md in args] for k in keys}
        mismatches = {k: v for k, v in mismatches.items() if any(x != v[0] for x in v[1:])}

        if mismatches:
            detail = ', '.join(f"{k}: {v}" for k, v in mismatches.items())
            raise ValueError(f"Incompatible run metadata (policies excluded) — {detail}")
        
    
    def _device_delta_u(self) -> dict:
        """
        Per-device ΔU = compactness(run) - compactness(baseline) of every run in `self.all_files`.

        The baseline is the last run (patient side, else the last interpolated run); rows sort by hospital-side ΔU, descending.

        Returns:
            dict: `{'runs', 'baseline', 'rows'}`; each row's `u` / `du` / `n` align with `runs` (`rows` empty without baseline or data).
        """
        runs = list(self.all_files)                                         # dashboard order, baseline last
        if len(runs) < 2:
            log('delta_u needs a patient-side or interpolated run as its baseline; skipping.', level='warning')
            return {'runs': runs, 'baseline': None, 'rows': []}

        # The last run is the baseline: the patient side, or — without one — the last interpolated run
        baseline_file = runs[-1]
        if not self.patient_file:
            log(f'No patient-side run; using the last interpolated run as the device ΔU baseline: {baseline_file}', level='warning')

        run_bookings = [self._device_bookings([f]) for f in runs]
        initial_vacants = self._initial_vacant_slots()
        if not initial_vacants:
            log(colorstr('yellow', 'No device schedule found next to the results; skipping device ΔU.'))
            return {'runs': runs, 'baseline': baseline_file, 'rows': []}

        nseg = len(convert_time_to_segment(self.start_hour, self.end_hour, self.time_unit))
        rows = []
        for dev in sorted(set().union(*run_bookings)):
            initial_vacant = initial_vacants.get(dev)
            if not initial_vacant:
                continue
            
            dates = sorted(initial_vacant)                       # only the dates this device is open
            offset = {d: i * nseg for i, d in enumerate(dates)}
            universe = set()
            
            for d in dates:
                universe.update(offset[d] + s for s in initial_vacant[d])
            
            us = [self._compactness(b.get(dev, []), offset, universe) for b in run_bookings]
            u_base = us[-1]

            # A run (or the baseline) that never booked this device has no score; its ΔU is None, but the row stays
            rows.append({
                'dev': dev,
                'u': [round(u, 4) if u is not None else None for u in us],                # aligned with `runs`
                'du': [round(u - u_base, 4) if u is not None and u_base is not None else None for u in us],
                'n': [len(b.get(dev, [])) for b in run_bookings],
            })
        rows.sort(key=lambda r: (r['du'][0] is None, -(r['du'][0] or 0)))   # hospital-side ΔU descending, None last
        return {'runs': runs, 'baseline': baseline_file, 'rows': rows}


    def _initial_vacant_slots(self) -> dict[str, dict[str, set[int]]]:
        """
        Each device's vacant segments before the simulation, from the hospital data next to the hospital-side run.

        Fully closed dates are dropped and pre-occupied segments removed.

        Returns:
            dict[str, dict[str, set[int]]]: `{device: {date: vacant segment indices}}` (empty if no hospital data is found).
        """
        # Every compared run shares the hospital data next to the hospital-side result (`<root>/<run>/<hospital>_agent_result.json`)
        parent = os.path.dirname(os.path.dirname(os.path.abspath(self.hospital_file)))
        hospital_id = os.path.basename(self.hospital_file).split('_agent_result')[0]
        source = next((c for c in (os.path.join(parent, 'agent_data', f'{hospital_id}_agent.json'),
                                   os.path.join(parent, 'data', f'{hospital_id}.json'))
                       if os.path.isfile(c)), None)
        if source is None:
            return {}

        nseg = len(convert_time_to_segment(self.start_hour, self.end_hour, self.time_unit))
        avail = {}
        data = json_load(source)
        for tests in (data.get('test') or {}).values():
            for test in tests:
                for device, info in (test.get('devices') or {}).items():
                    per_date = avail.setdefault(device, {})
                    for date, intervals in (info.get('schedule') or {}).items():
                        occupied = set()
                        for interval in intervals:
                            occupied.update(convert_time_to_segment(self.start_hour, self.end_hour, self.time_unit, interval))
                        free = set(range(nseg)) - occupied
                        if free:                    # skip dates the device is closed all day
                            per_date[date] = free
        return avail

    
    def _compactness(self,
                     bookings: list[tuple[str, float, float]],
                     offset: dict[str, int],
                     universe: set[int]) -> Optional[float]:
        """
        Mann-Whitney front-loading score of a device's bookings over its vacant slots (1.0 = packed early).

        Bookings off the vacant slots are a simulator bug (KeyError on a closed date, AssertionError on overlap).

        Args:
            bookings (list[tuple[str, float, float]]): `(date, start hour, end hour)` bookings of the device.
            offset (dict[str, int]): Global slot offset of each open date.
            universe (set[int]): Global indices of the device's vacant slots.

        Returns:
            Optional[float]: The score, 1.0 if no free slot remains, None without bookings.
        """
        booked = set()
        for date, s, e in bookings:
            segments = convert_time_to_segment(self.start_hour, self.end_hour, self.time_unit, [s, e])
            booked.update(offset[date] + i for i in segments)      # KeyError if booked on a closed date
        assert booked <= universe, 'A simulated booking overlaps a pre-filled device slot.'
        if not booked:
            return None
        free = universe - booked
        if not free:
            return 1.0
        good, total = frontload_pairs(booked, free)
        return good / total


    def interpolation_taus(self, tau_n: int = 4) -> dict[str, list[float]]:
        """
        Evenly-ranked quantiles of the hospital-side finite ``GPC·TCL`` > 0 per preference, as τ seeds for interpolation runs.

        Only seeds: ti shifts once a run at that τ re-books everyone.

        Args:
            tau_n (int, optional): Number of τ candidates per preference. Defaults to 4.

        Returns:
            dict[str, list[float]]: ``{'visit_min': [...], 'stay_min': [...]}`` with up to `tau_n` τ each.
        """
        by_pref = {'visit_min': [], 'stay_min': []}
        for m in self._iter_metrics([self.hospital_file]):
            pref, ti = m.get('preference'), self._ti(m)   # ti = GPC·TCL (τ-invariant)
            if pref in by_pref and ti is not None and 0 < ti < float('inf'):
                by_pref[pref].append(round(float(ti), 6))

        division = tau_n + 1
        result = {}
        for pref, vals in by_pref.items():
            vals.sort()
            if not vals:
                result[pref] = []
                continue
            idx = sorted({min(len(vals) - 1, round(len(vals) * i / division)) for i in range(1, division)})
            result[pref] = [vals[k] for k in idx]
        
        log(f'--------------Interpolation tau seeds (tau_n={tau_n})--------------')
        for pref, vals in result.items():
            log(f'{pref:<12} ({len(vals)}): {vals}')

        return result


    def dashboard_data(self, save: bool = False) -> dict:
        """
        Compute and log the negotiation dashboard values for every run in `self.all_files` order.

        Args:
            save (bool, optional): If True, save to `dashboard_data.json` in the dataset folder. Defaults to False.

        Returns:
            dict: ``eligible`` (per-run cohorts), ``devices`` (`_device_delta_u`), ``stats`` (per-run cohort and device summaries), ``meta``.
        """
        # Per-run eligible cohorts and their stats, in `self.all_files` order (hospital -> interpolated -> patient)
        cohorts = [self.eligible_cohort(f) for f in self.all_files]
        stats = {
            'runs': list(self.all_files),
            'cohort': [{'n_total': sum(1 for _ in self._iter_metrics([f])), **self._cohort_stats(c)}
                    for f, c in zip(self.all_files, cohorts)],
        }

        # Per-run device ΔU summary, aligned with `devices['runs']` (= `self.all_files`; the baseline's own ΔU is 0)
        devices = self._device_delta_u()
        stats['device'] = []
        for k in range(len(devices['runs'])):
            dus = [r['du'][k] for r in devices['rows'] if r['du'][k] is not None]
            stats['device'].append({
                'n_devices': len(dus),
                'delta_u_mean': round(self._mean(dus), 4),
                'n_more_frontloaded': sum(1 for x in dus if x > 1e-9),
            })

        payload = {
            'eligible': {
                'runs': list(self.all_files),                     # eligible['cohort'][k] belongs to this run
                'cohort': cohorts,
            },
            'devices': devices,
            'meta': {
                'model': self.model_name,
                'operating_window': [self.start_hour, self.end_hour],
                'time_unit': self.time_unit,
            },
            'stats': stats,
        }

        # Logging the dashboard summary, one block per run in `self.all_files` order
        device_by_run = dict(zip(devices['runs'], stats['device']))
        baseline_name = os.path.basename(os.path.dirname(devices['baseline'])) if devices['baseline'] else None
        log('--------------Follow-up Negotiation Dashboard--------------')
        for k, (file, s) in enumerate(zip(self.all_files, stats['cohort'])):
            role = 'hospital-side' if k == 0 else 'patient-side' if file == self.patient_file else 'interpolated'
            log(f'{colorstr(role)} ({os.path.basename(os.path.dirname(file))})')
            log(f'- cohort')
            log(f'    - booked: {colorstr("green", str(s["n_total"]))}')
            by_pref = lambda n_visit, n_stay: f' | visit_min: {colorstr("green", str(n_visit))}, stay_min: {colorstr("green", str(n_stay))}'
            log(f'    - eligible: {colorstr("green", str(s["n_eligible"]))} / {colorstr("green", str(s["n_total"]))}'
                + by_pref(s['n_eligible_visit'], s['n_eligible_stay']))
            targeted_pct = f'{s["n_target"] / s["n_eligible"] * 100:.2f}%' if s['n_eligible'] else 'n/a'
            log(f'    - targeted: {colorstr("green", str(s["n_target"]))} / {colorstr("green", str(s["n_eligible"]))} '
                f'(accepted: {colorstr("green", str(s["accepted"]))}, forced: {colorstr("red", str(s["forced"]))}, '
                f'kept: {colorstr("cyan", str(s["kept"]))}, auto: {colorstr("yellow", str(s["auto"]))}) '
                f'| {colorstr("green", targeted_pct)} of eligible'
                + by_pref(s['n_visit']['targeted'], s['n_stay']['targeted']))
            switched_pct = f'{s["switched"] / s["n_target"] * 100:.2f}%' if s['n_target'] else 'n/a'
            log(f'    - switched: {colorstr("green", str(s["switched"]))} / {colorstr("green", str(s["n_target"]))} '
                f'(accepted: {colorstr("green", str(s["accepted"]))}, forced: {colorstr("red", str(s["forced"]))}, '
                f'auto: {colorstr("yellow", str(s["auto"]))}) | {colorstr("green", switched_pct)} of targeted'
                + by_pref(s['n_visit']['switched'], s['n_stay']['switched']))
            log(f'    - unswitched: {colorstr("green", str(s["unswitched"]))} / {colorstr("green", str(s["n_target"]))} '
                f'(kept: {colorstr("cyan", str(s["kept"]))})'
                + by_pref(s['n_visit']['unswitched'], s['n_stay']['unswitched']))

            # Undefined values are logged as n/a with the reason, rather than as misleading zeros or a missing line
            if s['acceptance_rate'] is not None:
                log(f'- acceptance rate : {colorstr("green", str(s["acceptance_rate"]) + "%")} '
                    f'({s["accepted"]} / {s["concluded"]} concluded = '
                    f'accepted {colorstr("green", str(s["accepted"]))} + forced {colorstr("red", str(s["forced"]))} + kept {colorstr("cyan", str(s["kept"]))})')
            else:
                log(f'- acceptance rate : n/a (no negotiation concluded)')

            log(f'- ΔR (result-hours gained)')
            for group in ('targeted', 'switched', 'unswitched'):
                log(f'    - {group}: {colorstr("green", str(s["sum_R"][group]) + "h")} '
                    f'(mean {colorstr("green", str(s["mean_R"][group]) + "h")} / {group})')
                log(f'        - visit_min: {colorstr("green", str(s["sum_R_visit"][group]) + "h")} '
                    f'(mean {colorstr("green", str(s["mean_R_visit"][group]) + "h")} / {s["n_visit"][group]} patients)')
                log(f'        - stay_min: {colorstr("green", str(s["sum_R_stay"][group]) + "h")} '
                    f'(mean {colorstr("green", str(s["mean_R_stay"][group]) + "h")} / {s["n_stay"][group]} patients)')
            # visit_min concedes days and stay_min hours, so a group's ΔG is given per unit rather than as one total
            log(f'- ΔG (patient concession)')
            for group in ('targeted', 'switched', 'unswitched'):
                log(f'    - {group}: {colorstr("red", str(s["sum_G_visit"][group]) + " days")} + '
                    f'{colorstr("red", str(s["sum_G_stay"][group]) + "h")} '
                    f'(mean {colorstr("red", str(s["mean_G_visit"][group]) + " days")}, '
                    f'{colorstr("red", str(s["mean_G_stay"][group]) + "h")} / {group})')
                log(f'        - visit_min: {colorstr("red", str(s["sum_G_visit"][group]) + " days")} '
                    f'(mean {colorstr("red", str(s["mean_G_visit"][group]) + " days")} / {s["n_visit"][group]} patients)')
                log(f'        - stay_min: {colorstr("red", str(s["sum_G_stay"][group]) + "h")} '
                    f'(mean {colorstr("red", str(s["mean_G_stay"][group]) + "h")} / {s["n_stay"][group]} patients)')
            log(f'- mean ΔU (per patient, P -> T)')
            for group in ('targeted', 'switched', 'unswitched'):
                log(f'    - {group}: {colorstr("green", str(s["mean_U"][group]))}  '
                    f'(U_pref {colorstr("red", str(s["mean_U_pref"][group]))} -> U_thr {colorstr("green", str(s["mean_U_thr"][group]))})')
            log(f'- sum F (conversation friction): {colorstr("red", str(s["friction_total"]) + " rounds")} (mean {colorstr("red", str(s["friction_mean"]))} / target)')
            log(f'    - dist: {s["rounds_dist"]}')

            dev = device_by_run[file]
            log(f'- device ΔU vs {baseline_name} : {dev["n_devices"]} devices, mean {dev["delta_u_mean"]:+.4f}, '
                f'{dev["n_more_frontloaded"]} more front-loaded')
            log('')            

        # Save dashboard data
        if save:
            parent = os.path.dirname(os.path.dirname(os.path.abspath(self.hospital_file)))
            save_path = os.path.join(parent, 'dashboard_data.json')
            json_save_fast(save_path, encode_non_finite(payload))     # free-win GPC / ti saved as 'inf', not null
            log(f'Dashboard data saved to {colorstr("green", save_path)}')
        
        return payload
    

    def _protection_summary(self, result_file: str, window: dict) -> dict:
        """
        Urgent capacity protection outcome of one run, over live (uncancelled) bookings.

        `urgent` / `normal`: failures, partial bookings and waits (booking -> first test start / all results ready).
        `normal_cost`: cost to blocked normal bookings vs. their unprotected schedule. `window`: test hours booked
        inside `[release_hours, horizon_days * 24)` hours of booking, with a lead-time profile. `capacity`: utilization
        of the start-of-run test capacity, overall and over the first `ceil(horizon_days)` dates.

        Args:
            result_file (str): The run's result file.
            window (dict): Rolling window with `horizon_days` and `release_hours` (the run's own, or borrowed for a baseline).

        Returns:
            dict: `{'urgent', 'normal', 'normal_cost', 'window', 'capacity'}`.
        """
        # Keep only the window keys, so a borrowed protection's `ratio` does not leak into a baseline's report
        window = {k: window[k] for k in ('horizon_days', 'release_hours')}
        horizon_h = window['horizon_days'] * 24                 # protection horizon in hours (also the first-test access bound)
        horizon_dates = math.ceil(window['horizon_days'])       # number of dates the horizon spans (the early capacity view)
        # Lead-time bin edges: 24h steps up to a week, split at the window bounds so every bin is in one zone
        lead_edges = sorted({24 * k for k in range(1, 8)} | {window['release_hours'], horizon_h} - {0})

        end_waits, start_waits, failed, partial = defaultdict(list), defaultdict(list), Counter(), Counter()
        blocked_any, uncompared, native, delay_h, extra_days = 0, 0, defaultdict(list), [], []
        window_hours = Counter()
        lead_hours = defaultdict(Counter)                       # lead-bin index -> {urgency: Decimal hours}
        booked, booked_urgent = defaultdict(Decimal), defaultdict(Decimal)     # exact hour sums (no float drift)

        for gt, pred, status in self._booking_rows([result_file]):
            ok = status is True and isinstance(pred, dict)
            
            # Tests still held at the end of the run: a booked, uncancelled booking's placed, uncancelled tests
            if ok and pred.get('status') != 'cancelled':          # some booked preds carry no `status` key
                live = [t for t in pred.get('test') or [] if t.get('schedule') and t.get('status') != 'cancelled']
            else:
                live = []
            
            # Final occupancy: every live booking, including system reschedules
            for t in live:
                hours = Decimal(str(t['schedule'][1])) - Decimal(str(t['schedule'][0]))
                booked[(t['code'], t['date'])] += hours
                if pred.get('urgency') == 'urgent':
                    booked_urgent[(t['code'], t['date'])] += hours

            if not isinstance(gt, dict):
                continue
            
            urgency = gt.get('urgency')
            
            # Failed case
            if not ok:
                failed[urgency] += 1
                continue
            
            # Partial case
            if len(pred.get('test') or []) < len(gt.get('test') or []):
                partial[urgency] += 1

            # Full booking case
            tests = [t for t in live if t.get('result_ready_at')]
            booked_at = pred.get('last_updated_time')
            if not tests or not booked_at:
                continue
            booked_dt = str_to_datetime(booked_at)
            utc_offset = booked_dt.strftime('%z')       # the booking's own offset (e.g. '+0900'), so test times share its timezone
            ready = max((t['result_ready_at'] for t in tests), key=str_to_datetime)
            end_waits[urgency].append(hours_between(ready, booked_dt))
            start_waits[urgency].append(min(hours_between(get_iso_time(t['schedule'][0], t['date'], utc_offset), booked_dt) for t in tests))

            # Test hours by lead time at booking time: inside the protection window, and per lead bin
            for t in tests:
                lead = hours_between(get_iso_time(self.start_hour, t['date'], utc_offset), booked_dt)
                hours = Decimal(str(t['schedule'][1])) - Decimal(str(t['schedule'][0]))
                if window['release_hours'] <= lead < horizon_h:
                    window_hours[urgency] += hours
                lead_hours[bisect_right(lead_edges, lead)][urgency] += hours    # a same-day test (lead < 0) falls in the first bin

            # What the protection cost this booking against its unprotected counterfactual
            record = pred.get('capacity_protection') or {}
            if record.get('blocked'):
                blocked_any += 1
                cost = self._protection_cost(pred, tests, record)
                if cost is None:
                    uncompared += 1
                else:
                    native[cost['objective']].append(cost['native'])
                    delay_h.append(cost['result_delay_h'])
                    extra_days.append(cost['extra_visit_days'])

        # Per-urgency waits: booking -> first test start (`start`) and -> all results ready (`end`)
        summary = {
            urgency: {
                'failed': failed[urgency],
                'partial': partial[urgency],
                'start': {
                    **self._wait_stats(start_waits[urgency]),
                    'within_hours': horizon_h,
                    'within_share': round(float(np.mean(np.array(start_waits[urgency]) < horizon_h)), 4)
                                    if start_waits[urgency] else None,
                },
                'end': self._wait_stats(end_waits[urgency]),
            }
            for urgency in ('urgent', 'normal')
        }
        summary['normal_cost'] = {
            'blocked_any': blocked_any,
            'uncompared': uncompared,
            'by_objective': {obj: {'unit': self.PROTECTION_COST_UNITS.get(obj), **self._cost_stats(costs)}
                             for obj, costs in sorted(native.items())},
            'result_delay_h': self._cost_stats(delay_h),
            'extra_visit_days': self._cost_stats(extra_days),
        }

        # Exact Decimal sums, turned into floats only for the report (JSON-serializable)
        window_total = window_hours['urgent'] + window_hours['normal']
        lead_profile = []
        for i, (lo, hi) in enumerate(zip([None] + lead_edges, lead_edges + [None])):    # None = open end
            zone = 'released' if hi is not None and hi <= window['release_hours'] \
                else 'protected' if hi is not None and lo is not None and lo >= window['release_hours'] and hi <= horizon_h \
                else 'beyond'
            total = lead_hours[i]['urgent'] + lead_hours[i]['normal']
            lead_profile.append({
                'from_h': lo, 'to_h': hi, 'zone': zone,
                'booked_h': {u: round(float(lead_hours[i][u]), 2) for u in ('urgent', 'normal')},
                'urgent_share': round(float(lead_hours[i]['urgent'] / total), 4) if total else None,
            })
        summary['window'] = {
            **window,
            'booked_h': {u: round(float(window_hours[u]), 2) for u in ('urgent', 'normal')},
            'urgent_share': round(float(window_hours['urgent'] / window_total), 4) if window_total else None,
            'lead_profile': lead_profile,
        }

        # Utilization of the start-of-run capacity, over all dates and over the first `horizon_dates` test dates;
        # every sum is an exact Decimal, turned into a float only for the report (JSON-serializable)
        capacity = self._test_capacity(self.start_hour, self.end_hour, result_file, self.agent_data_path)
        early_dates = set(sorted({d for _, d in capacity})[:horizon_dates])
        early = [k for k in capacity if k[1] in early_dates]
        capacity_all = sum(capacity.values(), Decimal(0))
        capacity_early = sum((capacity[k] for k in early), Decimal(0))
        booked_early = sum((booked[k] for k in early), Decimal(0))
        summary['capacity'] = {
            'utilization_all': round(float(sum((booked[k] for k in capacity), Decimal(0)) / capacity_all), 4) if capacity_all else None,
            'utilization_early': round(float(booked_early / capacity_early), 4) if capacity_early else None,
            'urgent_share_early': round(float(sum((booked_urgent[k] for k in early), Decimal(0)) / booked_early), 4) if booked_early else None,
            'early_dates': sorted(early_dates),
        }
        return summary


    def protection_report(self, save: bool = False) -> list[dict]:
        """
        Compare the urgent capacity protection across the protection runs, grouped by mode and sorted by ratio.

        A run without protection is measured over the window of a protected run of its mode (else of any mode).

        Args:
            save (bool, optional): If True, save to `protection_report.json` in the dataset folder. Defaults to False.

        Returns:
            list[dict]: One `{'label', 'mode', 'capacity_protection', 'policies', **protection_summary}` per run.

        Raises:
            ValueError: If no run has capacity protection.
        """
        runs = []
        for path, md in zip(self.protection_files, self.protection_mds):
            summary = run_policy_summary(md, where=path)
            runs.append((path, summary['policies'], summary['mode'], summary['capacity_protection']))

        # Window of the protected runs, borrowed by the baselines (same mode first)
        reference = {}
        for _, _, mode, protection in runs:
            if protection:
                reference.setdefault(mode, protection)
                reference.setdefault(None, protection)

        if not reference:
            raise ValueError(colorstr("red", 'No runs with capacity protection found; cannot report.'))

        # Compute the protection summary
        report = []
        for path, policies, mode, protection in runs:
            window = protection or reference.get(mode) or reference.get(None)
            label = f"{mode} " + (f"ratio={protection['ratio']}" if protection else 'no protection')
            report.append({'label': label, 'mode': mode, 'capacity_protection': protection,
                           'policies': policies,
                           **self._protection_summary(path, window)})
        report.sort(key=lambda r: (r['mode'] or '', (r['capacity_protection'] or {}).get('ratio', 0.0)))

        # Logging the protection report, one block per run (grouped by mode, ratio ascending)
        def val(v: Any, unit: str = '', color: str = 'green') -> str:
            """
            A logged value colored with its unit, or `n/a` when None.

            Args:
                v (Any): The value.
                unit (str, optional): Unit suffix. Defaults to ''.
                color (str, optional): Color name. Defaults to 'green'.

            Returns:
                str: The formatted value.
            """
            return 'n/a' if v is None else colorstr(color, f'{v}{unit}')

        log('--------------Urgent capacity protection--------------')
        for r in report:
            c, w, cap = r['normal_cost'], r['window'], r['capacity']
            log(f'{colorstr(r["label"])}')

            # Waits per urgency
            for name in ('urgent', 'normal'):
                x = r[name]
                s, e = x['start'], x['end']
                log(f'- {colorstr("cyan", name)}: {val(e["n"])} bookings (failed:{val(x["failed"], color="red")}, '
                    f'partial: {val(x["partial"], color="yellow")})')
                share = None if s['within_share'] is None else round(s['within_share'] * 100, 2)
                log(f'    - A_H (first test started within the protection horizon): {val(share, "%")} ({s["within_hours"]}h)')
                log(f'    - W_start (booking -> first test start)')
                log(f'        - median: {val(s["median_h"], "h")}')
                log(f'        - mean: {val(s["mean_h"], "h")}')
                log(f'    - W_result (booking -> all results ready)')
                log(f'        - median: {val(e["median_h"], "h")}')
                log(f'        - mean: {val(e["mean_h"], "h")}')
                log(f'        - p90: {val(e["p90_h"], "h")}')

            # The rolling window the protection governs, and the capacity it used
            log(f'- {colorstr("cyan", "protection window")} ({w["horizon_days"]} days ahead, released {w["release_hours"]}h before opening)')
            log(f'    - S_urg (urgent share of window hours): {val(w["urgent_share"])} '
                f'(urgent {val(w["booked_h"]["urgent"], "h")}, normal {val(w["booked_h"]["normal"], "h")})')
            log(f'    - lead profile (urgent share of booked hours by lead time at booking)')
            hour = lambda h: f'{h:g}h'
            for b in w['lead_profile']:
                span = f'< {hour(b["to_h"])}' if b['from_h'] is None else f'>= {hour(b["from_h"])}' if b['to_h'] is None \
                    else f'{hour(b["from_h"])}-{hour(b["to_h"])}'
                log(f'        - {span:<10} [{b["zone"]:<9}]: {val(b["urgent_share"])} '
                    f'(urgent {val(b["booked_h"]["urgent"], "h")}, normal {val(b["booked_h"]["normal"], "h")})')
            log(f'- {colorstr("cyan", "capacity utilization")}')
            log(f'    - η_all (all dates): {val(cap["utilization_all"])}')
            log(f'    - η_early (first {len(cap["early_dates"])} dates: {", ".join(cap["early_dates"])}): '
                f'{val(cap["utilization_early"])}')
            log(f'    - S_early (urgent share of the first {len(cap["early_dates"])} dates): {val(cap["urgent_share_early"])}')

            # What the protection cost the normal bookings it held back
            if c['blocked_any']:
                log(f'- {colorstr("cyan", "κ (normal cost)")} (blocked:{val(c["blocked_any"], color="red")}, '
                    f'uncompared: {val(c["uncompared"], color="yellow")})')
                for obj, s in c['by_objective'].items():
                    log(f'    - {obj} ({s["unit"]})')
                    log(f'        - worse: {val(s["worse"], color="red")} / {val(s["n"])} '
                        f'(mean {val(s["mean_worse"], color="red")}, median {val(s["median_worse"], color="red")})')
                    log(f'        - better: {val(s["better"])}')
                    log(f'        - mean over all: {val(s["mean_all"], color="red")}')
                d, e = c['result_delay_h'], c['extra_visit_days']
                log(f'    - all objectives')
                log(f'        - result delay: worse {val(d["worse"], color="red")} / {val(d["n"])} '
                    f'(mean {val(d["mean_worse"], "h", "red")}, median {val(d["median_worse"], "h", "red")}), '
                    f'better {val(d["better"])}, mean over all {val(d["mean_all"], "h", "red")}')
                log(f'        - extra visit days: more {val(e["worse"], color="red")}, fewer {val(e["better"])}, '
                    f'mean over all {val(e["mean_all"], color="red")}')
            log('')
        
        # Save protection report
        if save:
            parent = os.path.dirname(os.path.dirname(os.path.abspath(self.protection_files[0])))
            save_path = os.path.join(parent, 'protection_report.json')
            json_save_fast(save_path, encode_non_finite(report))
            log(f'Protection report saved to {colorstr("green", save_path)}')
        
        return report
