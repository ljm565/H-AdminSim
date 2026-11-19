import os
import math
import numpy as np
from bisect import bisect_right
from decimal import Decimal, getcontext
from typing import Optional, Tuple, Union
from collections import Counter, defaultdict

from h_adminsim.utils import log, colorstr
from h_adminsim.utils.common_utils import *
from h_adminsim.tools.negotiation_metrics import frontload_pairs
from h_adminsim.utils.image_preprocess_utils import draw_fail_donut_subplots
from h_adminsim.utils.filesys_utils import get_files, json_load, json_save_fast



class Evaluator:
    """
    Base evaluator: file loading, generic accuracy / token-cost / supervisor / human evaluation shared
    across tasks. Task-specific evaluation lives in the child classes (`FirstVisitEvaluator`,
    `FollowUpVisitEvaluator`).
    """
    def __init__(self, path, human_eval=False):
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
        Perform micro-wise evaluation on the aggregated results.
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
        Estimate and print API cost from token usage statistics per task.

        Args:
            model_name (str): Model name. One of 'gpt-5-nano', 'gpt-5-mini', 'gemini-2.5-flash'.
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


    def supervisor_evaluation(self):
        """
        Evaluate the supervisor's necessity to intervene in tasks.
        """
        aggregated_results = dict()
        for file in self.files:
            data = json_load(file)

            for task, value in data.items():
                if not task in aggregated_results:
                    aggregated_results[task] = {'status': [], 'trial': []}

                aggregated_results[task]['status'].append(value['status'])
                aggregated_results[task]['trial'].append(value['trial'])

        log('-----Supervisor (or feedback) Evaluation----')
        for task, value in aggregated_results.items():
            status = sum(value['status'], [])
            trial = sum(value['trial'], [])

            if task == 'first_visit_intake':
                total_length = len(status)
                supervisor_effect_cnt, correct, error, tie = 0, 0, 0, 0
                for t in trial:
                    if 'mismatch' in t[0]:
                        supervisor_effect_cnt += 1
                        if 'better' in t[0]:
                            correct += 1
                        elif 'worse' in t[0]:
                            error += 1
                        else:
                            tie += 1

                correct_p = correct/supervisor_effect_cnt*100 if supervisor_effect_cnt > 0 else 0
                error_p = error/supervisor_effect_cnt*100 if supervisor_effect_cnt > 0 else 0
                tie_p = tie/supervisor_effect_cnt*100 if supervisor_effect_cnt > 0 else 0
                log(f'{colorstr(task):<27} | length: {total_length}, effected: {supervisor_effect_cnt} ({(supervisor_effect_cnt/total_length)*100:.2f}%)')
                log(f'    - {colorstr("green", "correct")}: {correct} ({correct_p:.2f}%), {colorstr("red", "worse")}: {error} ({error_p:.2f}%), {colorstr("yellow", "tie")}: {tie} ({tie_p:.2f}%)')

            elif task == 'first_visit_scheduling':
                feedback_n = dict()
                total_length = len(status)
                supervisor_effect_cnt, correct, tie = 0, 0, 0
                for t in trial:
                    if isinstance(t, list) and len(t) > 1:
                        supervisor_effect_cnt += 1
                        if t[-1] == 'pass':
                            correct += 1
                            feedback_n[len(t)-1] = feedback_n.setdefault(len(t)-1, 0) + 1
                        else:
                            tie += 1

                desc = ', '.join([f'{f}-feedback: {n}' for f, n in sorted(feedback_n.items())])
                correct_p = correct/supervisor_effect_cnt*100 if supervisor_effect_cnt > 0 else 0
                tie_p = tie/supervisor_effect_cnt*100 if supervisor_effect_cnt > 0 else 0
                log(f'{colorstr(task):<27} | length: {total_length}, effected: {supervisor_effect_cnt} ({(supervisor_effect_cnt/total_length)*100:.2f}%)')
                log(f'    - {colorstr("green", "correct")}: {correct} ({correct_p:.2f}%), {colorstr("yellow", "tie")}: {tie} ({tie_p:.2f}%)')
                log(f'    - Feedback distribution: {desc}')


    def human_evaluation(self):
        """
        Aggregate and evaluate human evaluation results from text files.
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
        Evaluate solely department prediction accuracy.
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
        Calculate average required intake rounds
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
    Read and interpret the follow-up task's stored records across any list of result files, so the same helpers
    serve every run (hospital-side, interpolated, patient-side, protection runs): each patient's final prediction,
    its ``negotiation_metrics`` (cohorts and their statistics), and its ``capacity_protection`` record (the
    protection's cost against the unprotected counterfactual), plus the generic wait / cost statistics and the
    run's start-of-simulation test capacity. Holds no run state; the files (and the operating window, where
    needed) are always passed in.
    """
    TASK = 'follow_up_visit_scheduling'
    METRIC_KEYS = ('preference', 'gpc', 'tcl', 'ti', 'G', 'R', 'U', 'U_pref', 'U_thr',
                   'negotiation_action', 'negotiation_outcome', 'negotiation_rounds')
    # Unit of `_protection_cost`'s `native` cost per scheduling objective (the quantity that objective minimizes)
    PROTECTION_COST_UNITS = {'throughput_max': 'result_delay_h', 'visit_min': 'extra_visit_days', 'stay_min': 'extra_idle_wait_h'}


    @classmethod
    def _booking_rows(cls, files):
        """
        `(gt, final pred, status)` of every follow-up row of this run. `gt` is the patient condition dict for a
        regular booking and a plain string for system rows (e.g. automatic waiting-list rescheduling).
        """
        for file in files:
            value = json_load(file).get(cls.TASK) or {}
            for gt, pred, status in zip(value.get('gt', []), value.get('pred', []), value.get('status', [])):
                yield gt, cls._final_pred(pred), cls._final_pred(status)


    @staticmethod
    def _final_pred(pred):
        """A prediction is a dict, or a list of retries; take the last (booked) attempt."""
        if isinstance(pred, list):
            return pred[-1] if pred else None
        return pred


    @classmethod
    def _iter_preds(cls, files):
        """Yield each patient's final prediction dict for the follow-up task across `files` (failed rows skipped)."""
        for _, fp, _ in cls._booking_rows(files):
            if isinstance(fp, dict):
                yield fp


    @classmethod
    def _iter_metrics(cls, files):
        """
        Yield each prediction's stored `negotiation_metrics` dict, with the `'inf'` / `'-inf'` strings the
        simulator writes for infinities (JSON has none) restored to floats — so a free win reads back as
        `gpc == inf`, while None stays "not computed".
        """
        restore = {'inf': float('inf'), '-inf': float('-inf')}
        for fp in cls._iter_preds(files):
            m = fp.get('negotiation_metrics')
            if isinstance(m, dict):
                yield {k: (restore[v] if isinstance(v, str) and v in restore else v) for k, v in m.items()}


    @staticmethod
    def _is_inf(x) -> bool:
        """Whether `x` is an infinite number (a free win's GPC / trigger index)."""
        return isinstance(x, (int, float)) and math.isinf(x)

    
    @staticmethod
    def _mean(xs):
        xs = [x for x in xs if x is not None]
        return sum(xs) / len(xs) if xs else 0.0
    

    @classmethod
    def _ti(cls, m):
        """
        τ-invariant trigger index ``ti = GPC·TCL``, recomputed from the stored GPC and TCL rather than
        read from the stored ``ti`` field. The stored ``ti`` is ``GPC·TCL / trigger_temperature``, so it
        is scaled per run/preference; recomputing keeps it comparable across runs (e.g. interpolation
        runs with trigger_temperature != 1). ``inf`` for a free win (regardless of TCL, as the policy
        computes it). Both values are required — a record missing either raises, so a malformed
        result surfaces instead of being skipped.
        """
        gpc, tcl = m['gpc'], m['tcl']
        if cls._is_inf(gpc):
            return gpc
        return gpc * tcl


    @classmethod
    def _metric_row(cls, m):
        """Flatten a `negotiation_metrics` dict to the dashboard fields (+ derived flags)."""
        row = {k: m.get(k) for k in cls.METRIC_KEYS}
        row['ti'] = cls._ti(m)                        # τ-invariant GPC·TCL (not the scaled stored ti)
        row['gpc_inf'] = cls._is_inf(m.get('gpc'))
        row['auto'] = (m.get('negotiation_action') == 'auto')
        return row
    

    @staticmethod
    def _wait_stats(hours: list) -> dict:
        """Count / median / mean / p90 of a list of waiting hours (n=0 and None stats when the list is empty)."""
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
        """Count / worse count, and mean / median over all compared bookings and over the worse ones only."""
        worse = [c for c in costs if c > 1e-6]
        return {
            'n': len(costs), 'worse': len(worse), 'better': sum(1 for c in costs if c < -1e-6),
            'mean_all': round(float(np.mean(costs)), 2) if costs else None,
            'mean_worse': round(float(np.mean(worse)), 2) if worse else None,
            'median_worse': round(float(np.median(worse)), 2) if worse else None,
        }
    

    @classmethod
    def _cohort_stats(cls, cohort):
        """
        Aggregate the tile/trade-off statistics of a run's eligible cohort (`eligible_cohort`) over the patients
        the policy targeted (`is_target`) — all of them on the hospital side, a subset at a threshold τ, none on
        the patient side. Outcome counts and friction cover every target. R / G / U are reported for three groups,
        each as `{'targeted': ..., 'switched': ..., 'unswitched': ...}`: `targeted` is every target (what the policy
        proposed, before refusals filter it), `switched` the targets that took the throughput_max schedule
        (accepted / forced / auto) — what was realized — and `unswitched` the rest (kept), whose proposal was refused.
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
    def _device_bookings(cls, files):
        """`{device: [(date, start, end), ...]}` from every test booking in a run."""
        book = {}
        for fp in cls._iter_preds(files):
            for t in (fp.get('test') or []):
                dev, date, sch = t.get('device'), t.get('date'), t.get('schedule')
                if not (dev and date and sch):
                    continue
                book.setdefault(dev, []).append((date, float(sch[0]), float(sch[1])))
        return book

    
    @classmethod
    def eligible_cohort(cls, files: Union[list[str], str]):
        """
        Negotiation-eligible cohort from any run: patients with something to gain (GPC > 0, free win
        included), whatever the run's policy did with them. `is_target` marks the ones the policy actually
        targeted (`negotiate`/`auto`) — every eligible patient on the hospital side, a subset at a
        threshold τ, none on the patient side. A target must also have an executed outcome: on the
        reasoning fallback (tool calling failed) the policy's action is recorded but never run, leaving
        the outcome 'none' with the preferred schedule booked, so such a patient is not a target.
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
        What the protection cost one booking against its unprotected counterfactual — the schedule the same objective
        would have booked with nothing held back — compared on the tests both schedules placed.

        Args:
            pred (dict): The booked prediction (its `idle_waiting_time` is read for a `stay_min` booking).
            tests (list): The booking's live tests, i.e. the schedule actually booked under the protection.
            record (dict): The booking's `capacity_protection` record with `objective` and `unprotected` (which
                           carries per-test placements in `tests`); only called for a booking the protection blocked.

        Returns:
            Optional[dict]: `{'objective', 'native', 'result_delay_h', 'extra_visit_days'}` (native = the cost in the
                            unit of the objective the booking was built with), or None when the two cannot be
                            compared — the schedules share no placed test, or a `stay_min` booking's two schedules do
                            not place the same tests (its idle wait spans the whole test set).
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
    def _test_capacity(start_hour: float, end_hour: float, result_file: str) -> dict:
        """
        `{(test_code, date): free_hours}` before any follow-up booking (total capacity minus fixed bookings), pooled
        over the test's devices, from the hospital data of the run (`<dataset>/agent_data/<hospital>_agent.json`).
        Closed days carry no capacity. Note the protection's booking limit is based on the total capacity instead
        (`HospitalEnvironment.test_capacity`).

        Args:
            start_hour (float): Opening hour of the operating window.
            end_hour (float): Closing hour of the operating window; a device-date's capacity is `end_hour - start_hour`
                              minus its fixed bookings, and a date booked for the whole window is closed.
            result_file (str): The run's result file (`<dataset>/<run>/<hospital>_agent_result.json`).

        Returns:
            dict: `{(test_code, date): Decimal free hours}` — exact sums, so a closed day is exactly zero.

        Raises:
            FileNotFoundError: If the run's hospital data is not found.
        """
        parent = os.path.dirname(os.path.dirname(os.path.abspath(result_file)))
        hospital_id = os.path.basename(result_file).split('_agent_result')[0]
        source = os.path.join(parent, 'agent_data', f'{hospital_id}_agent.json')
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
                 protection_paths: Optional[list[str]] = None):
        """
        Args:
            hospital_side_path (Optional[str], optional): Hospital-side (τ=0) results directory. Required for the
                                                          dashboard and the τ seeds; may be omitted when only the
                                                          protection runs are evaluated. Defaults to None.
            model_name (Optional[str], optional): Model label stored in the dashboard meta block. Defaults to None.
            patient_side_path (Optional[str], optional): Patient-side (τ=∞) results directory. Required for the
                                                         patient pole and per-device ΔU; the hospital-only values
                                                         are still produced without it. Defaults to None.
            interpolated_paths (Optional[list[str]], optional): List of interpolated results directories. Defaults to None.
            protection_paths (Optional[list[str]], optional): Results directories compared by the urgent capacity
                                                              protection, in any order (each run's mode and ratio come
                                                              from its `_metadata`). Defaults to None.

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

        log(f"Run metadata: window {self.start_hour}-{self.end_hour} (time_unit {self.time_unit})")


    @staticmethod
    def _read_data(path: str) -> Tuple[str, dict]:
        """
        Read the `_metadata` block the simulator stored in the result file — the operating window
        (start/end hour, time_unit) and the negotiation policy / trigger temperatures τ. Assumes the run
        was saved with metadata (re-run, or backfill older results).

        Args:
            path (str): Path to a run's results directory.

        Returns:
            Tuple[str, dict]: The result file path and the `_metadata` dict.
        
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
        
        return result_file, md
    

    @staticmethod
    def _sanity_check(*args):
        """
        Check that the runs are compatible: every `_metadata` key except `policy` (operating window, time_unit, ...)
        must have the same value across all runs.

        Args:
            *args (dict): `_metadata` dicts of the runs to compare.

        Raises:
            ValueError: If any non-policy metadata value differs between runs.
        """
        if len(args) < 2:
            return

        keys = sorted(set().union(*args) - {'policy'})
        mismatches = {k: [md.get(k) for md in args] for k in keys}
        mismatches = {k: v for k, v in mismatches.items() if any(x != v[0] for x in v[1:])}

        if mismatches:
            detail = ', '.join(f"{k}: {v}" for k, v in mismatches.items())
            raise ValueError(f"Incompatible run metadata (policy excluded) — {detail}")
        
    
    def _device_delta_u(self):
        """
        Per-device ΔU of every run against the baseline: compactness(run) - compactness(baseline) on the actual final
        bookings, scored against each device's real availability (closed dates and other patients' slots excluded —
        see `_initial_vacant_slots`). The runs are `self.all_files` (hospital side -> interpolated runs as given ->
        patient side), and each row's `u` / `du` / `n` lists follow that order. The baseline is the last run — the
        patient side, or the last interpolated run without one — so its own `du` is 0 (None when it never booked the
        device), anchoring the curve. Rows are sorted by the hospital-side ΔU descending.

        Returns:
            dict: `{'runs': [result file, ...], 'baseline': result file or None, 'rows': [...]}` — `runs` is what
                  each row's lists are aligned with (the baseline last); `rows` is empty when there is no baseline
                  or no device data.
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


    def _initial_vacant_slots(self):
        """
        `{device: {date: set(vacant segment indices)}}` — each device's vacant slots before the simulation booked
        anything, i.e. the slots a simulated booking could actually take. Read from the hospital data shared by every
        compared run (`test[dept][].devices[dev].schedule`, which lists the intervals already occupied before the
        simulation), so it is the same for every run.

        This matters: a device is closed on some dates (its schedule holds one interval covering the whole
        day) and on open dates a few hours are already taken. Treating those as vacant would invent slots
        that never existed and distort the front-loading score, so fully-closed dates are dropped and
        pre-occupied segments are removed.
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

    
    def _compactness(self, bookings, offset, universe):
        """
        Mann-Whitney front-loading score of a device's bookings: the fraction of (booked-before-free)
        slot pairs over the slots this device could actually be booked into (`universe`). 1.0 = every
        booking precedes every remaining free slot (packed early), 0.0 = fully back-loaded,
        0.5 = interleaved. None when the device has no booking. Every booking must lie on the vacant slots (an open
        date, no pre-filled overlap) — anything else is a simulator bug and raises.
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


    def interpolation_taus(self, tau_n=4):
        """
        Pick `tau_n` interior trigger-temperature (τ) candidates per preference for interpolating
        between the two poles (τ=0 hospital → τ=∞ patient).

        Uses ``ti = GPC·TCL`` recomputed τ-invariantly (via `_ti`, not the scaled stored ``ti`` field),
        so it is valid for any run regardless of `trigger_temperature`. Setting the ``'negotiation'``
        policy's ``trigger_temperature`` to a value τ (with ``negotiation_trigger_threshold == 1``) makes
        exactly the patients with ``GPC·TCL >= τ`` negotiate — equivalently one may keep
        ``trigger_temperature == 1`` and set ``negotiation_trigger_threshold`` to the same value. So per
        preference we sort the ti values ascending (visit_min / stay_min, ti > 0; the free-win ``auto``
        case has ti inf and is excluded) and take evenly-ranked quantile points — giving cutoff
        candidates that split the cohort into roughly equal bands. `auto` free wins always negotiate
        regardless of the cutoff, so they are not thresholds.

        NOTE: ti is endogenous — a run at one of these τ re-books everyone and shifts ti, so the realized
        negotiation fraction drifts from the intended quantile. Treat these as interpolation *seeds*.

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
        Compute every value the negotiation-policy dashboard needs over all runs in `self.all_files` order
        (hospital side -> interpolated runs as given -> patient side), log a per-run summary, and save it as
        `dashboard_data.json` in the dataset folder (the parent of the run folders).
        
        Args:
            save (bool, optional): Whether to save the dashboard data to `dashboard_data.json`. Defaults to False.

        Returns:
            dict: The saved payload —
                ``eligible``: ``{'runs', 'cohort'}`` — per-run eligible cohorts (`eligible_cohort` rows, GPC > 0),
                              aligned with ``runs``.
                ``devices``: per-device ΔU from `_device_delta_u` — ``{'runs', 'baseline', 'rows'}``.
                ``stats``: ``{'runs', 'cohort', 'device'}`` — per-run cohort stats (`_cohort_stats` plus
                           ``n_total``, every booking with negotiation metrics) and per-run device ΔU summaries,
                           each list aligned with its ``runs``.
                ``meta``: model name, operating window and time unit.
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
        Hospital-side outcome of the urgent capacity protection for this run, split by urgency.

        Every metric reads live bookings only (not cancelled; tests not cancelled), with the booking time taken from
        the prediction's `last_updated_time` — which a cancellation overwrites, so cancelled bookings are left out.

        - `urgent` / `normal`: regular bookings (patient-condition rows only) with the failure count and partial
          bookings (fewer tests than prescribed). `start` holds booking -> earliest test start hours and the share
          starting within the protection horizon (`horizon_days * 24` hours): the access the protection actually
          shifts. `end` holds booking -> all-results-ready hours (results of the latest test, not its end time).
        - `normal_cost`: normal bookings that had a date of their tests held back (`blocked_any`), compared with the
          unprotected schedule of the same objective on the tests both placed. `by_objective` reports the cost in each
          objective's own unit (`PROTECTION_COST_UNITS`) — the unit the scheduler minimized, so it is never negative
          by construction of a different objective; `result_delay_h` / `extra_visit_days` are the same comparison in
          fixed units over every objective, as auxiliary views. `mean_worse` / `median_worse` are over the bookings
          actually made worse, `mean_all` over every compared booking (mostly exact zeros).
        - `window`: the protection's rolling window — for each booking, the test hours placed on dates that were
          inside `[release_hours, horizon_days * 24)` hours of the booking time, i.e. the capacity the protection
          governed at that moment — and the urgent share of those hours. `lead_profile` spreads the same hours over
          lead-time bins (a test date's opening minus its booking time, so each booking is binned relative to its own
          time): 24h bins up to a week, split at `release_hours` and `horizon_days * 24`, each tagged as `released`,
          `protected` or `beyond` the window — the device-side view of where urgent bookings land.
        - `capacity`: utilization of the start-of-simulation test capacity by the final bookings, over all dates and
          over the first `ceil(horizon_days)` test dates (a fixed calendar view of the window at the start of the run).

        Args:
            result_file (str): The run's result file; only its rows are read.
            window (dict): The rolling window this run is measured over — the run's own `capacity_protection`, or, for
                           a run without protection, the one it borrows from the protected runs (see
                           `protection_report`). Only `horizon_days` and `release_hours` are read; both are required.

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
        capacity = self._test_capacity(self.start_hour, self.end_hour, result_file)
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


    def protection_report(self, save: bool = False):
        """
        Compare the urgent capacity protection across runs: one baseline without protection and runs at several
        `ratio`s, for one or more negotiation modes at once. Each folder's `_metadata` supplies its operating window,
        its mode and the policy's `capacity_protection`; runs are grouped by mode and ordered by ratio (disabled
        first). A run without protection measures the rolling window of the protected runs of its mode (or of any
        run), so baseline and protected runs are read over the same window.

        Args:
            save (bool, optional): Whether to save the protection report to `protection_report.json`. Defaults to False.

        Returns:
            list[dict]: One `{'label', 'mode', 'capacity_protection', 'policy', **protection_summary}` per run.
        """
        runs = []
        for path, md in zip(self.protection_files, self.protection_mds):
            policy = md.get('policy') or {}
            runs.append((path, policy, policy.get('mode'), policy.get('capacity_protection')))

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
        for path, policy, mode, protection in runs:
            window = protection or reference.get(mode) or reference.get(None)
            label = f"{mode} " + (f"ratio={protection['ratio']}" if protection else 'no protection')
            report.append({'label': label, 'mode': mode, 'capacity_protection': protection,
                           'policy': {k: v for k, v in policy.items() if k != 'capacity_protection'},
                           **self._protection_summary(path, window)})
        report.sort(key=lambda r: (r['mode'] or '', (r['capacity_protection'] or {}).get('ratio', 0.0)))

        # Logging the protection report, one block per run (grouped by mode, ratio ascending)
        def val(v, unit='', color='green'):
            """A logged value: colored with its unit, or `n/a` when undefined (e.g. no booking in the group)."""
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
