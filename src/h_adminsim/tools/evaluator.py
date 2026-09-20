import os
import math
import numpy as np
from typing import Optional
from bisect import bisect_right
from collections import Counter

from h_adminsim.utils import log, colorstr
from h_adminsim.utils.filesys_utils import get_files, json_load, json_save_fast
from h_adminsim.utils.image_preprocess_utils import draw_fail_donut_subplots



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
                aggregated_results[task]['token'].extend(value['token'])

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

            n = len(sim_costs)
            avg = {k: sum(s[k] for s in sim_costs) / n for k in agent_keys + ["total"]}

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

                print(f"{avg_cost:.6f} & {avg_input:.1f} & {avg_reasoning:.1f} & {avg_output:.1f}")

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
            aggregated_results['first_visit_intake']['gt'].extend(data['first_visit_intake']['gt'])
            aggregated_results['first_visit_intake']['pred'].extend(data['first_visit_intake']['pred'])
            aggregated_results['first_visit_intake']['status'].extend(data['first_visit_intake']['status'])

        gt = aggregated_results['first_visit_intake']['gt']
        pred = aggregated_results['first_visit_intake']['pred']
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
            dialogs = data['first_visit_intake']['dialog']
            for dialog in dialogs:
                counts.append(dialog.count('Staff: ')-1)

        mean, stdv = np.mean(counts), np.std(counts)
        log('-----------------Average Rounds-----------------')
        log(f'Average Rounds: {mean:.2f} ± {stdv:.2f}')



class FollowUpVisitEvaluator(Evaluator):
    """
    Follow-up-visit (OPFU test scheduling + negotiation) evaluation.

    Beyond the shared accuracy/cost evaluation, this computes every value the negotiation-policy
    dashboard needs and can dump it as a single JSON. The policy spectrum has two poles supplied as two
    runs: `path` is the hospital-side (τ=0, always negotiate) run and `counterpart_path` is the
    patient-side (τ=∞, never negotiate) run. Per-patient metrics (PCI/TCL/G/R/U/ti) are read straight
    from each prediction's stored ``negotiation_metrics``; the per-device ΔU is recomputed from the
    actual final bookings of both runs, so it is a real (non-counterfactual) systemic measure.
    """
    TASK = 'follow_up_visit_scheduling'
    METRIC_KEYS = ('preference', 'pci', 'tcl', 'ti', 'G', 'R', 'U', 'U_pref', 'U_thr',
                   'negotiation_action', 'negotiation_outcome', 'negotiation_rounds')

    def __init__(self, 
                 path: str, 
                 start_hour: float,
                 end_hour: float,
                 time_unit: float,
                 model_name: str,
                 counterpart_path: Optional[str] = None):
        """
        Args:
            path (str): Hospital-side (τ=0) results directory.
            start_hour (float): Device operating start hour. Required — it is randomized per hospital, so
                                pass the run's own value (`metadata.time.start_hour` in the hospital data);
                                `run/evaluate.py` loads it from there automatically.
            end_hour (float): Device operating end hour. Required (see `start_hour`;
                              `metadata.time.end_hour`).
            time_unit (float): Slot granularity in hours (`metadata.time.interval_hour`). Required.
                               start_hour/end_hour/time_unit define the timetable grid used to rebuild
                               each device's bookings for the ΔU compactness metric.
            model_name (str): Model label stored in the dashboard meta block. Required.
            counterpart_path (str, optional): Patient-side (τ=∞) results directory. Required for the
                                              patient pole and per-device ΔU; the hospital-only values
                                              are still produced without it.
        """
        super().__init__(path, False)
        self.counterpart_path = counterpart_path
        self.counterpart_files = get_files(counterpart_path, '_result.json') if counterpart_path else []
        self.start_hour = None if start_hour is None else float(start_hour)
        self.end_hour = None if end_hour is None else float(end_hour)
        self.time_unit = float(time_unit)
        self.model_name = model_name
        self._used_window = None   # resolved (start, end) actually used for ΔU


    # ---- record iteration -----------------------------------------------------
    @staticmethod
    def _final_pred(pred):
        """A prediction is a dict, or a list of retries; take the last (booked) attempt."""
        if isinstance(pred, list):
            return pred[-1] if pred else None
        return pred


    def _iter_preds(self, files):
        """Yield each patient's final prediction dict for the follow-up task across `files`."""
        for file in files:
            data = json_load(file)
            value = data.get(self.TASK)
            if not value:
                continue
            for pred in value.get('pred', []):
                fp = self._final_pred(pred)
                if isinstance(fp, dict):
                    yield fp


    def _iter_metrics(self, files):
        """Yield each prediction's stored `negotiation_metrics` dict."""
        for fp in self._iter_preds(files):
            m = fp.get('negotiation_metrics')
            if isinstance(m, dict):
                yield m


    def _metric_row(self, m):
        """Flatten a `negotiation_metrics` dict to the dashboard fields (+ derived flags)."""
        row = {k: m.get(k) for k in self.METRIC_KEYS}
        row['pci_inf'] = (m.get('pci') is None)       # orjson serializes float('inf') as null
        row['auto'] = (m.get('negotiation_action') == 'auto')
        return row


    # ---- cohorts --------------------------------------------------------------
    def eligible_cohort(self, files=None, actions=('negotiate', 'auto')):
        """
        Negotiation-eligible cohort from a run: metrics whose policy action is `negotiate`/`auto`
        (i.e. PCI > 0, something to gain). Defaults to the hospital-side run.
        """
        files = self.files if files is None else files
        return [self._metric_row(m) for m in self._iter_metrics(files)
                if m.get('negotiation_action') in actions]


    def patient_eligible(self, files=None):
        """
        Counterfactually eligible cohort (PCI > 0 or inf) from the patient-side run — used to show that
        the same population lands at different PCI×TCL positions when nobody is negotiated.
        """
        files = self.counterpart_files if files is None else files
        out = []
        for m in self._iter_metrics(files):
            pci = m.get('pci')
            if pci is None or (pci and pci > 0):
                out.append({'preference': m.get('preference'), 'pci': pci,
                            'tcl': m.get('tcl'), 'ti': m.get('ti'), 'pci_inf': pci is None})
        return out


    def interpolation_taus(self, tau_n=4):
        """
        Pick `tau_n` interior trigger-temperature (τ) candidates per preference for interpolating
        between the two poles (τ=0 hospital → τ=∞ patient).

        Uses the hospital-side eligible cohort's `ti` values. Since that run's `trigger_temperature`
        is 1, the stored ``ti == PCI·TCL``; setting the ``'negotiation'`` policy's
        ``trigger_temperature`` to a value τ (with ``negotiation_trigger_threshold == 1``) makes exactly
        the patients with ``PCI·TCL >= τ`` negotiate. So per preference we sort the ti values ascending
        (visit_min / stay_min, ti > 0; the free-win ``auto`` case has ti None and is excluded) and take
        evenly-ranked quantile points — giving τ candidates that split the cohort into roughly equal
        bands. `auto` free wins always negotiate regardless of τ, so they are not thresholds.

        NOTE: ti is endogenous — a run at one of these τ re-books everyone and shifts ti, so the realized
        negotiation fraction drifts from the intended quantile. Treat these as interpolation *seeds*.

        Returns:
            dict[str, list[float]]: ``{'visit_min': [...], 'stay_min': [...]}`` with up to `tau_n` τ each.
        """
        by_pref = {'visit_min': [], 'stay_min': []}
        for m in self._iter_metrics(self.files):
            pref, ti = m.get('preference'), m.get('ti')
            if pref in by_pref and ti is not None and ti > 0:
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
        return result


    # ---- per-device ΔU (compactness on actual bookings) -----------------------
    def _device_bookings(self, files):
        """`{device: [(date, start, end), ...]}` from every test booking in a run."""
        book = {}
        for fp in self._iter_preds(files):
            for t in (fp.get('test') or []):
                dev, date, sch = t.get('device'), t.get('date'), t.get('schedule')
                if not (dev and date and sch):
                    continue
                book.setdefault(dev, []).append((date, float(sch[0]), float(sch[1])))
        return book


    def _compactness(self, bookings, offset, nseg, total, start_hour):
        """
        Mann-Whitney front-loading score of a device's bookings over the shared slot axis: the fraction
        of (booked-before-free) slot pairs. 1.0 = all bookings precede all free slots (packed early),
        0.0 = fully back-loaded, 0.5 = interleaved. None when the device has no booking.
        """
        booked = set()
        for date, s, e in bookings:
            if date not in offset:
                continue
            a = max(0, math.floor((s - start_hour) / self.time_unit))
            b = min(nseg, math.ceil((e - start_hour) / self.time_unit))
            booked.update(offset[date] + i for i in range(a, b))
        if not booked:
            return None
        free = set(range(total)) - booked
        if not free:
            return 1.0
        free_sorted = sorted(free)
        good = sum(len(free_sorted) - bisect_right(free_sorted, x) for x in booked)
        return good / (len(booked) * len(free))


    def device_delta_u(self):
        """
        Per-device ΔU = compactness(hospital) − compactness(patient) on the actual final bookings, over a
        shared date axis. Returns rows sorted by ΔU descending. Empty if no counterpart run is given.
        """
        if not self.counterpart_files:
            log(colorstr('yellow', 'device_delta_u needs `counterpart_path` (patient-side run); skipping.'))
            return []

        hos = self._device_bookings(self.files)
        pat = self._device_bookings(self.counterpart_files)

        # Resolve the operating window: explicit values win; otherwise infer from observed bookings
        # (the true window is randomized per hospital, so a wrong fixed default would bias the metric).
        starts = [s for book in (hos, pat) for v in book.values() for (_, s, _) in v]
        ends = [e for book in (hos, pat) for v in book.values() for (_, _, e) in v]
        start_hour = self.start_hour if self.start_hour is not None else (min(starts) if starts else 9.0)
        end_hour = self.end_hour if self.end_hour is not None else (max(ends) if ends else 18.0)
        self._used_window = (start_hour, end_hour)
        inferred = ' [inferred from bookings]' if (self.start_hour is None or self.end_hour is None) else ''
        log(f'device ΔU operating window: {start_hour}–{end_hour} (time_unit {self.time_unit}){inferred}')

        nseg = round((end_hour - start_hour) / self.time_unit)
        dates = sorted({d for book in (hos, pat) for v in book.values() for (d, _, _) in v})
        offset = {d: i * nseg for i, d in enumerate(dates)}
        total = len(dates) * nseg

        rows = []
        for dev in sorted(set(hos) | set(pat)):
            uh = self._compactness(hos.get(dev, []), offset, nseg, total, start_hour)
            up = self._compactness(pat.get(dev, []), offset, nseg, total, start_hour)
            if uh is None or up is None:
                continue
            rows.append({'dev': dev, 'uh': round(uh, 4), 'up': round(up, 4),
                         'du': round(uh - up, 4), 'nh': len(hos.get(dev, [])), 'np': len(pat.get(dev, []))})
        rows.sort(key=lambda r: -r['du'])
        return rows


    # ---- summary statistics ---------------------------------------------------
    @staticmethod
    def _mean(xs):
        xs = [x for x in xs if x is not None]
        return sum(xs) / len(xs) if xs else 0.0

    def _cohort_stats(self, cohort):
        """Aggregate the tile/trade-off statistics over an eligible cohort."""
        accepted = [d for d in cohort if d['negotiation_outcome'] == 'accepted']
        forced = [d for d in cohort if d['negotiation_outcome'] == 'forced']
        auto = [d for d in cohort if d['negotiation_outcome'] == 'auto']
        concluded = accepted + forced
        gv = [d['G'] for d in cohort if d['preference'] == 'visit_min']
        gs = [d['G'] for d in cohort if d['preference'] == 'stay_min']
        rounds = [d['negotiation_rounds'] or 0 for d in cohort]
        return {
            'n_eligible': len(cohort),
            'accepted': len(accepted),
            'forced': len(forced),
            'auto': len(auto),
            'concluded': len(concluded),
            'acceptance_rate': round(len(accepted) / len(concluded) * 100, 2) if concluded else None,
            'sum_R': round(sum(d['R'] for d in cohort), 2),
            'mean_R': round(self._mean([d['R'] for d in cohort]), 3),
            'mean_G_visit': round(self._mean(gv), 3),
            'mean_G_stay': round(self._mean(gs), 3),
            'mean_U': round(self._mean([d['U'] for d in cohort]), 3),
            'mean_U_pref': round(self._mean([d['U_pref'] for d in cohort]), 3),
            'mean_U_thr': round(self._mean([d['U_thr'] for d in cohort]), 3),
            'friction_total': sum(rounds),
            'friction_mean': round(self._mean(rounds), 2),
            'rounds_dist': {str(k): v for k, v in sorted(Counter(rounds).items())},   # str keys for JSON
        }


    # ---- assembled dashboard payload ------------------------------------------
    def dashboard_data(self, save_path=None, verbose=True, tau_n=4):
        """
        Compute every value the negotiation-policy dashboard needs and return it as one dict:
        ``eligible`` (hospital cohort), ``eligible_pat`` (patient pole), ``devices`` (per-device ΔU),
        ``interpolation_taus`` (`tau_n` seeds per preference), ``meta`` and ``stats``. Optionally saves
        it to `save_path` and prints a summary.
        """
        eligible = self.eligible_cohort()
        eligible_pat = self.patient_eligible() if self.counterpart_files else []
        devices = self.device_delta_u()
        stats = self._cohort_stats(eligible)
        if devices:
            dus = [d['du'] for d in devices]
            stats['device'] = {
                'n_devices': len(devices),
                'delta_u_mean': round(self._mean(dus), 4),
                'n_more_frontloaded': sum(1 for x in dus if x > 1e-6),
            }

        payload = {
            'eligible': eligible,
            'eligible_pat': eligible_pat,
            'devices': devices,
            'interpolation_taus': self.interpolation_taus(tau_n),
            'meta': {
                'hos_total_metrics': sum(1 for _ in self._iter_metrics(self.files)),
                'pat_total_metrics': sum(1 for _ in self._iter_metrics(self.counterpart_files)) if self.counterpart_files else 0,
                'model': self.model_name,
                'operating_window': list(self._used_window) if self._used_window else [self.start_hour, self.end_hour],
                'time_unit': self.time_unit,
            },
            'stats': stats,
        }

        if verbose:
            self._report(payload)
        if save_path:
            json_save_fast(save_path, payload)
            log(f'Dashboard data saved to {colorstr("green", save_path)}')
        return payload


    def _report(self, payload):
        """Pretty-print the dashboard summary."""
        s = payload['stats']
        dev = s.get('device')
        n_elig, acc, forced, auto = s['n_eligible'], s['accepted'], s['forced'], s['auto']
        rate, concluded = s['acceptance_rate'], s['concluded']
        sum_r, mean_r = s['sum_R'], s['mean_R']
        gv, gs, mu, mup, mut = s['mean_G_visit'], s['mean_G_stay'], s['mean_U'], s['mean_U_pref'], s['mean_U_thr']
        fric_t, fric_m, rdist = s['friction_total'], s['friction_mean'], s['rounds_dist']
        n_pat = len(payload['eligible_pat'])

        log('--------------Follow-up Negotiation Dashboard--------------')
        log(f'{colorstr("hospital-side (t=0)"):<30} | eligible cohort: {n_elig} (accepted {acc}, forced {forced}, auto {auto})')
        if rate is not None:
            log(f'    - acceptance rate : {colorstr("green", str(rate) + "%")} ({acc}/{concluded} concluded)')
        log(f'    - sum dR (result-hours gained) : {colorstr("green", str(sum_r))}  (mean {mean_r} h)')
        log(f'    - mean dG : visit_min {gv} days, stay_min {gs} h  (patient concession)')
        log(f'    - mean U : {mu}  (U_pref {mup} -> U_thr {mut})')
        log(f'    - Friction : {fric_t} rounds total (mean {fric_m}/target), dist {rdist}')
        log(f'{colorstr("patient-side (t=inf)"):<30} | counterfactually eligible: {n_pat} (negotiations actually run: 0)')
        if dev:
            log(f'{colorstr("device dU (real bookings)"):<30} | {dev["n_devices"]} devices, '
                f'mean dU {dev["delta_u_mean"]:+.4f}, {dev["n_more_frontloaded"]} more front-loaded under hospital')
        itaus = payload.get('interpolation_taus')
        if itaus:
            log(f'{colorstr("interpolation tau seeds"):<30} | visit_min {itaus.get("visit_min")}')
            log(f'{"":<30} | stay_min  {itaus.get("stay_min")}')
