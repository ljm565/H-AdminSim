import os
import numpy as np
from collections import Counter

from h_adminsim.utils import log, colorstr
from h_adminsim.utils.filesys_utils import get_files, json_load
from h_adminsim.utils.image_preprocess_utils import draw_fail_donut_subplots



class Evaluator:
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
        
