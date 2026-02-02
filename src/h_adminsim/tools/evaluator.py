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
        
        try:
            self.dialog_files = get_files(self.path, '_dialog.json')
        except:
            pass


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
            accuracies = [sum(x if isinstance(x, bool) else sum(x) for x in status) / sum(1 if isinstance(x, bool) else len(x) for x in status) * 100 for status in value['status']]
            avg_accuracy = sum(accuracies) / len(accuracies)
            stdv = round((sum((x - avg_accuracy) ** 2 for x in accuracies) / len(accuracies)) ** 0.5, 2) if len(accuracies) > 1 else 0.0
            log(f'{colorstr(task):<27} | average accuracy: {colorstr("green", f"{avg_accuracy:.2f}% ± {stdv}")}, files: {len(accuracies)}')
            log(f'    - Individual accuracies: {", ".join([colorstr("green", f"{acc:.2f}%") for acc in accuracies])}')
        
    
        # Micro-wise evaluation
        log('')
        log('--------------Micro-wise Evaluation--------------')
        fail_data_dict = dict()
        for task, value in aggregated_results.items():
            status = [x for y in sum(value['status'], []) for x in (y if isinstance(y, list) or isinstance(y, tuple) else [y])]
            status_code = [x for y in sum(value['status_code'], []) for x in (y if isinstance(y, list) or isinstance(y, tuple) else [y])]
            accuracy = sum(status) / len(status) * 100
            failed_cases = [c for s, c in zip(status, status_code) if not s and 'unexpected' not in c]
            error_rate = (len(failed_cases) / len(status)) * 100
            log(f'{colorstr(task):<27} | accuracy: {colorstr("green", f"{accuracy:.2f}%")}, length: {sum(status)} / {len(status)}')
            log(f'{colorstr(task):<27} | Error   : {colorstr("red", f"{error_rate:.2f}%")}, length: {len(failed_cases)} / {len(status)}')

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
                    percent = (count / len(failed_cases)) * 100
                    reschedule_n = reschedule_fail_summary[fail_type] if fail_type in reschedule_fail_summary else 0
                    if reschedule_n:
                        log(f'    - Fail type {colorstr("red", fail_type):<30}: {count} (reschedule: {reschedule_n}) cases ({percent:.2f}%)')
                    else:
                        log(f'    - Fail type {colorstr("red", fail_type):<30}: {count} cases ({percent:.2f}%)')
                fail_data_dict[task] = failed_cases

        draw_fail_donut_subplots(fail_data_dict, os.path.join(self.path, 'fails.png'))


    def ipi_evaluation(self):
        """
        Micro-wise IPI performance evaluation on the aggregated results.
        """
        aggregated_results = dict()
        for file in self.files:
            data = json_load(file)

            if not 'intake' in aggregated_results:
                aggregated_results['intake'] = {'status': [], 'status_code': []}

            aggregated_results['intake']['status'].append(data['intake']['status'])
            aggregated_results['intake']['status_code'].append(data['intake']['status_code'])
    
        # Micro-wise evaluation
        log('')
        log('------------------IPI Evaluation-----------------')
        status = sum(aggregated_results['intake']['status'], [])
        status_code = sum(aggregated_results['intake']['status_code'], [])
        failed_cases = [c for s, c in zip(status, status_code) if not s]

        if failed_cases:
            if_err_count, ipi_err_count = 0, 0
            fail_summary = Counter(failed_cases)
            for fail_type, count in fail_summary.items():
                if fail_type in ['incorrect department and patient information', 'incorrect patient information']:
                    ipi_err_count += count
                elif fail_type in ['incorrect format']:
                    if_err_count += count
            
            if_percent = (if_err_count / len(status)) * 100
            ipi_percent = (ipi_err_count / len(status)) * 100
            log(f'    - Fail type {colorstr("red", "incorrect format"):<38}: {if_err_count} / {len(status)} ({if_percent:.2f}%)')
            log(f'    - Fail type {colorstr("red", "incorrect patient information"):<38}: {ipi_err_count} / {len(status)} ({ipi_percent:.2f}%)')


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
            
            if task == 'intake':
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

            elif task == 'schedule':
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
        aggregated_results = {'intake': {'gt': [], 'pred': [], 'status': []}}
        
        for file in self.files:
            data = json_load(file)
            aggregated_results['intake']['gt'].extend(data['intake']['gt'])
            aggregated_results['intake']['pred'].extend(data['intake']['pred'])
            aggregated_results['intake']['status'].extend(data['intake']['status'])

        gt = aggregated_results['intake']['gt']
        pred = aggregated_results['intake']['pred']
        status = aggregated_results['intake']['status']
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
        for file in self.dialog_files:
            data = json_load(file)
            dialogs = list(data.values())
            for dialog in dialogs:
                counts.append(dialog.count('Staff: ')-1)

        mean, stdv = np.mean(counts), np.std(counts)
        log('-----------------Average Rounds-----------------')
        log(f'Average Rounds: {mean:.2f} ± {stdv:.2f}')        
        
