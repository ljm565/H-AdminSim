import os
import numpy as np
from collections import Counter

from utils import log, colorstr
from utils.filesys_utils import get_files, json_load
from utils.image_preprocess_utils import draw_fail_donut_subplots



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
            accuracies = [sum(status) / len(status) * 100 for status in value['status']]
            avg_accuracy = sum(accuracies) / len(accuracies)
            stdv = round((sum((x - avg_accuracy) ** 2 for x in accuracies) / len(accuracies)) ** 0.5, 2) if len(accuracies) > 1 else 0.0
            log(f'{colorstr(task):<27} | average accuracy: {colorstr("green", f"{avg_accuracy:.2f}% ± {stdv}")}, files: {len(accuracies)}')
            log(f'    - Individual accuracies: {", ".join([colorstr("green", f"{acc:.2f}%") for acc in accuracies])}')
        
    
        # Micro-wise evaluation
        log('')
        log('--------------Micro-wise Evaluation--------------')
        fail_data_dict = dict()
        for task, value in aggregated_results.items():
            status = sum(value['status'], [])
            status_code = sum(value['status_code'], [])
            accuracy = sum(status) / len(status) * 100
            failed_cases = [c for s, c in zip(status, status_code) if not s]
            error_rate = (len(failed_cases) / len(status)) * 100
            log(f'{colorstr(task):<27} | accuracy: {colorstr("green", f"{accuracy:.2f}%")}, length: {sum(status)} / {len(status)}')
            log(f'{colorstr(task):<27} | Error   : {colorstr("red", f"{error_rate:.2f}%")}, length: {len(failed_cases)} / {len(status)}')

            if failed_cases:
                fail_summary = Counter(failed_cases)
                for fail_type, count in fail_summary.items():
                    percent = (count / len(failed_cases)) * 100
                    log(f'    - Fail type {colorstr("red", fail_type):<30}: {count} cases ({percent:.2f}%)')
                fail_data_dict[task] = failed_cases

        draw_fail_donut_subplots(fail_data_dict, os.path.join(self.path, 'fails.png'))

    
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

        log('--------------Supervisor Evaluation--------------')
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
                log(f'{colorstr(task):<27} | length: {total_length}, supervisor effect: {supervisor_effect_cnt} ({(supervisor_effect_cnt/total_length)*100:.2f}%)')
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
                log(f'{colorstr(task):<27} | length: {total_length}, supervisor effect: {supervisor_effect_cnt} ({(supervisor_effect_cnt/total_length)*100:.2f}%)')
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
        
