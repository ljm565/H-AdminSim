import os
from collections import Counter

from utils import log, colorstr
from utils.filesys_utils import get_files, json_load
from utils.image_preprocess_utils import draw_fail_donut_subplots



class Evaluator:
    def __init__(self, path):
        self.path = path
        self.files = get_files(self.path, '_result.json')


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

