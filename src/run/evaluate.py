import os
import sys
from collections import Counter
from argparse import ArgumentParser
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from utils import log, colorstr
from utils.filesys_utils import get_files, json_load
from utils.image_preprocess_utils import draw_fail_donut_subplots




def main(args):
    aggregated_results = dict()
    files = get_files(args.path, 'json')

    # Aggregate the results
    for file in files:
        data = json_load(file)

        for task, value in data.items():
            if not task in aggregated_results:
                aggregated_results[task] = {'status': [], 'status_code': []}
            
            aggregated_results[task]['status'].append(value['status'])
            aggregated_results[task]['status_code'].append(value['status_code'])


    # Micro-wise evaluation
    fail_data_dict = dict()
    for task, value in aggregated_results.items():
        status = sum(value['status'], [])
        status_code = sum(value['status_code'], [])
        accuracy = sum(status) / len(status) * 100
        failed_cases = [c for s, c in zip(status, status_code) if not s]
        log(f'{colorstr(task):<27} | accuracy: {accuracy:.2f}%, length: {len(status)}')

        if failed_cases:
            fail_summary = Counter(failed_cases)
            for fail_type, count in fail_summary.items():
                percent = (count / len(failed_cases)) * 100
                log(f'    - Fail type {colorstr("red", fail_type):<30}: {count} cases ({percent:.2f}%)')
            fail_data_dict[task] = failed_cases

    draw_fail_donut_subplots(fail_data_dict, os.path.join(args.path, 'fails.png'))




if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-p', '--path', type=str, required=True, help='Agent test results folder directory')
    args = parser.parse_args()

    main(args)