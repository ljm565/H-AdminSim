import os
import sys
from argparse import ArgumentParser
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from h_adminsim.tools import Evaluator
from h_adminsim.utils import log



def main(args):
    evaluator = Evaluator(args.path, human_eval='human' in args.type)
    
    if 'task' in args.type:
        evaluator.task_evaluation()
        log('')
    
    if 'ipi' in args.type:
        evaluator.ipi_evaluation()
        log('')

    if 'feedback' in args.type:
        evaluator.supervisor_evaluation()
        log('')

    if 'human' in args.type:
        evaluator.human_evaluation()
        log('')

    if 'department' in args.type:
        evaluator.department_evaluation()
        log('')

    if 'rounds' in args.type:
        evaluator.calculate_avg_rounds()
        log('')




if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-p', '--path', type=str, required=True, help='Agent test results folder directory')
    parser.add_argument(
        '-t', '--type', 
        type=str, 
        required=True, 
        nargs='+', 
        choices=['task', 'feedback', 'human', 'department', 'rounds', 'ipi'], 
        help='Task types you want to evaluate (you can specify multiple)'
    )
    args = parser.parse_args()

    main(args)