import os
import sys
from argparse import ArgumentParser
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from h_adminsim.tools import FirstVisitEvaluator, FollowUpVisitEvaluator
from h_adminsim.utils import colorstr, log
from h_adminsim.utils.filesys_utils import get_files, json_load



def main(args):
    # First-visit (intake + first-visit scheduling) evaluations
    if any(t in args.type for t in ('fv_task', 'fv_human', 'fv_department', 'fv_rounds', 'fv_token')):
        log('################### First-visit evaluation ###################')
        evaluator = FirstVisitEvaluator(args.fv_path, human_eval='fv_human' in args.type)
        
        if 'fv_task' in args.type:
            evaluator.task_evaluation()
            log('')
        
        if 'fv_token' in args.type:
            evaluator.token_cost(args.model.lower())
        
        if 'fv_human' in args.type:
            evaluator.human_evaluation()
            log('')
        
        if 'fv_department' in args.type:
            evaluator.department_evaluation()
            log('')
        
        if 'fv_rounds' in args.type:
            evaluator.calculate_avg_rounds()
            log('')

    # Follow-up (negotiation): dashboard values and/or interpolation tau seeds
    if any(t in args.type for t in ('fu_dashboard', 'fu_tau')):
        log('################### Follow-up evaluation ###################')
        evaluator = FollowUpVisitEvaluator(
            hospital_side_path=args.fu_hospital_path,
            model_name=args.model,
            patient_side_path=args.fu_patient_path,
            interpolated_paths=args.fu_interpolated_paths,
        )

        if 'fu_tau' in args.type:
            taus = evaluator.interpolation_taus(args.fu_tau_n)
            log('')

        if 'fu_dashboard' in args.type:
            evaluator.dashboard_data(save_path=save_path, tau_n=args.fu_tau_n)
            log('')

    # # discrete multi-run dashboard: one results folder per τ stop, assembled from each run's _metadata
    # if 'dashboard_multi' in args.type:
    #     save_path = args.out or 'dashboard_multi_data.json'
    #     FollowUpVisitEvaluator.multi_run_dashboard_data(
    #         run_paths=args.runs,
    #         model_name=(args.model or 'n/a'),
    #         save_path=save_path,
    #     )
    #     log('')

    # # urgent capacity protection: one results folder per protection ratio (a run without protection as the baseline)
    # if 'protection' in args.type:
    #     FollowUpVisitEvaluator.protection_report(
    #         run_paths=args.runs,
    #         save_path=args.out or 'protection_report.json',
    #         early_days=args.early_days,
    #     )
    #     log('')



if __name__ == '__main__':
    parser = ArgumentParser()
    # Common options
    parser.add_argument('--model', type=str, required=False, default=None,
                        help='Model name (required for "token" and "dashboard")')
    
    # First-visit options
    parser.add_argument('--fv_path', type=str, required=False, default=None,
                        help='Agent test results folder. Required for first-visit evaluations (fv_task, fv_human, fv_department, fv_rounds, fv_token).')
    
    # Follow-up options
    parser.add_argument('--fu_hospital_path', type=str, required=False, default=None,
                        help='Hospital-side (tau=0) results folder; enables the hospital pole and per-device dU.')
    parser.add_argument('--fu_patient_path', type=str, required=False, default=None,
                        help='Patient-side (tau=inf) results folder; enables the patient pole and per-device dU.')
    parser.add_argument('--fu_interpolated_paths', type=str, required=False, nargs='+',
                        help='Interpolated (threshold-mode) results folders for follow-up evaluation. ORDER MATTERS: '
                             'the dashboard is built in the order hospital-side -> these folders exactly as given -> patient-side, '
                             'so list them in the interpolation order you want shown.')
    parser.add_argument('--fu_tau_n', type=int, required=False, default=4,
                        help='Number of interior tau seeds to extract per preference (default: 4)')
    
    # Evaluation types
    parser.add_argument('-t', '--type', type=str, required=True, nargs='+',
        choices=[
            'fv_task', 'fv_department', 'fv_rounds', 'fv_token', 'fv_human',
            'fu_tau', 'fu_dashboard', 'dashboard_multi', 'protection'
        ],
        help='Evaluations to run (you can specify multiple). "dashboard" computes the follow-up '
             'negotiation dashboard values; "tau" extracts per-preference interpolation tau seeds; '
             '"dashboard_multi" assembles the discrete multi-run dashboard from several run folders; '
             '"protection" compares the urgent capacity protection across run folders (--runs).'
    )
    
    # # follow-up (dashboard / tau) options
    # parser.add_argument('--out', type=str, required=False, default=None,
    #                     help='[dashboard] Output JSON path (default: <path>/dashboard_data.json)')
    
    # parser.add_argument('--runs', type=str, nargs='+', required=False, default=None,
    #                     help='[dashboard_multi] Results folders, one per τ stop (hospital, interpolation runs, patient). '
    #                          '[protection] Results folders, one per protection ratio.')
    # parser.add_argument('--early_days', type=int, required=False, default=3,
    #                     help='[protection] Number of first test dates counted as early capacity (default: 3)')
    args = parser.parse_args()

    # Argument sanity checks
    ## First-visit sanity checks
    if any(t in args.type for t in ('fv_task', 'fv_human', 'fv_department', 'fv_rounds', 'fv_token')) and not args.fv_path:
        parser.error('--fv_path is required for first-visit evaluations (fv_task, fv_human, fv_department, fv_rounds, fv_token)')

        if 'fv_token' in args.type and not args.model:
            parser.error('--model is required for "fv_token" evaluation')

    ## Follow-up sanity checks
    if 'fu_tau' in args.type and not args.fu_hospital_path:
        parser.error('--fu_hospital_path is required for "fu_tau" evaluation')

    if 'fu_dashboard' in args.type:
        if not args.model:
            parser.error('--model is required when running "dashboard"')

    # for multi_run in ('dashboard_multi', 'protection'):
    #     if multi_run in args.type and not args.runs:
    #         parser.error(f'--runs is required when running "{multi_run}"')
    
    # # every mode except the multi-run ones needs a single -p/--path
    # if any(t not in ('dashboard_multi', 'protection') for t in args.type) and not args.fv_path:
    #     parser.error('-p/--fv_path is required')

    main(args)
