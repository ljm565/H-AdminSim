import os
import sys
from argparse import ArgumentParser
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from h_adminsim.pipeline import Evaluator



def main(args):
    evaluator = Evaluator(
        fv_path=args.fv_path,
        fu_hospital_side_path=args.fu_hospital_path,
        fu_patient_side_path=args.fu_patient_path,
        fu_interpolated_paths=args.fu_interpolated_paths,
        fu_protection_paths=args.fu_protection_paths,
        fu_agent_data_path=args.fu_agent_data_path,
        fu_custom_paths=args.fu_custom_paths,
        fu_custom_configs=args.fu_custom_configs,
        model_name=args.model,
    )
    evaluator.evaluate(args.type, save=args.save, tau_n=args.fu_tau_n)



if __name__ == '__main__':
    parser = ArgumentParser()
    # Common options
    parser.add_argument('--model', type=str, required=False, default=None,
                        help='Model name (required for "fv_token" and "fu_dashboard")')
    
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
    parser.add_argument('--fu_protection_paths', type=str, required=False, nargs='+',
                        help='Results folders compared by the urgent capacity protection, in any order — each run\'s '
                             'mode and protection ratio are read from its _metadata. Include the runs without protection '
                             'as the baselines.')
    parser.add_argument('--fu_custom_paths', type=str, required=False, nargs='+',
                        help='Results folders evaluated by "fu_custom", with the evaluations listed in each run\'s policy '
                             'config snapshot (<run>/policy/).')
    parser.add_argument('--fu_custom_configs', type=str, required=False, nargs='+', default=None,
                        help='Policy config files whose `evaluations` "fu_custom" runs on every --fu_custom_paths run '
                             'instead of each run\'s own snapshot (e.g. evaluations written after the runs).')
    parser.add_argument('--fu_agent_data_path', type=str, required=False, default=None,
                        help='Agent simulation data folder the follow-up runs used (read by "fu_protection"). '
                             'Defaults to <results folder>/../agent_data.')

    # Evaluation types
    parser.add_argument('-t', '--type', type=str, required=True, nargs='+',
        choices=Evaluator.FV_TASKS + Evaluator.FU_TASKS,
        help='Evaluations to run (you can specify multiple). First-visit (--fv_path): "fv_task" task accuracy, '
             '"fv_department" department accuracy, "fv_rounds" average intake rounds, "fv_token" API cost (--model), '
             '"fv_human" human evaluation. Follow-up: "fu_tau" interpolation tau seeds from the hospital-side run, '
             '"fu_dashboard" the negotiation dashboard over the hospital-side, interpolated and patient-side runs '
             '(--model), "fu_protection" the urgent capacity protection across --fu_protection_paths, '
             '"fu_custom" the user-defined evaluations on --fu_custom_paths.'
    )
    parser.add_argument('--save', action='store_true', required=False, help='Whether to save the evaluation results to a file (default: False)')
    args = parser.parse_args()

    # Argument sanity checks
    ## First-visit sanity checks
    if any(t in args.type for t in ('fv_task', 'fv_human', 'fv_department', 'fv_rounds', 'fv_token')) and not args.fv_path:
        parser.error('--fv_path is required for first-visit evaluations (fv_task, fv_human, fv_department, fv_rounds, fv_token)')

    if 'fv_token' in args.type and not args.model:
        parser.error('--model is required for "fv_token" evaluation')

    ## Follow-up sanity checks
    if any(t in args.type for t in ('fu_tau', 'fu_dashboard')) and not args.fu_hospital_path:
        parser.error('--fu_hospital_path is required for follow-up evaluations (fu_tau, fu_dashboard)')

    if any(t in args.type for t in ('fu_dashboard', 'fu_protection')) and not args.model:
        parser.error('--model is required when running "fu_dashboard" or "fu_protection" evaluation')
    
    if 'fu_protection' in args.type and not args.fu_protection_paths:
        parser.error('--fu_protection_paths is required for "fu_protection" evaluation')

    if 'fu_custom' in args.type and not args.fu_custom_paths:
        parser.error('--fu_custom_paths is required for "fu_custom" evaluation')

    main(args)
