import os
import sys
from argparse import ArgumentParser
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from h_adminsim.tools import FirstVisitEvaluator, FollowUpVisitEvaluator
from h_adminsim.utils import log
from h_adminsim.utils.filesys_utils import get_files, json_load, json_save_fast



def read_run_metadata(results_path):
    """
    Read the `_metadata` block the simulator stored in the result file — the operating window
    (start/end hour, time_unit) and the negotiation policy / trigger temperatures τ. Assumes the run
    was saved with metadata (re-run, or backfill older results).

    Returns:
        dict: `{start_hour, end_hour, time_unit, policy, tau_visit, tau_stay}`.
    """
    result_files = get_files(results_path, '_result.json')
    if not result_files:
        raise FileNotFoundError(f"No '*_result.json' under {results_path}")
    md = json_load(result_files[0]).get('_metadata')
    if not md:
        raise ValueError(f"No '_metadata' in {result_files[0]} — re-run the simulation or backfill it.")
    return md



def main(args):
    # first-visit (intake + first-visit scheduling) evaluations
    if any(t in args.type for t in ('task', 'human', 'department', 'rounds', 'token')):
        evaluator = FirstVisitEvaluator(args.path, human_eval='human' in args.type)

        if 'task' in args.type:
            evaluator.task_evaluation()
            log('')
        if 'token' in args.type:
            evaluator.token_cost(args.model.lower())
        if 'human' in args.type:
            evaluator.human_evaluation()
            log('')
        if 'department' in args.type:
            evaluator.department_evaluation()
            log('')
        if 'rounds' in args.type:
            evaluator.calculate_avg_rounds()
            log('')

    # follow-up (OPFU negotiation): dashboard values and/or interpolation tau seeds
    if any(t in args.type for t in ('dashboard', 'tau')):
        md = read_run_metadata(args.path)
        log(f"Run metadata: window {md['start_hour']}-{md['end_hour']} (time_unit {md['time_unit']}), "
            f"policy {md.get('policy')}, tau ({md.get('tau_visit')}, {md.get('tau_stay')})")
        fu = FollowUpVisitEvaluator(
            path=args.path,
            start_hour=md['start_hour'],
            end_hour=md['end_hour'],
            time_unit=md['time_unit'],
            model_name=(args.model or 'n/a'),   # label only; not needed for tau extraction
            counterpart_path=args.counterpart_path,
        )

        if 'dashboard' in args.type:
            save_path = args.out or os.path.join(args.path, 'dashboard_data.json')
            fu.dashboard_data(save_path=save_path, tau_n=args.tau_n)
            log('')

        if 'tau' in args.type:
            taus = fu.interpolation_taus(args.tau_n)
            log(f'--------------Interpolation tau seeds (tau_n={args.tau_n})--------------')
            for pref, vals in taus.items():
                log(f'{pref:<12} ({len(vals)}): {vals}')
            log('')

    # discrete multi-run dashboard: one results folder per τ stop, assembled from each run's _metadata
    if 'dashboard_multi' in args.type:
        save_path = args.out or 'dashboard_multi_data.json'
        FollowUpVisitEvaluator.multi_run_dashboard_data(
            run_paths=args.runs,
            model_name=(args.model or 'n/a'),
            save_path=save_path,
        )
        log('')




if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-p', '--path', type=str, required=False, default=None,
                        help='Agent test results folder. For "dashboard" this is the hospital-side (tau=0) run. '
                             'Not used by "dashboard_multi" (use --runs).')
    parser.add_argument(
        '-t', '--type',
        type=str,
        required=True,
        nargs='+',
        choices=['task', 'human', 'department', 'rounds', 'token', 'dashboard', 'tau', 'dashboard_multi'],
        help='Evaluations to run (you can specify multiple). "dashboard" computes the follow-up '
             'negotiation dashboard values; "tau" extracts per-preference interpolation tau seeds; '
             '"dashboard_multi" assembles the discrete multi-run dashboard from several run folders.'
    )
    parser.add_argument('--model', type=str, required=False, default=None,
                        help='Model name (required for "token" and "dashboard")')
    # follow-up (dashboard / tau) options
    parser.add_argument('--counterpart_path', type=str, required=False, default=None,
                        help='[dashboard] Patient-side (tau=inf) results folder; enables the patient pole and per-device dU.')
    parser.add_argument('--out', type=str, required=False, default=None,
                        help='[dashboard] Output JSON path (default: <path>/dashboard_data.json)')
    parser.add_argument('--tau_n', type=int, required=False, default=4,
                        help='[tau/dashboard] Number of interior tau seeds to extract per preference (default: 4)')
    parser.add_argument('--runs', type=str, nargs='+', required=False, default=None,
                        help='[dashboard_multi] Results folders, one per τ stop (hospital, interpolation runs, patient).')
    args = parser.parse_args()

    if 'dashboard' in args.type and not args.model:
        parser.error('--model is required when running "dashboard"')
    if 'token' in args.type and not args.model:
        parser.error('--model is required when running "token"')
    if 'dashboard_multi' in args.type and not args.runs:
        parser.error('--runs is required when running "dashboard_multi"')
    # every mode except dashboard_multi needs a single -p/--path
    if any(t != 'dashboard_multi' for t in args.type) and not args.path:
        parser.error('-p/--path is required')

    main(args)
