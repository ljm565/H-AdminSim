import os
import sys
from argparse import ArgumentParser
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from h_adminsim.tools import FirstVisitEvaluator, FollowUpVisitEvaluator
from h_adminsim.utils import log
from h_adminsim.utils.filesys_utils import get_files, json_load, json_save_fast



def load_time_metadata(results_path):
    """
    Find the synthetic-hospital data that produced `results_path` and read its operating window
    (`metadata.time`). The window (start/end hour) is randomized per hospital, so it must come from the
    hospital's own data rather than a fixed default.

    A results folder (e.g. `.../opfu_scheduling_hos_gpt-5-mini`) sits next to the run's `data/` and
    `agent_data/` folders under a shared parent (e.g. `.../tertiary_fvfu`); each result file
    `<hospital>_agent_result.json` maps to `data/<hospital>.json` (or `agent_data/<hospital>_agent.json`).

    Returns:
        tuple[float, float, float]: (start_hour, end_hour, time_unit). Across multiple hospitals the
        widest window is used (min start, max end).
    """
    parent = os.path.dirname(os.path.normpath(results_path))
    result_files = get_files(results_path, '_result.json')
    if not result_files:
        raise FileNotFoundError(f"No '*_result.json' under {results_path}")

    starts, ends, units = [], [], []
    for rf in result_files:
        hospital_id = os.path.basename(rf).split('_agent_result')[0]
        candidates = (
            os.path.join(parent, 'data', f'{hospital_id}.json'),
            os.path.join(parent, 'agent_data', f'{hospital_id}_agent.json'),
        )
        meta_time = next((json_load(c).get('metadata', {}).get('time', {})
                          for c in candidates if os.path.isfile(c)), None)
        if not meta_time:
            raise FileNotFoundError(
                f"Cannot find hospital metadata for '{hospital_id}' under {parent}/(data|agent_data)")
        starts.append(float(meta_time['start_hour']))
        ends.append(float(meta_time['end_hour']))
        units.append(float(meta_time.get('interval_hour', meta_time.get('time_unit'))))

    return min(starts), max(ends), units[0]



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
        start_hour, end_hour, time_unit = load_time_metadata(args.path)
        log(f'Loaded operating window from hospital metadata: {start_hour}-{end_hour} (time_unit {time_unit})')
        fu = FollowUpVisitEvaluator(
            path=args.path,
            start_hour=start_hour,
            end_hour=end_hour,
            time_unit=time_unit,
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




if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-p', '--path', type=str, required=True,
                        help='Agent test results folder. For "dashboard" this is the hospital-side (tau=0) run.')
    parser.add_argument(
        '-t', '--type',
        type=str,
        required=True,
        nargs='+',
        choices=['task', 'human', 'department', 'rounds', 'token', 'dashboard', 'tau'],
        help='Evaluations to run (you can specify multiple). "dashboard" computes the follow-up '
             'negotiation dashboard values; "tau" extracts per-preference interpolation tau seeds.'
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
    args = parser.parse_args()

    if 'dashboard' in args.type and not args.model:
        parser.error('--model is required when running "dashboard"')
    if 'token' in args.type and not args.model:
        parser.error('--model is required when running "token"')

    main(args)
