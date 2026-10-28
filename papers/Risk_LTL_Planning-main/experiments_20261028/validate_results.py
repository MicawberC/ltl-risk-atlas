"""Untimed independent checks of policies actually returned during CARLA drives.

This pass evaluates the complete saved policy, rather than reconstructing a
possibly different policy from its forward probability. Nonreturning planner
failures receive fresh-LP diagnostics and never count as certified returns.
"""
from __future__ import annotations

import os
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '1'
os.environ.update(LTL_RISK_GAMMA='0.8', LTL_RISK_SOFT='1.0', LTL_RISK_HARD='2.0')

import argparse
from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path
import sys
import traceback
import zipfile

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / 'reproduction/adaptive_risk_ltl'))
import numpy as np
from experiment import GAMMA, HARD, SOFT, build_grid, evaluate_policy
from highs_reopt import prepare as prepare_lp, query as query_lp

assert (GAMMA, SOFT, HARD) == (.8, 1., 2.), 'Validation parameters differ from driving'
TOLERANCES = {'policy_minimum': 1e-12, 'probability_row_sum': 1e-10,
              'policy_value': 1e-5, 'risk_budget': 1e-7, 'LP_bracket': 1e-5,
              'flow_residual': 1e-7, 'gap': 1e-7, 'forward_probability': 1e-12}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve_recorded_path(value: str, source: Path) -> Path:
    """Support absolute paths and source-, experiment-, or repo-relative paths."""
    path = Path(value)
    if path.is_absolute():
        if not path.exists():
            raise FileNotFoundError(path)
        return path
    for root in (source.parent, HERE, REPO):
        candidate = root / path
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f'Cannot locate recorded path {value!r} from {source}')


def build_reference(reference: dict):
    grid = int(reference['grid_n'])
    if grid != 20:
        raise ValueError('This protocol requires the unchanged 20x20 driving product')
    base = build_grid(grid, 'carla_crossing',
                      labels_override=np.asarray(reference['labels'], dtype=np.int8),
                      lane_only=True)
    directions = np.stack([
        np.where(np.all(base.xy == cell, axis=1), base.cost, 0.)
        for cell in reference['risk_cells']
    ])
    if len(directions) != 5:
        raise ValueError('Expected exactly five risk-cost directions')
    return base, directions


def fresh_lp_diagnostic(base, directions, theta, initial: int) -> dict:
    """A solver status is a numerical diagnosis, not an exact infeasibility proof."""
    try:
        lp = prepare_lp(replace(base, initial=initial), directions,
                        feasibility_tolerance=1e-9, solver_threads=1)
        result = query_lp(lp, theta, initial=initial)
        return {'status': 'Optimal', 'objective': float(result['objective']),
                'risk': float(result['risk']), 'basis_valid': bool(result['basis_valid'])}
    except Exception as exc:
        return {'status': str(exc), 'exception_type': type(exc).__name__}


def validate_trial(path: Path, *, max_rows: int | None = None) -> dict:
    trial = json.loads(path.read_text('utf8'))
    reference_path = resolve_recorded_path(trial['reference_path'], path)
    if trial.get('reference_sha256') != sha256(reference_path):
        raise ValueError('Recorded road-reference SHA256 mismatch')
    reference = json.loads(reference_path.read_text('utf8'))
    base, directions = build_reference(reference)
    policy_path = resolve_recorded_path(trial['policy_archive'], path)
    parameters = trial.get('parameters', {})
    if (parameters.get('gamma'), parameters.get('soft'), parameters.get('hard')) != (.8, 1., 2.):
        raise ValueError('Recorded parameters do not match gamma=.8, soft=1, hard=2')
    measured_rows = trial['rows']
    inspected = measured_rows if max_rows is None else measured_rows[:max_rows]
    violations = []
    rows = []
    lp = None
    previous_go = False
    previous_tick = -1
    probability_minimum = 1.
    row_sum_error_maximum = 0.
    maximum_objective_error = 0.
    maximum_risk_error = 0.
    maximum_flow_residual = 0.
    action_checks = 0
    with zipfile.ZipFile(policy_path) as policies:
        for index, measured in enumerate(inspected):
            tick = int(measured['tick'])
            item = {'index': index, 'tick': tick, 'returned': 'policy_key' in measured}
            local_violations = []

            def check(condition: bool, message: str):
                if not condition:
                    local_violations.append(message)

            try:
                check(tick > previous_tick, 'Non-increasing simulation tick')
                previous_tick = tick
                theta = np.asarray(measured['theta'], dtype=float)
                initial = int(measured['initial'])
                check(theta.shape == (5,) and bool(np.isfinite(theta).all()), 'Invalid risk parameters')
                check(0 <= initial < base.states, 'Initial state outside product')
                check(np.isfinite(measured['plan_s']) and measured['plan_s'] >= 0,
                      'Planning time missing, negative, or non-finite')
                check(np.isfinite(measured['product_s']) and measured['product_s'] >= 0,
                      'Product-update time missing, negative, or non-finite')
                x, y = map(int, measured['ego_xy'])
                check(int(base.xy[initial, 0]) == x and int(base.xy[initial, 1]) == y,
                      'Initial product state and measured ego cell disagree')
                current = replace(base, initial=initial, cost=base.cost + theta @ directions)
                check(bool(np.min(current.cost) >= -1e-12), 'Negative risk costs')
                if not item['returned']:
                    check(bool(measured.get('planner_failure')), 'Nonreturning row has no failure reason')
                    item['planner_failure'] = measured.get('planner_failure')
                    item['fresh_lp'] = fresh_lp_diagnostic(base, directions, theta, initial)
                    item['violations'] = local_violations
                    item['passed'] = not local_violations
                    rows.append(item)
                    violations.extend({'tick': tick, 'message': msg} for msg in local_violations)
                    continue

                check(not measured.get('planner_failure'), 'Returned policy also marked as planner failure')
                policy = np.load(io.BytesIO(policies.read(measured['policy_key'])), allow_pickle=False)
                check(policy.dtype == np.dtype('float64'), 'Policy is not stored as float64')
                check(policy.shape == (base.states, base.actions), 'Policy dimensions disagree with product')
                check(bool(np.isfinite(policy).all()), 'Policy contains non-finite probabilities')
                policy_hash = hashlib.sha256(np.ascontiguousarray(policy).tobytes(order='C')).hexdigest()
                check(policy_hash == measured['policy_sha256'], 'Saved full-policy SHA256 mismatch')
                minimum = float(np.min(policy))
                row_error = float(np.max(np.abs(policy.sum(axis=1) - 1.)))
                probability_minimum = min(probability_minimum, minimum)
                row_sum_error_maximum = max(row_sum_error_maximum, row_error)
                check(minimum >= -TOLERANCES['policy_minimum'], 'Negative action probability')
                check(row_error <= TOLERANCES['probability_row_sum'], 'Action probabilities do not sum to one')
                forward = float(policy[initial, 5])
                check(abs(forward - float(measured['forward_probability'])) <= TOLERANCES['forward_probability'],
                      'Actuation probability differs from saved full policy')

                due = bool(measured['action_due'])
                go = bool(measured['go'])
                random_value = measured.get('action_random')
                interval = trial.get('action_every_ticks', trial.get('action_every'))
                if interval is not None:
                    check(due == (tick % int(interval) == 0), 'Action sampling schedule differs from declared interval')
                if due:
                    if random_value is not None:
                        check(np.isfinite(random_value) and 0 <= random_value < 1,
                              'Sampled uniform variate outside [0,1)')
                        check(go == (float(random_value) < forward), 'Applied action disagrees with Bernoulli policy sample')
                    elif trial.get('sample_policy') is False:
                        check(go == (forward >= .5), 'Thresholded action disagrees with policy')
                    else:
                        check(False, 'Action update did not record its sampled uniform variate')
                else:
                    check(random_value is None, 'Held action unexpectedly records a new sample')
                    check(go == previous_go, 'Held action changes without action_due')
                previous_go = go
                action_checks += 1

                evaluated = evaluate_policy(current, policy)
                objective = float(measured['objective'])
                risk = float(measured['risk'])
                upper = float(measured['upper'])
                gap = float(measured['gap'])
                check(bool(np.isfinite([objective, risk, upper, gap]).all()), 'Non-finite reported objective/risk/certificate')
                objective_error = abs(float(evaluated['objective']) - objective)
                risk_error = abs(float(evaluated['risk']) - risk)
                flow_residual = float(evaluated['flow_residual'])
                maximum_objective_error = max(maximum_objective_error, objective_error)
                maximum_risk_error = max(maximum_risk_error, risk_error)
                maximum_flow_residual = max(maximum_flow_residual, flow_residual)
                check(objective_error <= TOLERANCES['policy_value'], 'Reported and independently evaluated objectives differ')
                check(risk_error <= TOLERANCES['policy_value'], 'Reported and independently evaluated risks differ')
                check(float(evaluated['risk']) <= HARD + TOLERANCES['risk_budget'], 'Independently evaluated hard risk budget violated')
                check(risk <= HARD + TOLERANCES['risk_budget'], 'Reported hard risk budget violated')
                check(flow_residual <= TOLERANCES['flow_residual'], 'Independent occupation flow residual too large')
                tolerance = max(1e-4, .01 * abs(objective))
                check(-1e-6 <= upper - objective <= tolerance + TOLERANCES['gap'], 'Global upper/lower gap fails the original return gate')
                check(-1e-6 <= gap <= tolerance + TOLERANCES['gap'], 'Recorded gap fails the original return gate')
                check(abs(gap - (upper - objective)) <= TOLERANCES['policy_value'], 'Recorded gap and upper-minus-objective disagree')

                if lp is None:
                    lp = prepare_lp(replace(base, initial=initial), directions,
                                    feasibility_tolerance=1e-9, solver_threads=1)
                exact = query_lp(lp, theta, initial=initial)
                check(bool(exact['basis_valid']), 'Independent LP has no valid basis')
                check(objective <= exact['objective'] + TOLERANCES['LP_bracket'], 'Returned value exceeds independent LP optimum')
                check(exact['objective'] <= upper + TOLERANCES['LP_bracket'], 'Independent LP optimum exceeds claimed global upper bound')
                item.update(policy_sha256=policy_hash, evaluated_objective=float(evaluated['objective']),
                            evaluated_risk=float(evaluated['risk']), objective_error=objective_error,
                            risk_error=risk_error, flow_residual=flow_residual,
                            independent_lp_objective=float(exact['objective']),
                            reported_objective=objective, upper=upper, gap=gap,
                            allowed_gap=tolerance, policy_minimum=minimum,
                            maximum_probability_row_sum_error=row_error,
                            actual_forward_probability=forward, action_check=True)
            except Exception as exc:
                local_violations.append(f'{type(exc).__name__}: {exc}')
                item['traceback'] = traceback.format_exc()
            item['violations'] = local_violations
            item['passed'] = not local_violations
            rows.append(item)
            violations.extend({'tick': tick, 'message': msg} for msg in local_violations)
    full_validation = len(inspected) == len(measured_rows)
    if not trial.get('trial_complete'):
        violations.append({'tick': None, 'message': 'Trial output not marked complete'})
    returned_rows = [row for row in rows if row['returned']]
    summary = {'source': str(path), 'case': trial.get('case'), 'method': trial.get('method'),
               'seed': trial.get('seed'), 'repetition': trial.get('repetition'),
               'drive_goal': bool(trial.get('goal')), 'drive_collision': bool(trial.get('collision')),
               'total_recorded_queries': len(measured_rows), 'checked_queries': len(rows),
               'checked_returned_policies': len(returned_rows),
               'nonreturning_failed_queries': sum(not row['returned'] for row in rows),
               'action_checks': action_checks,
               'minimum_probability': probability_minimum if returned_rows else None,
               'maximum_probability_row_sum_error': row_sum_error_maximum,
               'maximum_objective_reconstruction_error': maximum_objective_error,
               'maximum_risk_reconstruction_error': maximum_risk_error,
               'maximum_flow_residual': maximum_flow_residual,
               'full_validation': full_validation, 'violation_count': len(violations),
               'passed': not violations,
               'source_sha256': sha256(path), 'policy_archive_sha256': sha256(policy_path),
               'reference_sha256': sha256(reference_path),
               'validator_sha256': sha256(Path(__file__))}
    return {'scope': 'Untimed verification of complete actually returned policies; planner failures are separate fresh-LP diagnoses',
            'parameters': {'gamma': GAMMA, 'soft': SOFT, 'hard': HARD},
            'tolerances': TOLERANCES, 'summary': summary, 'violations': violations, 'rows': rows}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', type=Path, default=HERE / 'results')
    parser.add_argument('--out', type=Path, default=HERE / 'validation')
    parser.add_argument('--limit', type=int, help='Inspect at most this many trial files')
    parser.add_argument('--max-rows', type=int, help='Development smoke check: inspect only this many rows per trial')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    summaries = []
    trial_paths = []
    for path in sorted(args.results_dir.glob('*.json')):
        try:
            trial = json.loads(path.read_text('utf8'))
        except Exception:
            continue
        if 'rows' in trial and 'method' in trial and 'reference_path' in trial:
            trial_paths.append(path)
    if args.limit is not None:
        trial_paths = trial_paths[:args.limit]
    for index, path in enumerate(trial_paths):
        print(f'VALIDATE {index + 1}/{len(trial_paths)} {path.name}', flush=True)
        try:
            result = validate_trial(path, max_rows=args.max_rows)
        except Exception as exc:
            result = {'scope': 'Validation failed before all recorded policies could be checked',
                      'summary': {'source': str(path), 'passed': False, 'full_validation': False,
                                  'violation_count': 1, 'error': f'{type(exc).__name__}: {exc}'},
                      'traceback': traceback.format_exc()}
        (args.out / path.name).write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf8')
        summaries.append(result['summary'])
    audit = {'scope': 'Model-certificate and logging integrity checks, not a driving-success or physical safety guarantee',
             'parameters': {'gamma': GAMMA, 'soft': SOFT, 'hard': HARD}, 'tolerances': TOLERANCES,
             'trial_files': len(summaries),
             'checked_queries': sum(row.get('checked_queries', 0) for row in summaries),
             'checked_returned_policies': sum(row.get('checked_returned_policies', 0) for row in summaries),
             'nonreturning_failed_queries': sum(row.get('nonreturning_failed_queries', 0) for row in summaries),
             'violation_count': sum(row.get('violation_count', 0) for row in summaries),
             'all_integrity_checks_passed': bool(summaries) and all(row['passed'] for row in summaries),
             'all_trials_fully_checked': bool(summaries) and all(row['full_validation'] for row in summaries),
             'limited_run': args.limit is not None or args.max_rows is not None,
             'trials': summaries}
    (args.out / 'audit.json').write_text(json.dumps(audit, indent=2, allow_nan=False) + '\n', encoding='utf8')
    print(json.dumps({key: value for key, value in audit.items() if key != 'trials'}, indent=2), flush=True)
    return 0 if audit['all_integrity_checks_passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
