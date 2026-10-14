"""Primal/dual certificate for a changed LTL product CMDP.

An old occupation is *not* reused after P or the start state changes.  The
cached stationary policy is evaluated in the new model.  An old LP dual value
is repaired by a Bellman-residual upper bound in the new model.  A failed
certificate invokes the full LP; this is a research baseline, not a realtime
controller.
"""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np

from experiment import GAMMA, HARD, SOFT, Model, evaluate_policy, solve_lp


@dataclass
class Anchor:
    policy: np.ndarray
    dual_v: np.ndarray
    lambda_soft: float
    lambda_hard: float


def policy_from_occupation(occupation: np.ndarray) -> np.ndarray:
    totals = occupation.sum(axis=1)
    policy = np.divide(occupation, totals[:, None],
                       out=np.zeros_like(occupation),
                       where=totals[:, None] > 1e-12)
    policy[totals <= 1e-12, 0] = 1.0
    return policy


def anchor_from_lp(solution: dict) -> Anchor:
    return Anchor(policy_from_occupation(solution["occupation"]),
                  np.asarray(solution["dual_v"]),
                  max(0.0, float(solution["lambda_soft"])),
                  max(0.0, float(solution["lambda_hard"])))


def anchor_from_warm_lp(cache: dict, occupation: np.ndarray) -> Anchor:
    """Read the same scalarized dual from the retained-basis HiGHS model."""
    n = cache["n"]
    duals = np.asarray(cache["solver"].getSolution().row_dual)
    # With zero parameter directions: [flow rows, risk equality, soft, hard].
    return Anchor(policy_from_occupation(occupation), -duals[:n].copy(),
                  max(0.0, -float(duals[n + 1])),
                  max(0.0, -float(duals[n + 2])))


def residual_upper(model: Model, anchor: Anchor,
                   iterations: int = 65) -> tuple[float, float, int]:
    """Valid scalarized upper bound at the *new* model and start state.

    e(s) is the positive Bellman residual of the old dual potential under the
    new transition/reward/cost.  Value iteration from the global supersolution
    max(e)/(1-gamma) stays above its fixed point at every finite iteration.
    """
    t = anchor.lambda_soft + anchor.lambda_hard
    v = anchor.dual_v
    q = model.reward - t * model.cost[:, None]
    q += GAMMA * (model.p @ v).reshape(model.states, model.actions)
    residual = np.maximum(0.0, np.max(q - v[:, None], axis=1))
    largest = float(residual.max())
    correction = np.full(model.states, largest / (1.0 - GAMMA))
    for i in range(iterations):
        following = residual + GAMMA * (
            model.p @ correction).reshape(model.states, model.actions).max(axis=1)
        correction = following
    upper = (float(v[model.initial] + correction[model.initial])
             + SOFT * anchor.lambda_soft + HARD * anchor.lambda_hard + 1e-6)
    return upper, largest, iterations


def try_certify(model: Model, anchor: Anchor | None,
                relative_tolerance: float = .01) -> dict:
    started = time.perf_counter()
    if anchor is not None:
        evaluation = evaluate_policy(model, anchor.policy)
        if evaluation["risk"] <= HARD + 1e-7:
            upper, residual, iterations = residual_upper(model, anchor)
            lower = float(evaluation["objective"])
            gap = upper - lower
            tolerance = max(1e-4, relative_tolerance * abs(lower)) + 1e-7
            attempt = {"objective": lower, "risk": evaluation["risk"],
                       "upper": upper, "gap": gap,
                       "max_residual": residual,
                       "residual_iterations": iterations,
                       "flow_residual": evaluation["flow_residual"]}
            if -1e-5 <= gap <= tolerance:
                attempt.update(certified=True, lp_fallback=False,
                               policy=anchor.policy,
                               seconds=time.perf_counter() - started)
                return attempt
            attempt.update(certified=False,
                           seconds=time.perf_counter() - started)
            return attempt
    return {"certified": False, "reason": "no_feasible_cached_policy",
            "seconds": time.perf_counter() - started}


def query(model: Model, anchor: Anchor | None,
          relative_tolerance: float = .01) -> tuple[dict, Anchor]:
    started = time.perf_counter()
    attempt = try_certify(model, anchor, relative_tolerance)
    if attempt["certified"]:
        return attempt, anchor
    solution = solve_lp(model)
    next_anchor = anchor_from_lp(solution)
    answer = {"certified": True, "lp_fallback": True,
              "objective": solution["objective"],
              "risk": float(np.sum(solution["occupation"] *
                                   model.cost[:, None])),
              "upper": solution["objective"] + 1e-5,
              "gap": 1e-5, "policy": next_anchor.policy,
              "attempt": attempt, "seconds": time.perf_counter() - started}
    return answer, next_anchor
