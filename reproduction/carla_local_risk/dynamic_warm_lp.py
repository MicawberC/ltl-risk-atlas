"""HiGHS simplex LP with in-place updates for changed product CMDPs.

This is a stronger dynamic baseline than rebuilding and solving a fresh LP.
Its Python-to-HiGHS coefficient update cost is included in every query.
"""

from __future__ import annotations

import time

import highspy
import numpy as np

from experiment import GAMMA, Model
from highs_reopt import prepare as prepare_highs, query as query_highs


def prepare(model: Model) -> dict:
    return prepare_highs(model, np.empty((0, model.states)))


def update_and_query(cache: dict, old: Model, new: Model) -> dict:
    if old.states != new.states or old.actions != new.actions:
        raise ValueError("The fixed product state/action dimensions changed")
    started = time.perf_counter()
    solver = cache["solver"]
    n, actions = new.states, new.actions
    changed = (new.p - old.p).tocsr()
    changed.data[np.abs(changed.data) < 1e-12] = 0.0
    changed.eliminate_zeros()
    row_ids = np.flatnonzero(np.diff(changed.indptr))
    for rr in row_ids:
        source = rr // actions
        row = new.p.getrow(int(rr))
        values = dict(zip(map(int, row.indices), map(float, row.data)))
        for destination in changed.indices[
                changed.indptr[rr]:changed.indptr[rr + 1]]:
            destination = int(destination)
            coefficient = ((1.0 if destination == source else 0.0) -
                           GAMMA * values.get(destination, 0.0))
            status = solver.changeCoeff(destination, int(rr), coefficient)
            if status != highspy.HighsStatus.kOk:
                raise RuntimeError("HiGHS could not update product flow")
    reward_difference = np.abs(new.reward - old.reward).ravel()
    for rr in np.flatnonzero(reward_difference > 1e-12):
        status = solver.changeColCost(int(rr), -float(new.reward.ravel()[rr]))
        if status != highspy.HighsStatus.kOk:
            raise RuntimeError("HiGHS could not update reward")
    for state in np.flatnonzero(np.abs(new.cost - old.cost) > 1e-12):
        for action in range(actions):
            status = solver.changeCoeff(n, int(state * actions + action),
                                        float(new.cost[state]))
            if status != highspy.HighsStatus.kOk:
                raise RuntimeError("HiGHS could not update risk")
    update_s = time.perf_counter() - started
    result = query_highs(cache, np.empty(0), initial=new.initial)
    occupation = np.asarray(
        solver.getSolution().col_value[:cache["nx"]]).reshape(n, actions)
    result.update(occupation=occupation, update_seconds=update_s,
                  total_seconds=update_s + result["seconds"],
                  changed_transition_rows=len(row_ids))
    return result
