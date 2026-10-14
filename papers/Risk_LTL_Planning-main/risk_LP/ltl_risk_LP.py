#!/usr/bin/env python
import gurobipy as grb
import numpy as np

class Risk_LTL_LP:

    def __init__(self):
        self.state_num = None
        self.action_num = None

    def solve(self, P, c_map, initial_state, accept_states, initial_guess=None):
        self.action_num = P.shape[1]  # number of actions
        self.state_num = P.shape[0]  # number of states not in T
        S0 = initial_state  # The initial state
        gamma = 0.9 # discount factor
        th_hard = 5
        th_soft = 0.8
        model = grb.Model("risk_lp")
        y = model.addVars(self.state_num, self.action_num, vtype=grb.GRB.CONTINUOUS, name='x') # occupation measure
        z = model.addVars(1, vtype=grb.GRB.CONTINUOUS, name='z')

        # pi = model.addVars(self.state_num, vtype=grb.GRB.CONTINUOUS, name='strategy')
        # Set initial values for warm start if provided
        if initial_guess is not None:
            for s in range(self.state_num):
                for a in range(self.action_num):
                    y[s, a].start = initial_guess[s, a]

        x = []
        for s in range(self.state_num):  # compute occupation
            x += [grb.quicksum(y[s, a] for a in range(self.action_num))]

        xi = []
        for sn in range(self.state_num):  # compute incoming occupation
            # from s to s' sum_a x(s, a) P(s,a,s)'
            xi += [gamma * grb.quicksum(
                   y[s, a] * P[s][a][sn] for a in range(self.action_num) for s in range(self.state_num))]

        lhs = [x[i] - xi[i] for i in range(len(xi))]
        rhs = [0] * self.state_num
        rhs[S0] = 1
        for i in range(self.state_num):
            model.addConstr(lhs[i] == rhs[i])

        obj = 2 * grb.quicksum(y[s, a] * P[s][a][sn]
                           for a in range(self.action_num)
                           for s in range(self.state_num)
                           for sn in accept_states) - z[0]

        model.addConstr(grb.quicksum(y[s, a] * c_map[s] for s in range(self.state_num)
                                     for a in range(self.action_num)) <= th_soft + z[0])
        model.addConstr(grb.quicksum(y[s, a] * c_map[s] for s in range(self.state_num)
                                     for a in range(self.action_num)) <= th_hard)
        model.addConstr(z[0] >= 0)

        model.setObjective(obj, grb.GRB.MAXIMIZE)
        model.setParam('MIPGap', 1e-4)
        model.optimize()
        sol = model.getAttr('x', y)
        relax = model.getAttr('x', z)
        print("relax:", relax)
        return sol

    
    def extract(self, occup_dict):
        strategy = np.zeros(self.state_num)
        risk_field = np.zeros(self.state_num)
        for occ in occup_dict.items():
            state = occ[0][0]
            action = occ[0][1]
            prob = occ[1]
            if prob > risk_field[state]:
                risk_field[state] = np.log10(10 * prob + 1)
                strategy[state] = int(action)
        return strategy, risk_field/max(risk_field)


