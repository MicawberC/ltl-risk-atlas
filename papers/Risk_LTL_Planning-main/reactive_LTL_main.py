# author: Shuhao Qi
# Email: s.qi@tue.nl
# Date: January 6nd, 2025
import matplotlib.pyplot as plt
import numpy as np

from abstraction.MDP import MDP
from sim.visualizer import Visualizer
import sim.simulator as sim
from risk_LP.ltl_risk_LP import Risk_LTL_LP
from specification.prod_auto import Product
from specification.specification import LTL_Spec
from abstraction.abstraction import Abstraction
from sim.controller import MPC
from abstraction.prod_MDP import Prod_MDP



def main():

    # ---------- Environment Setting ---------------------
    params = {"dt": 0.05, "WB": 1.5}
    region_size = (20, 20)
    region_res = (5, 5)

    # ---------- Traffic Light --------------------
    traffic_light = ['g', 'r']
    state_set = range(len(traffic_light))
    # action_set = [0, 1]
    # transitions = np.array([[[1, 0],[0, 1]],
    #                        [[0, 0.0], [1, 0]]])
    action_set = [0]
    transitions = np.array([[[0.8, 0.2],[0.2, 0.8]]])
    initial_state = 0
    mdp_env = MDP(state_set, action_set, transitions, traffic_light, initial_state)

    # ---------- Specification Define --------------------
    # safe_frag = LTL_Spec("G(~c U g) & G(~v)", AP_set=['c', 'g', 'v'])
    safe_frag = LTL_Spec("G(~g -> ~c) & G(~v)", AP_set=['c', 'g', 'v'])
    scltl_frag = LTL_Spec("F(t)", AP_set=['t'])

    # ------------- Labelling -----------------------------
    def dyn_labelling(sta_labels, v_pos, v_action):
        labels = sta_labels.copy()
        g_x = int(np.floor(v_pos[0] / region_res[0]))
        g_y = int(np.floor(v_pos[1] / region_res[1]))
        v_1 = ((g_x + min(0, v_action[0])) * region_res[0], (g_x + max(1, 1 + v_action[0])) * region_res[0],
                  (g_y + min(0, v_action[1])) * region_res[1], (g_y + max(1, 1 + v_action[1])) * region_res[1])
        v_0 = (g_x * region_res[0], (g_x + 1) * region_res[0], g_y * region_res[1], (g_y + 1) * region_res[1])
        labels[v_0] = "v0"
        labels[v_1] = "v1"
        return labels

    static_label = {(10, 20, 10, 15): "c",
                    (10, 15, 15, 20): "c",
                    (15, 20, 15, 20): "t"}
    # cost_func = {"safe": 100, "co_safe": 0}
    cost_func = {"c": 5, "v0": 10, "v1": 2}

    # ---------- Visualization ---------------------
    fig, ax_1 = plt.subplots(1, 1, figsize=(10, 10))
    plt.axis('off')
    plt.axis('equal')
    vis = Visualizer(ax_1)

    # ---------- Initialization --------------------
    LP_prob = Risk_LTL_LP()
    mpc_con = MPC(params, horizon_steps=5)

    ego_state = np.array([2.5, 2.5, np.pi / 2])
    ego_pos = ego_state[:2]
    ego_prod_state = (0, 1, 1)
    abs_state_sys = [-1, -1]
    abs_state_env = 1

    oppo_car_state = np.array([12, 7.5, np.pi])
    oppo_car_pos = oppo_car_state[:2]

    label_func = dyn_labelling(static_label, oppo_car_pos, [-1, 0])
    abs_model = Abstraction(region_size, region_res, ego_pos, label_func)
    oppo_abs_state = abs_model.get_abs_state(oppo_car_pos)
    target_abs_state_sys = None
    occ_measure = None

    # -------------- Simulation Loop --------------------
    iter = 1
    while True:
        iter += 1
        if iter == 150:
            abs_state_env = 0 # change the traffic light
            abs_state_sys = [-1, -1]

        ax_1.cla()
        ax_1.set_aspect(1)
        ego_pos = ego_state[:2]
        label_func = dyn_labelling(static_label, oppo_car_pos, [-1, 0])

        # ----------- Abstraction -----------------------------
        abs_model.update_abs_state(ego_pos)
        if ((abs_state_sys != abs_model.init_abs_state) or
                (oppo_abs_state != abs_model.get_abs_state(oppo_car_pos)) or
                (occ_measure is None)): # replan only when the state changes
            abs_model = Abstraction(region_size, region_res, ego_pos, label_func)
            mdp_sys = abs_model.MDP
            mdp_prod = Prod_MDP(mdp_sys, mdp_env)
            prod_auto = Product(mdp_prod.MDP, scltl_frag.dfa, safe_frag.dfa)
            P_matrix = prod_auto.prod_transitions
            cost_map = prod_auto.gen_cost_map(cost_func)

            abs_state_sys_index, abs_state_sys = abs_model.get_abs_ind_state(ego_pos)
            oppo_abs_state = abs_model.get_abs_state(oppo_car_pos)
            state_sys_env_index = mdp_prod.get_prod_state_index((abs_state_sys_index, abs_state_env))
            ego_prod_state_index, ego_prod_state  = prod_auto.update_prod_state(state_sys_env_index, ego_prod_state)
            occ_measure = LP_prob.solve(P_matrix, cost_map, ego_prod_state_index,
                                        prod_auto.accepting_states, None)
            optimal_policy, Z = LP_prob.extract(occ_measure)
            decision_index = optimal_policy[ego_prod_state_index]
            sys_decision = abs_model.action_set[int(decision_index)]
            target_abs_state_sys = abs_state_sys + sys_decision

        target_point = target_abs_state_sys * 5 + np.array([2.5, 2.5])
        control_input = mpc_con.solve(ego_state, target_point)
        print("decision:", sys_decision)
        print("target_point", target_point)
        print("control_input:", control_input)

        if iter == 120:
            print('d')

        ego_state = sim.car_dyn(ego_state, control_input, params)
        if ((oppo_car_pos[0] > 0) and (oppo_car_pos[0] < region_size[0])
                and (oppo_car_pos[1] > 0) and (oppo_car_pos[1] < region_size[1])):
            oppo_car_state = sim.car_dyn(oppo_car_state, [3, 0], params)
            oppo_car_pos = oppo_car_state[:2]
            vis.plot_car(oppo_car_pos[0], oppo_car_pos[1], oppo_car_state[2], 0)

        plt.gca().set_aspect(1)
        vis.plot_grid(region_size, region_res, label_func, abs_state_env)
        vis.plot_car(ego_pos[0], ego_pos[1], ego_state[2], -control_input[1])
        plt.pause(0.001)


if __name__ == '__main__':
    main()