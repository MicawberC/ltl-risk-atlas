import matplotlib.pyplot as plt
from matplotlib import cm
# from mayavi import mlab
import numpy as np
import sim.perception as pept
from ltl_risk_LP import Risk_LTL_LP
from abstraction.abstraction import Abstraction_2
import mpl_toolkits.mplot3d.art3d as art3d


def meshgrid(range, res):
    xbl = 0
    xbu = range[0]
    ybl = 0
    ybu = range[1]
    grid_x = np.arange(xbl, xbu, res[0])
    grid_y = np.arange(ybl, ybu, res[1])
    X, Y = np.meshgrid(grid_x, grid_y)
    return X, Y

def main():
    # ---------- Environment Setting ---------------------
    map_size = [64, 64]
    # square_obs_list = [[-32, 10, 32, 15], [0, 40, 32, 15]]
    square_obs_list = [[-32, 24, 32, 15]]
    # square_obs_list = [[-32, 24, 0, 0]]
    target_region = [12, 56, 8, 4]
    # square_obs_list = [[-32, 20, 0, 0]]
    params = {"dt": 0.1, "WB": 3.5}
    car_state = np.array([0, 0, np.pi/2])

    LP_prob = Risk_LTL_LP()
    pcpt_res = (4, 4)
    X, Y =  meshgrid(map_size, pcpt_res)
    car_pos = car_state[:2]

    pcpt_dic = pept.gen_pcpt_dic(map_size, square_obs_list, cost=5)
    cost_map = pept.gen_cost_map(pcpt_dic, car_pos, map_size, pcpt_res)
    cost_map =  cost_map.flatten(order='F')

    abs_model = Abstraction_2(map_size, pcpt_res)
    P_matrix = abs_model.linear()
    init_state = int(map_size[0] / (2 * pcpt_res[0]))
    target_state = [abs_model.get_state_index([12, 60]), abs_model.get_state_index([16, 60])]
    occ_measure = LP_prob.solve(P_matrix, cost_map, init_state, target_state)
    # occ_measure = LP_prob.solve(P_matrix, cost_map, init_state, gamma=0.99)
    optimal_policy, Z = LP_prob.extract(occ_measure)
    Z = Z.reshape((int(map_size[1]/pcpt_res[1]), int(map_size[0]/pcpt_res[0])))
    # Z = np.delete(Z, 0, axis=1)

    # ---------- Visualization ---------------------

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d', computed_zorder=True)
    plt.axis('off')
    ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False, alpha=0.7)
    # for obs in square_obs_list:
    #    ax.bar3d(obs[0] + 32, obs[1], 0., obs[2], obs[3], 0.01, color = 'yellow', alpha=1)
    # ax.bar3d(target_region[0], target_region[1], 0., target_region[2], target_region[3], 0.01, color='green', alpha=1)

    z_panel = -3
    rect = plt.Rectangle((0, 0), map_size[0]- pcpt_res[0], map_size[1]- pcpt_res[1], color='grey', alpha=0.5)
    ax.add_patch(rect)
    art3d.pathpatch_2d_to_3d(rect, z=z_panel, zdir="z")
    for obs in square_obs_list:
        rect = plt.Rectangle((obs[0]+32, obs[1]), obs[2], obs[3], color='red', alpha=1)
        ax.add_patch(rect)
        art3d.pathpatch_2d_to_3d(rect, z=z_panel, zdir="z")
    rect = plt.Rectangle(target_region[:2], target_region[2], target_region[3], color='green')
    ax.add_patch(rect)
    art3d.pathpatch_2d_to_3d(rect, z=z_panel, zdir="z")

    ax.plot([0, 0], [0, 0], [z_panel, 0], '--', color='black', alpha=0.5)
    ax.plot([map_size[0]-pcpt_res[0], map_size[0]-pcpt_res[0]], [0, 0], [z_panel, 0], '--', color='black', alpha=0.5)
    ax.plot([0, 0], [map_size[1]-pcpt_res[1], map_size[1]-pcpt_res[1]], [z_panel, 0], '--', color='black', alpha=0.5)
    ax.plot([map_size[0]-pcpt_res[0], map_size[0]-pcpt_res[0]],
            [map_size[1]-pcpt_res[1], map_size[1]-pcpt_res[1]], [z_panel, 0], '--', color='black', alpha=0.5)


    traj_pos = [28,0]
    action_arrow = np.array([0, 0])
    traj_state = abs_model.get_state_index(traj_pos)
    for i in range(14):
        traj_pos = [int(a) for a in traj_pos + action_arrow]
        state_index = abs_model.get_state_index(traj_pos)
        action_index = optimal_policy[state_index]
        action_vec = abs_model.action_set[int(action_index)]
        action_arrow = np.multiply(action_vec, pcpt_res)
        ax.quiver(traj_pos[0] + pcpt_res[0]/2, traj_pos[1] + pcpt_res[1]/2, 0.1, action_arrow[0], action_arrow[1], 0,
                  arrow_length_ratio=0.02, color='r', linewidth = 2)

    plt.show()


if __name__ == '__main__':
    main()