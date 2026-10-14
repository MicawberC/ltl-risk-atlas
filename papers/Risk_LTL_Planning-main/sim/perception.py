import numpy as np

Inf = 9999

def gen_pcpt_dic(road_size, square_obs_list, cost):
    pcpt_dic = []
    pcpt_dic.append(([[-Inf, -road_size[0] / 2], [0, Inf]], -5))
    pcpt_dic.append(([[road_size[0] / 2, Inf], [0, Inf]], -5))

    for square_obs in square_obs_list:
        pcpt_dic.append(([[square_obs[0], square_obs[0] + square_obs[2]],
                          [square_obs[1], square_obs[1] + square_obs[3]]], cost))
    return pcpt_dic



def gen_cost_map(perception_dic, car_pos, perception_range, res):
    N = int(perception_range[0] / res[0]) # Note: N should be an odd number
    n = int(np.floor(N / 2))
    m = int(perception_range[1] / res[1])
    cost_map = np.zeros((N, m))
    # cost_map[n, -1] = 1
    for (area, cost) in perception_dic:
        # cost_map[int((area[0][0]+dla)/res): int((area[0][1]+dla)/res), int((area[1][0]+dla)/res): int((area[1][1]+dla)/res)] += cost
        relative_area = [[area[0][0] - car_pos[0], area[0][1] - car_pos[0]], [area[1][0] - car_pos[1], area[1][1] - car_pos[1]]]
        print("relative area:", relative_area)
        lb_x = int(min(max(np.floor(relative_area[0][0] / res[0]) + n, 0), 2*n+1))
        ub_x = int(min(max(np.ceil(relative_area[0][1] / res[0]) + n, 0), 2*n+1))
        lb_y = int(min(max(np.floor(relative_area[1][0] / res[1]), 0), m+1))
        ub_y = int(min(max(np.ceil(relative_area[1][1] /res[1]), 0), m+1))
        print("x-y bound:", lb_x, ub_x, lb_y, ub_y)
        cost_map[lb_x: ub_x, lb_y: ub_y] = cost
        print(cost_map)
    return cost_map



if __name__ == '__main__':
    road_size = [12, 80]
    square_obs = [-6, 30, 6, 10]
    pcpt_dic = gen_pcpt_dic(road_size, square_obs)
    # pcpt_dic = [([[-Inf, -6], [0, 80]], -5), ([[6, Inf], [0, 80]], -5), ([[-6, 0], [30, 40]], -5)]
    cost_map = gen_cost_map(pcpt_dic, (0, 20), (20, 20), 2)
    print(cost_map)