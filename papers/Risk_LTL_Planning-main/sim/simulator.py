import numpy as np

def f_dyn(q, u, params):

    WB = params['WB']
    x, y, theta = q
    v, tan_delta = u
    q_dot = np.zeros(q.shape)
    q_dot[0] = v * np.cos(theta)
    q_dot[1] = v * np.sin(theta)
    q_dot[2] = v * tan_delta / WB

    return q_dot


def car_dyn(state, input, param):
    p_x, p_y, yaw = state
    velocity, angle_vec = input
    new_state = [p_x + np.cos(yaw) * velocity * param['dt'],
             p_y + np.sin(yaw) * velocity * param['dt'],
             yaw + velocity * angle_vec * param['dt'] / param['WB']]
    return new_state


def update_1(state, action, params):
    dic = {'-2': -1.0, '-1': -0.5, '0': 0, '1': 0.5, '2': 1.0}
    vx = dic[str(action[0])]
    vy = dic[str(action[1])]
    v = np.linalg.norm([vx, vy])
    des_angle = np.arctan2(vy,vx)
    e = des_angle - state[2]
    e = (e + np.pi) % (2 * np.pi) - np.pi
    angle_vel = 0.5 * e
    input = (v, angle_vel)
    print("angle:", des_angle, state[2], e, angle_vel)
    state = car_dyn(state, input, params)
    return state


def update_2(q, action, params):
    # dic = {'2': -0.2, '1': -0.1, '0': 0, '-1': 0.1, '-2': 0.2}
    dic = {'-2': -0.2, '-1': -0.1, '0': 0, '1': 0.1, '2': 0.2}
    u = (5, dic[action])
    q_ = q + f_dyn(q, u, params) * params["dt"]
    return q_

def oppo_update(q, u, params, max_vel):
    u[1] = max(-1, min(u[1], 1))
    q_ = q + f_dyn(q, u, params) * params["dt"]
    q_[3] = max(min(max_vel, q_[3]), 0)
    return q_
