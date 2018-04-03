# 太阳帆运动模型搭建
import numpy as np
import matplotlib.pyplot as plt


class SolarSail:

    def __init__(self, random=False):
        self.t = None
        self.state = None
        self.random = random
        # 归一化参数长度除以AU,时间除以TU
        self.AU = 1.4959787 * (10 ** 11)
        self.mu = 1.32712348 * (10 ** 20)
        self.VU = np.sqrt(self.mu / self.AU)
        self.TU = np.sqrt(self.AU ** 3 / self.mu)
        self.constant = {'beta': 0.5 / 5.93, 'u0': 0, 'phi0': 0, 'r_f': 1.524, 'u_f': 0, 'phi_f': 0}
        self.constant['v_f'] = 1.0 / np.sqrt(self.constant['r_f'])
        # 特征加速度ac和光压因子beta或者说k的转换关系ac = 5.93beta
        self.delta_d = 1  # 仿真步长，未归一化，单位天
        self.delta_t = self.delta_d * (24 * 60 * 60) / self.TU  # 无单位
        self.reset()
        self.ob_dim = len(self.observation)
        self.action_dim = 5
        self.a_bound = np.array([-1*np.ones(self.action_dim), 1*np.ones(self.action_dim)])
        self.c1 = 200
        self.c2 = 200
        self.c3 = 200



    def render(self):
        pass

    def reset(self):
        self.t = 0
        self.td = 0
        self.delta_d = 1  # 仿真步长，未归一化，单位天
        self.delta_t = self.delta_d * (24 * 60 * 60) / self.TU  # 无单位
        if self.random == True:
            rand_r0 = np.random.rand(1)
            self.constant['r0'] = (0.1 * rand_r0 + 1.0)[0]
            self.constant['v0'] = 1.0 / np.sqrt(self.constant['r0'])
        else:
            rand_r0 = 0.
            self.constant['r0'] = (0.2 * rand_r0 + 1.0)
            self.constant['v0'] = 1.0 / np.sqrt(self.constant['r0'])

        self.state = np.array([self.constant['r0'], self.constant['phi0'],
                               self.constant['u0'], self.constant['v0']])  # [r phi u v]
        self.observation = np.array([self.constant['r0'], rand_r0])

        return self.observation.copy()

    def step(self, action):
        # 传入单位是度

        ob_profile = np.empty((0, 4))
        alpha_profile = np.empty((0, 1))
        reward_profile = np.empty((0, 1))
        lambda_all = action[0:4] * 10
        td_f = action[4] * 250 + 350

        while True:
            lambda1, lambda2, lambda3, lambda4 = lambda_all
            r, phi, u, v = self.state  # 当前状态的参数值
            if np.abs(lambda4) < 0.0001:
                print('lambad4=', lambda4)
                if lambda3 <= 0:
                    alpha = 0
                else:
                    alpha = np.pi / 2
            else:
                aaa = (-3 * lambda3 - np.sqrt(9*lambda3**2+8*lambda4**2))/4/lambda4
                alpha = np.arctan(aaa)

            if self.td > td_f - 3:
                self.delta_d = 0.01  # 仿真步长，未归一化，单位天
                self.delta_t = self.delta_d * (24 * 60 * 60) / self.TU  # 无单位

            # 求state微分
            if r > 0.001:
                r_dot = u
                phi_dot = v / r
                u_dot = self.constant['beta'] * ((np.cos(alpha)) ** 3) / (r ** 2) + \
                        (v ** 2) / r - 1 / (r ** 2)
                v_dot = self.constant['beta'] * np.sin(alpha) * (np.cos(alpha) ** 2) / (r ** 2) - u * v / r

                lambda1_dot = lambda2 * v / (r ** 2) + \
                              lambda3 * (2 * self.constant['beta'] * np.cos(alpha) ** 3 / (r ** 3) + \
                                         v ** 2 / (r ** 2) - 2 / (r ** 3)) + \
                              lambda4 * (2 * self.constant['beta'] * np.sin(alpha) * np.cos(alpha) ** 2 / (r ** 3) - \
                                         u * v ** 2 / r)
                lambda2_dot = 0
                lamnad3_dot = - lambda1 + lambda4 * v / r
                lambda4_dot = - lambda2 / r - 2 * lambda3 * v / r + lambda4 * u / r

                # 下一个状态
                self.state += self.delta_t * np.array([r_dot, phi_dot, u_dot, v_dot])  # [r,phi,u,v]
                lambda_all += self.delta_t * np.array([lambda1_dot, lambda2_dot, lamnad3_dot, lambda4_dot])

                self.td += self.delta_d
                self.t += self.delta_t

                # memory
                ob_profile = np.vstack((ob_profile, self.state))
                alpha_profile = np.vstack((alpha_profile, alpha))
                # reward_profile = np.vstack((reward_profile, reward))

                # terminate
                # if self.state[3] >= self.constant['v_f']:
                if self.td >= td_f:

                    # reward calculation

                    reward = 30 - self.t - self.c1 * np.abs(self.constant['r_f'] - self.state[0]) - \
                             self.c2 * np.abs(self.constant['u_f'] - self.state[2]) - \
                             self.c3 * np.abs(self.constant['v_f'] - self.state[3])

                    done = True
                    info = {}
                    info['ob_profile'] = ob_profile
                    info['alpha_profile'] = alpha_profile
                    info['reward_profile'] = reward_profile
                    if reward > 1000:
                        reward = 1000
                    elif reward < -1000:
                        reward = -1000
                    break
            else:
                print('trajectory r ===============================0')
                done = True
                info = {}
                info['ob_profile'] = ob_profile
                info['alpha_profile'] = alpha_profile
                info['reward_profile'] = reward_profile
                reward = -1000
                break

        return self.observation.copy(), reward, done, info


if __name__ == '__main__':

    env = SolarSail()
    # action = np.array([(-1.609601 + 5) / 10, (0.042179 + 5) / 10, \
    #                    (-0.160488 + 5) / 10, (-1.597537 + 5) / 10, (568 - 100) / 500])
    action = np.array([-0.9999999, -0.9999932, 1., -1., 0.9961483])
    observation, reward, done, info = env.step(action)
    print(reward)

    ob_profile = info['ob_profile']
    alpha_profile = info['alpha_profile']

    plt.subplot(111, polar=True)
    theta = np.arange(0, 2 * np.pi, 0.02)
    plt.plot(theta, 1 * np.ones_like(theta), 'm')
    plt.plot(theta, 1.524 * np.ones_like(theta), 'b')
    plt.plot(ob_profile[:, 1], ob_profile[:, 0], 'r')

    plt.figure(2)
    plt.plot(ob_profile[:, 0], 'm')
    plt.plot(env.constant['r_f']*np.ones(len(ob_profile[:, 0])))
    plt.title('r')

    plt.figure(3)
    plt.plot(ob_profile[:, 2], 'm')
    plt.plot(env.constant['u_f'] * np.ones(len(ob_profile[:, 0])))
    plt.title('v')
    plt.title('u')

    plt.figure(4)
    plt.plot(ob_profile[:, 3], 'm')
    plt.plot(env.constant['v_f'] * np.ones(len(ob_profile[:, 0])))
    plt.title('v')

    plt.figure(5)
    plt.plot(alpha_profile*57.3, 'm')
    plt.title('alpha')

    plt.figure(6)
    plt.plot(info['reward_profile'], 'm')
    plt.title('reward')

    plt.show()

