

import matplotlib.pyplot as plt
import numpy as np

def display(A3C, display_flag):

    if display_flag == 1:
        # test the critic
        print('----------------paper result-------------------')
        action = np.array([(-1.609601) / 5, (0.042179) / 5, (-0.160488) / 5, (-1.597537) / 5, (568 - 350) / 250])
        observation, reward, done, info = A3C.para.env.step(action)
        print(action)
        lambda_all = action[0:4] * 5
        td_f = action[4] * 250 + 350
        print('lambda_all', lambda_all)
        print('td_f', td_f)
        print('reward_actor', reward)
        print('total_day', A3C.para.env.td)
        print('r_f_error', A3C.para.env.constant['r_f'] - A3C.para.env.state[0])
        print('u_f_error', A3C.para.env.state[2])
        print('v_f_error', A3C.para.env.state[3] - A3C.para.env.constant['v_f'])


        print('----------------DL result-------------------')
        observation = A3C.para.env.reset()
        info = {}
        # test the actor
        action, sigma = A3C.GLOBAL_AC.choose_best(observation)
        print(action)
        lambda_all = action[0:4] * 5
        td_f = action[4] * 250 + 350
        print('lambda_all', lambda_all)
        print('td_f', td_f)
        observation, reward, done, info = A3C.para.env.step(action)
        print('reward_actor', reward)
        print('total_day', A3C.para.env.td)
        print('r_f_error', (A3C.para.env.constant['r_f'] - A3C.para.env.state[0]))
        print('r_f_error', (A3C.para.env.constant['r_f'] - A3C.para.env.state[0])*A3C.para.env.AU/1000)
        print('u_f_error', (A3C.para.env.state[2]))
        print('u_f_error', (A3C.para.env.state[2])*A3C.para.env.VU)
        print('v_f_error', (A3C.para.env.state[3] - A3C.para.env.constant['v_f']))
        print('v_f_error', (A3C.para.env.state[3] - A3C.para.env.constant['v_f'])*A3C.para.env.VU)

        print('----------------Critic result-------------------')
        # A3C.GLOBAL_AC.critic_verify(observation, action, reward)

        ob_profile = info['ob_profile']
        alpha_profile = info['alpha_profile']

        plt.subplot(111, polar=True)
        theta = np.arange(0, 2 * np.pi, 0.02)
        plt.plot(theta, 1 * np.ones_like(theta), 'm')
        plt.plot(theta, 1.524 * np.ones_like(theta), 'b')
        plt.plot(ob_profile[:, 1], ob_profile[:, 0], 'r')

        plt.figure(2)
        plt.plot(ob_profile[:, 0], 'm')
        plt.plot(A3C.para.env.constant['r_f'] * np.ones(len(ob_profile[:, 0])))
        plt.title('r')
        plt.ylim((0, 1.7))

        plt.figure(3)
        plt.plot(ob_profile[:, 2], 'm')
        plt.plot(A3C.para.env.constant['u_f'] * np.ones(len(ob_profile[:, 0])))
        plt.title('v')
        plt.title('u')
        plt.ylim((-0.5, 0.5))

        plt.figure(4)
        plt.plot(ob_profile[:, 3], 'm')
        plt.plot(A3C.para.env.constant['v_f'] * np.ones(len(ob_profile[:, 0])))
        plt.title('v')
        plt.ylim((0.5, 1.2))

        plt.figure(5)
        plt.plot(alpha_profile * 57.3, 'm')
        plt.title('alpha')

        plt.figure(6)
        plt.plot(info['reward_profile'], 'm')
        plt.title('reward')




        plt.show()





