

import matplotlib.pyplot as plt
import numpy as np

def display(A3C, display_flag):

    if display_flag == 1:

        # reset
        observation = A3C.para.env.reset()

        ob_profile = np.empty((0, 4))
        time_profile = np.empty(0)
        action_profile = np.empty(0)

        reward_t = 0

        while True:

            action = A3C.GLOBAL_AC.choose_best(observation)
            # action = A3C.workers[0].AC.choose_best(observation)
            print(action)

            observation_, reward, done, info = A3C.para.env.step(action)

            reward_t += reward

            # memorize the profile
            ob_profile = np.vstack((ob_profile, observation))
            time_profile = np.hstack((time_profile, A3C.para.env.t))
            action_profile= np.hstack((action_profile, action))

            observation = observation_

            if done:
                break

        print('转移轨道时间%d天' % A3C.para.env.t)
        print(A3C.para.env.state)
        print(reward_t)

        plt.figure(1)
        plt.subplot(111, polar=True)
        theta = np.arange(0, 2 * np.pi, 0.02)
        plt.plot(theta, 1 * np.ones_like(theta), 'm')
        plt.plot(theta, 1.524 * np.ones_like(theta), 'b')
        plt.plot(ob_profile[:, 1], ob_profile[:, 0], 'r')

        plt.figure(2)
        plt.plot(time_profile, action_profile)
        # print(action_profile)

        plt.show()



    elif  display_flag == 2:
        state_now = A3C.para.env.reset_random()
        state_start = state_now

        state_track = []
        action_track = []
        time_track = []
        reward_track = []
        reward_me = 0
        while True:

            omega = A3C.GLOBAL_AC.choose_action(state_now)
            print(omega)
            state_next, reward, done, info = A3C.para.env.step(omega)

            state_track.append(state_next.copy())
            action_track.append(info['action'])
            time_track.append(info['time'])
            reward_track.append(info['reward'])

            state_now = state_next
            reward_me += info['reward']

            if done:
                break

        print('start', state_start)
        print('totla_reward', reward_me)
        print('x_end', A3C.para.env.x)
        plt.figure(1)
        plt.plot(time_track, [x[0] for x in state_track])
        plt.grid()
        plt.title('x')

        #
        plt.figure(2)
        plt.plot(time_track, action_track)
        plt.title('action')
        plt.grid()

        plt.figure(3)
        plt.plot(time_track, reward_track)
        plt.grid()
        plt.title('reward')

        plt.show()
    elif display_flag == 3:

        plt.axis([-0.1, 1.2, -0.1, 1.2])
        plt.ion()
        plt.grid(color='g', linewidth='0.3', linestyle='--')

        ep_num = 10

        state_track = np.zeros([ep_num, 200])
        action_track = np.zeros([ep_num, 200])
        time_track = np.zeros([ep_num, 200])
        reward_track = np.zeros([ep_num, 200])
        step_all = np.zeros([ep_num])

        for ep in range(ep_num):
            print('step', ep)
            state_now = A3C.para.env.reset()
            reward_me = 0
            for step in range(1000):

                action = A3C.GLOBAL_AC.choose_action(state_now)
                state_next, reward, done, info = A3C.para.env.step(action)
                state_track[ep, step]= info['x']
                action_track[ep, step] = info['action']
                time_track[ep, step] = info['time']
                reward_track[ep, step] = info['reward']

                state_now = state_next
                reward_me += info['reward']

                if done:
                    print('x_error', info['x']-1)
                    step_all[ep] = step
                    break

        plt.scatter(1, 1, color='r')
        color_list = ['b', 'c', 'g', 'k', 'm', 'w', 'y','b', 'c', 'g', 'k', 'm', 'w', 'y','b', 'c', 'g', 'k', 'm',
                      'w', 'y','b', 'c', 'g', 'k', 'm', 'w', 'y','b', 'c', 'g', 'k', 'm', 'w', 'y','b', 'c', 'g',
                      'k', 'm', 'w', 'y']
        for aa in range(200):
            for bb in range(ep_num):
                if aa<=step_all[bb]:
                    plt.scatter(time_track[bb, aa], state_track[bb, aa], color=color_list[bb], marker='.')
            plt.pause(0.001)
        plt.pause(100000000)





