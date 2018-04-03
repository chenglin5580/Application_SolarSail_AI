
"""
DDPG is Actor-Critic based algorithm
Designer: Lin Cheng  17.08.2017
"""

# TODO A3C结构改进

########################### Package  Input  #################################

from Method import Method as Method
from SolarSail import SolarSail as Object_AI
import numpy as np
import matplotlib.pyplot as plt

############################ Hyper Parameters #################################

max_Episodes = 50000
max_Ep_Steps = 2000
rendering = False
############################ Object and Method  ####################################

env = Object_AI(random=True)

ob_dim = env.ob_dim
print("环境状态空间维度为", ob_dim)
print('-----------------------------\t')
a_dim = env.action_dim
print("环境动作空间维度为", a_dim)
print('-----------------------------\t')
a_bound = env.a_bound
print("环境动作空间的上界为", a_bound)
print('-----------------------------\t')


## method settting
# tensorboard --logdir="2 DL/2 AC_extend/3 SL_MultiScene/logs"
method = 'random 1000,200,200+1'

train_flag = True
# train_flag = False
RLmethod = Method(
            method,
            env.action_dim,  # 动作的维度
            env.ob_dim,  # 状态的维度
            env.a_bound,  # 动作的上下限
            e_greedy_end=0.05,  # 最后的探索值 0.1倍幅值
            e_liner_times=1500*50,  # 探索值经历多少次学习变成e_end
            epilon_init=0.5,  # 表示1倍的幅值作为初始值
            LR_A=0.000002,  # Actor的学习率
            LR_C=0.0001,  # Critic的学习率
            GAMMA=0.9,  # 衰减系数
            TAU=0.01,  # 软替代率，例如0.01表示学习eval网络0.01的值，和原网络0.99的值
            MEMORY_SIZE=32,  # 记忆池容量
            BATCH_SIZE=32,  # 批次数量
            units_a=500,  # Actor神经网络单元数
            units_c=500,  # Crtic神经网络单元数
            actor_learn_start=5000,  # Actor开始学习的代数
            tensorboard=True,  # 是否存储tensorboard
            train=train_flag  # 训练的时候有探索
            )

###############################  training  ####################################


if RLmethod.train:
    for i in range(max_Episodes):

        observation = env.reset()
        # env.c1 += (1000 - 100)/max_Episodes
        # env.c2 += (200 - 20) / max_Episodes
        # env.c3 += (200 - 20) / max_Episodes

        action = RLmethod.chose_action(observation)
        # print(action)
        qqq, reward, done, info = env.step(action)

        RLmethod.store_transition(observation, action, reward)

        RLmethod.learn()

        print('step', i, 'reward', reward, 'total_day %.3f' % env.td)

    RLmethod.net_save()

else:
    # test the critic
    print('----------------paper result-------------------')
    action = np.array([(-1.609601)/10, (0.042179)/10, (-0.160488)/10, (-1.597537)/10, (568-350)/250])
    observation, reward, done, info = env.step(action)
    print('paper_reward', reward)
    ba = action
    bq = reward

    # for _ in range(10):
    #     action = np.random.rand(1, 5).reshape(5)
    #     observation, reward, done, info = env.step(action)
    #     ba = np.vstack((ba, action))
    #     bq = np.vstack((bq, reward))
    #
    # RLmethod.verify(ba, bq)

    print('----------------DL result-------------------')
    observation = env.reset()
    info = {}
    # test the actor
    action = RLmethod.chose_action(observation)
    print(action)
    lambda_all = action[0:4] * 5
    td_f = action[4] * 250 + 350
    print('lambda_all', lambda_all)
    print('td_f', td_f)
    observation, reward, done, info = env.step(action)
    print('reward_actor', reward)
    print('total_day',  env.t)
    print('r_f_error', (env.constant['r_f'] - env.state[0]))
    print('r_f_error', (env.constant['r_f'] - env.state[0]) * env.AU / 1000)
    print('u_f_error', (env.state[2]))
    print('u_f_error', (env.state[2]) * env.VU)
    print('v_f_error', (env.state[3] - env.constant['v_f']))
    print('v_f_error', (env.state[3] - env.constant['v_f']) * env.VU)
    print('v_f', (env.state[3]) * env.VU)
    # print('r_f_error', env.constant['r_f']-env.state[0])
    # print('u_f_error', env.state[2])
    # print('v_f_error', env.state[3]-env.constant['v_f'])

    print('----------------Critic result-------------------')
    # RLmethod.critic_verify(observation, action, reward)


    ob_profile = info['ob_profile']
    alpha_profile = info['alpha_profile']

    plt.subplot(111, polar=True)
    theta = np.arange(0, 2 * np.pi, 0.02)
    plt.plot(theta, 1 * np.ones_like(theta), 'm')
    plt.plot(theta, 1.524 * np.ones_like(theta), 'b')
    plt.plot(ob_profile[:, 1], ob_profile[:, 0], 'r')

    plt.figure(2)
    plt.plot(ob_profile[:, 0], 'm')
    plt.plot(env.constant['r_f'] * np.ones(len(ob_profile[:, 0])))
    plt.title('r')
    plt.ylim((0.8, 1.7))


    plt.figure(3)
    plt.plot(ob_profile[:, 2], 'm')
    plt.plot(env.constant['u_f'] * np.ones(len(ob_profile[:, 0])))
    plt.title('v')
    plt.title('u')
    plt.ylim((-0.5, 0.5))

    plt.figure(4)
    plt.plot(ob_profile[:, 3], 'm')
    plt.plot(env.constant['v_f'] * np.ones(len(ob_profile[:, 0])))
    plt.title('v')
    plt.ylim((0.5, 1.2))

    plt.figure(5)
    plt.plot(alpha_profile * 57.3, 'm')
    plt.title('alpha')

    plt.figure(6)
    plt.plot(info['reward_profile'], 'm')
    plt.title('reward')

    plt.show()



# setting

# tensorboard --logdir="2 DL/3 SL_MultiScene/logs"










