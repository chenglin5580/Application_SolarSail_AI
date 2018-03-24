
"""
DDPG is Actor-Critic based algorithm
Designer: Lin Cheng  17.08.2017
"""

########################### Package  Input  #################################

from Method import Method as Method
from SolarSail import SolarSail as Object_AI
import numpy as np
import matplotlib.pyplot as plt

############################ Hyper Parameters #################################

max_Episodes = 30000
max_Ep_Steps = 2000
rendering = False
############################ Object and Method  ####################################

env = Object_AI()

s_dim = env.state_dim
print("环境状态空间维度为", s_dim)
print('-----------------------------\t')
a_dim = env.action_dim
print("环境动作空间维度为", a_dim)
print('-----------------------------\t')
a_bound = env.a_bound
print("环境动作空间的上界为", a_bound)
print('-----------------------------\t')


## method settting
train_flag = True
train_flag = False
RLmethod = Method(
            env.action_dim,  # 动作的维度
            env.state_dim,  # 状态的维度
            env.a_bound,  # 动作的上下限
            e_greedy_end=0.05,  # 最后的探索值 0.1倍幅值
            e_liner_times=1500*50,  # 探索值经历多少次学习变成e_end
            epilon_init=0.5,  # 表示1倍的幅值作为初始值
            LR_A=0.001,  # Actor的学习率
            LR_C=0.001,  # Critic的学习率
            GAMMA=0.9,  # 衰减系数
            TAU=0.01,  # 软替代率，例如0.01表示学习eval网络0.01的值，和原网络0.99的值
            MEMORY_SIZE=3000,  # 记忆池容量
            BATCH_SIZE=128,  # 批次数量
            units_a=10,  # Actor神经网络单元数
            units_c=1000,  # Crtic神经网络单元数
            tensorboard=True,  # 是否存储tensorboard
            train=train_flag  # 训练的时候有探索
            )

###############################  training  ####################################

observation = env.reset()

if RLmethod.train:
    for i in range(max_Episodes):

        action = RLmethod.chose_action(observation)

        observation, reward, done, info = env.step(action)

        RLmethod.store_transition(observation, action, reward)

        RLmethod.learn()

        print('step', i, 'reward', reward)

    RLmethod.net_save()

else:
    # test the critic
    action = np.array([(-1.609601+5)/10, (0.042179+5)/10, (-0.160488+5)/10, (-1.597537+5)/10, (568-100)/500])
    observation, reward, done, info = env.step(action)
    ba = action
    bq = reward

    for _ in range(10):
        action = np.random.rand(1, 5).reshape(5)
        observation, reward, done, info = env.step(action)
        ba = np.vstack((ba, action))
        bq = np.vstack((bq, reward))

    RLmethod.verify(ba, bq)

    # test the actor
    action = RLmethod.chose_action(observation)
    observation, reward, done, info = env.step(action)
    print('reward_actor', reward)

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
    plt.plot(alpha_profile * 57.3, 'm')
    plt.title('alpha')

    plt.figure(6)
    plt.plot(info['reward_profile'], 'm')
    plt.title('reward')

    plt.show()










