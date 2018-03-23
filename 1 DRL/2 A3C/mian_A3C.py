

import A3C as A3C

from SolarSail2 import SolarSail as Objective_AI

env = Objective_AI()
train_flag = True
train_flag = False
para = A3C.Para(env,  # 环境参数包括state_dim,action_dim,abound,step,reset
                a_constant=True,  # 动作是否是连续
                units_a=100,  # 双层网络，第一层的大小
                units_c=100,  # 双层网络，critic第一层的大小
                MAX_GLOBAL_EP=2000,  # 全局需要跑多少轮数
                UPDATE_GLOBAL_ITER=50,  # 多少代进行一次学习，调小一些学的比较快
                gamma=0.9,  # 奖励衰减率
                ENTROPY_BETA=0.0001,  # 表征探索大小的量，越大结果越不确定
                LR_A=0.0001,  # Actor的学习率
                LR_C=0.001,  # Crtic的学习率
                sigma_mul=1,
                MAX_EP_STEP=600,  # 控制一个回合的最长长度
                train=train_flag  # 表示训练
                )
RL = A3C.A3C(para)
if para.train:
    RL.run()
else:
    # 1 stable
    # 2 random
    # 3 multi
    RL.display(1)
#




