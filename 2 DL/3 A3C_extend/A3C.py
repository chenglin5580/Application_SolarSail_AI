"""
Asynchronous Advantage Actor Critic (A3C) with continuous
action space, Reinforcement Learning by MoFan.
Add API by MrFive
"""

import multiprocessing
import threading
import tensorflow as tf
import numpy as np
import os
import shutil
import matplotlib
# matplotlib.use('Agg')
from display import display

import sys
import copy


# tf.set_random_seed(2)


class Para:
    def __init__(self,
                 env,  # 环境参数包括state_dim,action_dim,abound,step,reset
                 a_constant=True,  # 动作是否是连续
                 units_a=30,  # 双层网络，第一层的大小
                 units_c=100,  # 双层网络，critic第一层的大小
                 MAX_GLOBAL_EP=2000,  # 全局需要跑多少轮数
                 UPDATE_GLOBAL_ITER=30,  # 多少代进行一次学习，调小一些学的比较快
                 gamma=0.9,  # 奖励衰减率
                 ENTROPY_BETA=0.01,  # 表征探索大小的量，越大结果越不确定
                 LR_A=0.0001,  # Actor的学习率
                 LR_C=0.001,  # Crtic的学习率
                 sigma_mul = 0.1, #sigma的乘子
                 MAX_EP_STEP=510,  # 控制一个回合的最长长度
                 train=True  # 表示训练
                 ):
        self.N_WORKERS = multiprocessing.cpu_count()
        self.MAX_EP_STEP = MAX_EP_STEP
        self.MAX_GLOBAL_EP = MAX_GLOBAL_EP
        self.GLOBAL_NET_SCOPE = 'Global_Net'
        self.UPDATE_GLOBAL_ITER = UPDATE_GLOBAL_ITER
        self.gamma = gamma
        self.units_a = units_a
        self.units_c = units_c
        self.a_constant = a_constant
        self.ENTROPY_BETA = ENTROPY_BETA
        self.LR_A = LR_A
        self.LR_C = LR_C
        self.sigma_mul = sigma_mul
        self.train = train

        # 保存网络位置
        self.model_path0 = os.path.join(sys.path[0], 'A3C_Net')
        if not os.path.exists(self.model_path0):
            os.mkdir(self.model_path0)
        self.model_path = os.path.join(self.model_path0, 'data.chkp')

        self.env = env
        self.N_S = env.ob_dim  # 状态的维度
        self.N_A = env.action_dim  # 动作的维度
        self.A_BOUND = env.a_bound  # 动作的上下界


        self.GLOBAL_RUNNING_R = []
        self.GLOBAL_EP = 0

        self.SESS = tf.Session()
        with tf.device("/cpu:0"):
            self.OPT_A = tf.train.RMSPropOptimizer(self.LR_A, name='RMSPropA')  # actor优化器定义
            self.OPT_C = tf.train.RMSPropOptimizer(self.LR_C, name='RMSPropC')  # critic优化器定义


class A3C:
    # 类似于主函数，每个程序只有一个实例
    def __init__(self, para):
        self.para = para
        with tf.device("/cpu:0"):
            self.GLOBAL_AC = ACNet(para.GLOBAL_NET_SCOPE, para)  # 定义global ， 不过只需要它的参数空间
            self.workers = []
            for i in range(para.N_WORKERS):  # N_WORKERS 为cpu个数
                i_name = 'W_%i' % i  # worker name，形如W_1
                self.workers.append(Worker(i_name, self.GLOBAL_AC, para))  # 添加名字为W_i的worker
        self.actor_saver = tf.train.Saver()

    def run(self):
        self.para.SESS.run(tf.global_variables_initializer())
        COORD = tf.train.Coordinator()
        worker_threads = []
        for worker in self.workers:
            job = lambda: worker.work()
            t = threading.Thread(target=job)
            t.start()
            worker_threads.append(t)
        COORD.join(worker_threads)
        self.actor_saver.save(self.para.SESS, self.para.model_path)


    def choose_action(self, state):
        return self.GLOBAL_AC.choose_best(state)

    def display(self, display_flag):
        if not self.para.train:
            self.actor_saver.restore(self.para.SESS, self.para.model_path)
        # display
        display(self, display_flag)


class ACNet(object):
    # AC框架网络，包含全局网络和每个分网络
    def __init__(self, scope, para, globalAC=None):
        self.para = para
        A_BOUND = self.para.A_BOUND
        if scope == self.para.GLOBAL_NET_SCOPE:  # get global network
            with tf.variable_scope(scope):
                self.s = tf.placeholder(tf.float32, [None, self.para.N_S], 'S')
                mu, sigma, self.v, self.a_params, self.c_params = self._build_net(scope)

                with tf.name_scope('wrap_a_out'):
                    mu, sigma = mu * abs(A_BOUND[1] - A_BOUND[0]) / 2 + \
                                np.mean(A_BOUND), sigma + 1e-4  # 归一化反映射，防止方差为零
            with tf.name_scope('choose_a'):  # use local params to choose action
                self.B = tf.clip_by_value(mu, self.para.A_BOUND[0], self.para.A_BOUND[1])  # 根据actor给出的分布，选取
                self.C = sigma
        else:  # worker, local net, calculate losses
            with tf.variable_scope(scope):
                # 网络引入
                self.s = tf.placeholder(tf.float32, [None, self.para.N_S], 'S')  # 状态
                self.a_his = tf.placeholder(tf.float32, [None, self.para.N_A], 'A')  # 动作
                self.v_target = tf.placeholder(tf.float32, [None, 1], 'Vtarget')  # 目标价值
                # 网络构建

                mu, sigma, self.v, self.a_params, self.c_params = self._build_net(scope)  # mu 均值 sigma 均方差
                with tf.name_scope('wrap_a_out'):
                    # 似乎sigma值定下来会效果更好
                    mu, sigma = mu * abs(A_BOUND[1] - A_BOUND[0]) / 2 + \
                                    np.mean(A_BOUND), sigma + 1e-4  # 归一化反映射，防止方差为零
                normal_dist = tf.contrib.distributions.Normal(mu, sigma)  # tf自带的正态分布函数
                with tf.name_scope('choose_a'):  # use local params to choose action
                    self.A = tf.clip_by_value(tf.squeeze(normal_dist.sample(1), axis=0), self.para.A_BOUND[0],
                                                  self.para.A_BOUND[1])  # 根据actor给出的分布，选取动作
                    self.B = tf.clip_by_value(mu, self.para.A_BOUND[0], self.para.A_BOUND[1])  # 根据actor给出的分布，选取动作
                    self.C = sigma

                # 价值网络优化
                td = tf.subtract(self.v_target, self.v, name='TD_error')
                with tf.name_scope('c_loss'):
                    self.c_loss = tf.reduce_mean(tf.square(td))

                # 动作网络优化
                with tf.name_scope('a_loss'):

                    log_prob = normal_dist.log_prob(self.a_his)  # 概率的log值
                    exp_v = log_prob * tf.stop_gradient(td)  # stop_gradient停止梯度传递的意思
                    entropy = normal_dist.entropy()
                    # encourage exploration，香农熵，评价分布的不确定性，鼓励探索，防止提早进入次优
                    self.exp_v = self.para.ENTROPY_BETA * entropy + exp_v
                    self.a_loss = tf.reduce_mean(-self.exp_v)  # actor的优化目标是价值函数最大


                with tf.name_scope('local_grad'):
                    self.a_grads = tf.gradients(self.a_loss, self.a_params)  # 计算梯度
                    self.c_grads = tf.gradients(self.c_loss, self.c_params)  # 计算梯度
            with tf.name_scope('sync'):
                with tf.name_scope('pull'):  # 把全局的pull到本地
                    self.pull_a_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.a_params, globalAC.a_params)]
                    self.pull_c_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.c_params, globalAC.c_params)]
                with tf.name_scope('push'):  # 根据本地的梯度，优化global的参数
                    self.update_a_op = self.para.OPT_A.apply_gradients(zip(self.a_grads, globalAC.a_params))
                    self.update_c_op = self.para.OPT_C.apply_gradients(zip(self.c_grads, globalAC.c_params))

    def _build_net(self, scope):  # 网络定义
        w_init = tf.random_normal_initializer(0., .1)
        with tf.variable_scope('actor'):
            l_a1 = tf.layers.dense(self.s, self.para.units_a, tf.nn.relu6, kernel_initializer=w_init, name='la1')
            l_a2 = tf.layers.dense(l_a1, self.para.units_a, tf.nn.relu6, kernel_initializer=w_init, name='la2')
            # l_a3 = tf.layers.dense(l_a2, self.para.units_a, tf.nn.relu6, kernel_initializer=w_init, name='la3')
            # l_a = tf.layers.dense(l_a3, self.para.units_a, tf.nn.relu6, kernel_initializer=w_init, name='la')

            mu = tf.layers.dense(l_a2, self.para.N_A, tf.nn.tanh, kernel_initializer=w_init, name='mu')
            # sigma_1 = tf.layers.dense(l_a2, self.para.N_A, tf.sigmoid, kernel_initializer=w_init, name='sigma')
            sigma_1 = tf.layers.dense(l_a2, self.para.N_A, tf.nn.softplus, kernel_initializer=w_init, name='sigma')
            sigma = tf.multiply(sigma_1, self.para.sigma_mul, name='scaled_a')

        with tf.variable_scope('critic'):
            l_c1 = tf.layers.dense(self.s, self.para.units_c, tf.nn.relu6, kernel_initializer=w_init, name='lc1')
            l_c2 = tf.layers.dense(l_c1, self.para.units_c, tf.nn.relu6, kernel_initializer=w_init, name='lc2')
            # l_c3 = tf.layers.dense(l_c2, self.para.units_c, tf.nn.relu6, kernel_initializer=w_init, name='lc3')
            # l_c = tf.layers.dense(l_c3, self.para.units_c, tf.nn.relu6, kernel_initializer=w_init, name='lc')
            v = tf.layers.dense(l_c2, 1, kernel_initializer=w_init, name='v')  # state value
        a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/actor')
        c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/critic')

        return mu, sigma, v, a_params, c_params


    def update_global(self, feed_dict):  # 函数：执行push动作
        self.para.SESS.run([self.update_a_op, self.update_c_op], feed_dict)  # local grads applies to global net

    def pull_global(self):  # 函数：执行pull动作
        self.para.SESS.run([self.pull_a_params_op, self.pull_c_params_op])

    # 连续动作，全局网络选择连函数等价
    def choose_action(self, s):  # 函数：选择动作action
        s = s[np.newaxis, :]
        return self.para.SESS.run(self.A, {self.s: s})[0]

    def choose_best(self, s): # 函数：选择最好的动作action
        s = s[np.newaxis, :]
        mu = self.para.SESS.run(self.B, {self.s: s})[0]
        sigma = self.para.SESS.run(self.C, {self.s: s})[0]
        return mu, sigma

class Worker(object):
    # 并行处理核的数量为实例数量
    # 拥有一个AC_net
    def __init__(self, name, globalAC, para):
        self.name = name
        self.para = para
        self.env_l = copy.deepcopy(self.para.env)
        self.AC = ACNet(name, para, globalAC)

    def work(self):
        # 并行处理核的环境是独立的
        # 某些参数是共享的例如self.para.GLOBAL_EP
        while self.para.GLOBAL_EP < self.para.MAX_GLOBAL_EP:
            s = self.env_l.reset()
            a = self.AC.choose_action(s)  # 选取动作
            s_, r, done, info = self.env_l.step(a)
            r = np.array([r])
            s_in = s[np.newaxis, :]
            a_in = a[np.newaxis, :]
            r_in = r[np.newaxis, :]

            feed_dict = {
                self.AC.s: s_in,
                self.AC.a_his: a_in,
                self.AC.v_target: r_in,
            }

            self.AC.update_global(feed_dict)
            self.AC.pull_global()

            if done:  # 每个片段结束，输出一下结果
                self.para.GLOBAL_RUNNING_R.append(r)
                if self.para.GLOBAL_EP % 1 == 0:
                    print(
                        self.name,
                        "Ep:", self.para.GLOBAL_EP,
                        # "r_f: %.4f" % r,
                        "| Ep_r: %.4f" % self.para.GLOBAL_RUNNING_R[-1],
                    )
                self.para.GLOBAL_EP += 1




if __name__ == '__main__':
    env = ENV()
    para = A3C.Para(env,
                    a_constant = False,
                    units_a=10,
                    units_c=20,
                    MAX_GLOBAL_EP=40000,
                    UPDATE_GLOBAL_ITER=2,
                    gamma=0.9,
                    ENTROPY_BETA=0.1,
                    LR_A=0.0007,
                    LR_C=0.001)
    RL = A3C.A3C(para)
    RL.run()
    #可以使用
    RL.choose_action()