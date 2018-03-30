'''
Add API by MrFive
DDPG Method
'''

# 这里面的DDPG是双层网络
# 增加done的处理
# 可以保存网络
# 三层网络

import tensorflow as tf
import numpy as np
import os
import sys


# tf.set_random_seed(2)


class Method(object):
    def __init__(
            self,
            method,
            a_dim,  # 动作的维度
            ob_dim,  # 状态的维度
            a_bound,  # 动作的上下限
            e_greedy_end=0.1,  # 最后的探索值 0.1倍幅值
            e_liner_times=1000,  # 探索值经历多少次学习变成e_end
            epilon_init=1,  # 表示1倍的幅值作为初始值
            LR_A=0.0001,  # Actor的学习率
            LR_C=0.001,  # Critic的学习率
            GAMMA=0.9,  # 衰减系数
            TAU=0.01,  # 软替代率，例如0.01表示学习eval网络0.01的值，和原网络0.99的值
            MEMORY_SIZE=10000,  # 记忆池容量
            BATCH_SIZE=256,  # 批次数量
            units_a=64,  # Actor神经网络单元数
            units_c=64,  # Crtic神经网络单元数
            actor_learn_start=10000,  # Actor开始学习的代数
            tensorboard=True,
            train=True  # 训练的时候有探索
    ):
        # DDPG网络参数
        # self.method = method + '/' + 'LR_A' + str(LR_A) + '/' + 'LR_C' + str(LR_C) + '/' + 'units_a' + str(units_a) \
        #               + '/' + 'units_c' + str(units_c)
        self.method = method
        self.LR_A = LR_A
        self.LR_C = LR_C
        self.GAMMA = GAMMA
        self.TAU = TAU
        self.MEMORY_CAPACITY = MEMORY_SIZE
        self.BATCH_SIZE = BATCH_SIZE
        self.units_a = units_a
        self.units_c = units_c
        self.actor_learn_start = actor_learn_start
        self.epsilon_init = epilon_init  # 初始的探索值
        self.epsilon = self.epsilon_init
        self.epsilon_end = e_greedy_end
        self.e_liner_times = e_liner_times
        self.train = train
        self.tensorboard = tensorboard
        self.abound_low = np.array([-1, -1, -1, -1, -1])
        self.abound_high = np.array([1, 1, 1, 1, 1])
        self.OPT_A = tf.train.RMSPropOptimizer(self.LR_A, name='RMSPropA')  # actor优化器定义

        self.pointer = 0
        self.a_replace_counter, self.c_replace_counter = 0, 0
        self.iteration = 0

        self.model_path0 = os.path.join(sys.path[0], 'DDPG_Net')
        if not os.path.exists(self.model_path0):
            os.mkdir(self.model_path0)
        self.model_path = os.path.join(self.model_path0, 'data.chkp')

        # DDPG构建
        self.memory = np.zeros((self.MEMORY_CAPACITY, ob_dim + a_dim + 1), dtype=np.float32)  # 存储s,a,r,s_,done
        self.sess = tf.Session()

        self.a_dim, self.s_dim, self.a_bound = a_dim, ob_dim, a_bound,
        self.S = tf.placeholder(tf.float32, [None, ob_dim], 'state')
        self.a_his = tf.placeholder(tf.float32, [None, a_dim], 'action')
        self.q_target = tf.placeholder(tf.float32, [None, 1], 'q_target')

        # 建立actor网络
        with tf.variable_scope('Actor'):
            mu, sigma = self._build_a(self.S, scope='eval', trainable=True)
            mu, sigma = mu, sigma + 1e-4
            self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval')

        # 建立Critic网络
        with tf.variable_scope('Critic'):
            self.q = self._build_c(self.S, scope='eval', trainable=True)
            tf.summary.histogram('Critic/eval', self.q)
            self.ce_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval')

        # q train
        with tf.name_scope('q_train'):
            td = tf.subtract(self.q_target, self.q, name='TD_error')
            self.c_loss = tf.reduce_mean(tf.square(td))
            tf.summary.scalar('c_loss', self.c_loss)
            self.ctrain = tf.train.AdamOptimizer(self.LR_C).minimize(self.c_loss, var_list=self.ce_params)

        # choose_a
        with tf.name_scope('choose_a'):  # use local params to choose action
            normal_dist = tf.contrib.distributions.Normal(mu, sigma)  # tf自带的正态分布函数
            self.A = tf.clip_by_value(tf.squeeze(normal_dist.sample(1), axis=0), self.abound_low, self.abound_high)  # 根据actor给出的分布，选取动作
            self.B = tf.clip_by_value(mu, self.abound_low, self.abound_high)  # 根据actor给出的分布，选取动作
            self.C = sigma

        # 动作网络优化
        with tf.name_scope('a_train'):
            log_prob = normal_dist.log_prob(self.a_his)  # 概率的log值
            exp_v = log_prob * tf.stop_gradient(td)  # stop_gradient停止梯度传递的意思
            entropy = normal_dist.entropy()
            # encourage exploration，香农熵，评价分布的不确定性，鼓励探索，防止提早进入次优
            self.exp_v = 0.01 * entropy + exp_v
            self.a_loss = tf.reduce_mean(-self.exp_v)  # actor的优化目标是价值函数最大
            self.a_grads = tf.gradients(self.a_loss, self.ae_params)  # 计算梯度
            self.atrain = self.OPT_A.apply_gradients(zip(self.a_grads, self.ae_params))
            tf.summary.scalar('a_loss', self.a_loss)

        self.actor_saver = tf.train.Saver()
        if self.train:
            self.sess.run(tf.global_variables_initializer())
        else:
            self.actor_saver.restore(self.sess, self.model_path)

        if self.train and self.tensorboard:
            self.merged = tf.summary.merge_all()
            self.writer = tf.summary.FileWriter('./logs/' + self.method, self.sess.graph)

    def chose_action(self, s):
        if self.train:
            s = s[np.newaxis, :]
            action = self.sess.run(self.A, {self.S: s})[0]
            mu = self.sess.run(self.B, {self.S: s})[0]
            sigma = self.sess.run(self.C, {self.S: s})[0]
            q = self.sess.run(self.q, {self.S: s})[0]
            print('mu', mu, 'sigma', sigma, 'q', q)
        else:
            s = s[np.newaxis, :]
            action = self.sess.run(self.B, {self.S: s})[0]
        return action


    def learn(self):
        if self.pointer < self.MEMORY_CAPACITY:
            # 未存储够足够的记忆池的容量
            print('store')
            pass
        else:
            # 更新目标网络，有可以改进的地方，可以更改更新目标网络的频率，不过减小tau会比较好
            indices = np.random.choice(self.MEMORY_CAPACITY, size=self.BATCH_SIZE)
            bt = self.memory[indices, :]
            bs = bt[:, :self.s_dim]
            ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
            bq = bt[:, -1:]

            # 更新a和c，有可以改进的地方，可以适当更改一些更新a和c的频率
            self.sess.run(self.ctrain, {self.S: bs, self.a_his: ba, self.q_target: bq})
            self.sess.run(self.atrain, {self.S: bs, self.a_his: ba, self.q_target: bq})

            if self.tensorboard:
                if self.iteration % 10 == 0:
                    result_merge = self.sess.run(self.merged, {self.S: bs, self.a_his: ba, self.q_target: bq})
                    self.writer.add_summary(result_merge, self.iteration)

            self.iteration += 1


    def critic_verify(self, s, action, reward):
        bs = s[np.newaxis, :]
        ba = action[np.newaxis, :]

        q = self.sess.run(self.q, {self.S: bs, self.a: ba})
        td_error = q - reward
        print('td_error', td_error)

    def store_transition(self, s, a, r):
        # 存储需要的信息到记忆池
        transition = np.hstack((s, a, r))
        index = self.pointer % self.MEMORY_CAPACITY  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1

    def _build_a(self, s, scope, trainable):
        # 建立actor网络
        with tf.variable_scope(scope):
            n_l1 = self.units_a
            net0 = tf.layers.dense(s, n_l1, activation=tf.nn.relu, name='l0', trainable=trainable)
            net1 = tf.layers.dense(net0, n_l1, activation=tf.nn.relu, name='l1', trainable=trainable)
            mu = tf.layers.dense(net1, self.a_dim, activation=tf.nn.tanh, name='mu', trainable=trainable)
            # sigma = tf.layers.dense(net1, self.a_dim, activation=tf.sigmoid, name='sigma_1', trainable=trainable)
            sigma = tf.layers.dense(net1, self.a_dim, activation=tf.nn.softplus, name='sigma_1', trainable=trainable)
            # sigma = tf.multiply(sigma_1, 1, name='sigma')
            return mu, sigma


    def _build_c(self, s, scope, trainable):
        # 建立critic网络
        with tf.variable_scope(scope):
            n_l1 = self.units_c
            net1 = tf.layers.dense(s, n_l1, activation=tf.nn.relu, name='l1', trainable=trainable)
            net2 = tf.layers.dense(net1, n_l1, activation=tf.nn.relu, name='l2', trainable=trainable)
            net3 = tf.layers.dense(net2, n_l1, activation=tf.nn.relu, name='l3', trainable=trainable)
            # net4 = tf.layers.dense(net3, n_l1, activation=tf.nn.relu, name='l4', trainable=trainable)
            # net5 = tf.layers.dense(net4, n_l1, activation=tf.nn.relu, name='l5', trainable=trainable)
            # net6 = tf.layers.dense(net5, n_l1, activation=tf.nn.relu, name='l6', trainable=trainable)
            q = tf.layers.dense(net3, 1, trainable=trainable)  # Q(s,a)
            return q

    def net_save(self):
        self.actor_saver.save(self.sess, self.model_path)
