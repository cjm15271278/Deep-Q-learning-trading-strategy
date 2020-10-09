"""
This part of code is the DQN brain, which is a brain of the agent.
All decisions are made in here.
Using Tensorflow to build the neural network.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
Tensorflow: 1.0
gym: 0.7.3
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

np.random.seed(1)
tf.set_random_seed(1)


# Deep Q Network off-policy
class DeepQNetwork:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate,#=0.01,
            reward_decay,#=0.9,
            e_greedy,#=0.9,  # 百分之九十选择q值最大的动作
            replace_target_iter,#=300,
            memory_size,#=500,
            batch_size,#=32,
            e_greedy_increment=None, # 缩小随机的范围
            output_graph=False,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        # total learning step  记录学习次数 (用于判断是否更换 target_net 参数)
        self.learn_step_counter = 0

        # initialize zero memory [s, a, r, s_]   初始化全 0 记忆 [s, a, r, s_]
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))

        # consist of [target_net, evaluate_net] 创建 [target_net, evaluate_net]神经网络
        self._build_net()
        
        # 替换 target net 的参数
        # tf.get_collection(key,scope=None)返回具有该给定名称的集合中的值列表
        t_params = tf.get_collection('target_net_params')  # 提取 target_net 的参数
        e_params = tf.get_collection('eval_net_params')    # 提取  eval_net 的参数
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)] # 更新 target_net 参数

        self.sess = tf.Session()

        if output_graph:
            # $ tensorboard --logdir=logs
            # tf.train.SummaryWriter soon be deprecated, use following
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())
        self.cost_his = [] # 定义空列表方便后面存储cost

    def _build_net(self): # 搭建网络 
        # ------------------ build evaluate_net ------------------预测网络具备最新参数，最后输出q_evaluate
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # 用来接收 observation，input当前的状态，作为NN的输入
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target')  # for calculating loss 用来接收 q_target 的值
        with tf.variable_scope('eval_net'): 
            # variblae（）构造函数或者get_variable自动将变量添加到图形集合：GraphKeys.GLOBAL_VARIABLES（默认）
            # c_names(collections_names) are the collections to store variables  是在更新 target_net 参数时会用到
            # 首先对图层进行配置，w，b初始化，第一层网络的神经元数位n_l1
            c_names, n_l1, w_initializer, b_initializer = \
                ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], 32, \
                tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)  # config of layers   
            # c_names作为一个存储变量的集合，其名称为eval_net_params，将q估计神经网络的参数都放入这个集合当中

            # first layer. collections is used later when assign to target net 第一层
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s, w1) + b1) # tf.nn.relu()代表激活函数，python有广播功能，第一层输出l1，维度[None,n_l1]

            # second layer. collections is used later when assign to target net 第二层
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.q_eval = tf.matmul(l1, w2) + b2 # 第二层输出q估计，维度[None,self.n_actions]

        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval)) # 基于q估计和q现实，构造loss_function
        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)       # 梯度下降，进行训练

        # ------------------ build target_net ------------------目标值网络的参数是之前的，最后输出q_next
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')    # input， 接收下个 observation，s_表示下一个状态
        with tf.variable_scope('target_net'):
            # c_names(collections_names) are the collections to store variables
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

            # first layer. collections is used later when assign to target net
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s_, w1) + b1)

            # second layer. collections is used later when assign to target net
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.q_next = tf.matmul(l1, w2) + b2

    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0

        transition = np.hstack((s, [a, r], s_))

        # replace the old memory with new memory 总 memory 大小是固定的, 如果超出总大小, 取index为余数，旧 memory 就被新 memory 替换
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition

        self.memory_counter += 1

    def choose_action(self, observation, train=True):
        # to have batch dimension when feed into tf placeholder
        observation = observation[np.newaxis, :]

        if np.random.uniform() < self.epsilon or (train == False): 
            # 当train=True时，仅百分之九十概率选择最大的，而当train=False，则直接q值选择最大的动作
            # forward feed the observation and get q value for every actions
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def learn(self): # 如何学习，更新参数的，设计target_net和eval_net的交互使用
        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0: # 提前检测是否替换target_net参数
            self.sess.run(self.replace_target_op)
            # print('\ntarget_params_replaced\n')

        # sample batch memory from all memory 调用记忆库中的记忆，从memory中随机抽取batch_size这么多记忆
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]
        
        # 获取 q_next (target_net 产生的 q) 和 q_eval(eval_net 产生的 q)
        q_next, q_eval = self.sess.run(
            [self.q_next, self.q_eval],
            feed_dict={
                self.s_: batch_memory[:, -self.n_features:],  # fixed params
                self.s: batch_memory[:, :self.n_features],  # newest params
            })

        # change q_target w.r.t q_eval's action
        q_target = q_eval.copy()

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = batch_memory[:, self.n_features + 1]

        q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1) # q_target是算法中的yj

        """
        For example in this batch I have 2 samples and 3 actions:
        q_eval =
        [[1, 2, 3],
         [4, 5, 6]]

        q_target = q_eval =
        [[1, 2, 3],
         [4, 5, 6]]

        Then change q_target with the real q_target value w.r.t the q_eval's action.
        For example in:
            sample 0, I took action 0, and the max q_target value is -1;
            sample 1, I took action 2, and the max q_target value is -2:
        q_target =
        [[-1, 2, 3],
         [4, 5, -2]]

        So the (q_target - q_eval) becomes:
        [[(-1)-(1), 0, 0],
         [0, 0, (-2)-(6)]]

        We then backpropagate this error w.r.t the corresponding action to network,
        leave other action as error=0 cause we didn't choose it.
        """

        # train eval network  训练 eval_net，这里self.cost 即 self.loss的结果
        _, self.cost = self.sess.run([self._train_op, self.loss],
                                     feed_dict={self.s: batch_memory[:, :self.n_features],
                                                self.q_target: q_target}) 
                                                # 这里的两个输入对应了搭建网络时需要的两个输入
                                                # 一个是输入的状态s得到q_eval,另一个输入是计算出的q_targrt
                                                # 利用上面两个输入计算loss并更新evla_net
        self.cost_his.append(self.cost)
        #loss = self.cost_his

        # increasing epsilon
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1
        
        

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Loss')
        plt.xlabel('training steps')
        plt.title('Loss')
        plt.legend()
        plt.savefig('Loss function')
        plt.show()



