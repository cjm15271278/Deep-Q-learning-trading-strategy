import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.compat.v1 as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# from maze_env import Maze
#from stock_env_zero import stock
#from RL_brain import DeepQNetwork
import iisignature
np.random.seed(0) 

tf.disable_v2_behavior()
np.random.seed(1)
tf.set_random_seed(1)


# Deep Q Network off-policy
class test_DeepQNetwork:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate,  # =0.01,
            reward_decay,  # =0.9,
            e_greedy,  # =0.9,  # 百分之九十选择q值最大的动作
            replace_target_iter,  # =300,
            memory_size,  # =500,
            batch_size,  # =32,
            e_greedy_increment=None,  # 缩小随机的范围
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
        e_params = tf.get_collection('eval_net_params')  # 提取  eval_net 的参数
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]  # 更新 target_net 参数

        self.sess = tf.Session()

        self.sess.run(tf.global_variables_initializer())
        self.cost_his = []  # 定义空列表方便后面存储cost

        #调用保存好的模型参数
        saver = tf.train.Saver()
        #saver.restore(self.sess, "Best_Model/model" + ".ckpt")
        saver.restore(self.sess, "Model/model" + ".ckpt")

    def _build_net(self):  # 搭建网络
        # ------------------ build evaluate_net ------------------预测网络具备最新参数，最后输出q_evaluate
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # 用来接收 observation，input当前的状态，作为NN的输入
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions],
                                       name='Q_target')  # for calculating loss 用来接收 q_target 的值
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
                l1 = tf.nn.relu(tf.matmul(self.s, w1) + b1)  # tf.nn.relu()代表激活函数，python有广播功能，第一层输出l1，维度[None,n_l1]

            # second layer. collections is used later when assign to target net 第二层
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.q_eval = tf.matmul(l1, w2) + b2  # 第二层输出q估计，维度[None,self.n_actions]

        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))  # 基于q估计和q现实，构造loss_function
        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)  # 梯度下降，进行训练

        # ------------------ build target_net ------------------目标值网络的参数是之前的，最后输出q_next
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')  # input， 接收下个 observation，s_表示下一个状态
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




#if __name__ == "__main__":
#    df = generate_data(1000, 10000)[1]
#    df = pd.Series(df)
#    trend = df.values.tolist()  # 选取收盘数据做测试
#    env = stock(trend[0:9000])
    #调用模型，加载网络参数
#    RL = test(env.n_actions, env.n_features,
#                      learning_rate=0.001,
#                      reward_decay=0.99,
#                      e_greedy=0.9,
#                      replace_target_iter=200,
#                      batch_size=200,
#                      memory_size=4000,
                      # output_graph=True
#                      )
#    name ='lr=0.001, gamma=0.99, bs=200, ms=4000'
    #从加载到的模型中读取训练参数
#    run()
#    env = BackTest(env)
#    env.draw(name)



#df = generate_data(1000 ,10000)[1]
#df = pd.Series(df)
#test = df.values.tolist()
#env = stock(test)
#env = BackTest(env)
#name ='lr=0.001, gamma=0.95, bs=100, ms=3000'
#env.draw(name)
