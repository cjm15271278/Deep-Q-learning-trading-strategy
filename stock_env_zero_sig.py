# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 15:16:28 2020

@author: hcb
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as patches
import iisignature
from sklearn import preprocessing

np.random.seed(1)

class stock:
    
    def __init__(self, trend, init_money=10000, window_size=20): #这里把df改为trend
        
        self.n_actions = 3 # 动作数量
        self.n_features = window_size  +  iisignature.siglength(2,3) # 特征数量   
        #self.trend = df['close'].values # 收盘数据
        #self.df = df #数据的DataFrame
        self.trend = trend # 收盘数据
        self.init_money = init_money # 初始化资金
        self.n_lagged_time_steps = 20  #用于计算signature的p-lag value 200
        
        
        self.window_size = window_size #滑动窗口大小
        self.half_window = window_size // 2
        #self.signature = np.empty(shape=[len(self.trend),iisignature.siglength(2,3)])
        self.signature = self.signature_normalisation(self.trend)
        
        #self.buy_rate = 0.0003  # 买入费率
        #self.buy_min = 5   # 最小买入费率
        #self.sell_rate = 0.0003  # 卖出费率
        #self.sell_min = 5  # 最大买入费率
        #self.stamp_duty = 0.001  # 印花税
        
    def reset(self):
        self.hold_money = self.init_money # 持有资金
        self.buy_num = 0 # 买入数量
        self.sell_num=0 # 卖出数量
        self.hold_num = 0 # 持有股票数量（充当current_inventory的作用，存储仓位变化情况）
        self.stock_value = 0 # 持有股票总市值
        self.maket_value = 0 # 总市值（加上现金）
        self.last_value = self.init_money # 上一天市值
        self.total_profit = 0 # 总盈利
        self.t = self.window_size // 2 # 时间
        #self.t = 0 #时间
        self.reward = 0 # 收益
        self.n = 100    # reward窗口
        self.min_buy =100
        self.min_sell=100
        #self.current_inventory = 0 #现有库存
        #self.inventory = []
        
        self.states_sell = [] #卖股票时间
        self.states_buy = [] #买股票时间
        
        self.profit_rate_account = [] # 账号盈利
        self.profit_rate_stock = [] # 股票波动情况
        self.daily_return = []
        
        #self.stream=[]
        return self.get_state(self.t)
        
        
    def leadlag(self, X):  #定义lead-lag transformation函数
    #Returns lead-lag-transformed stream of X

    #Arguments:
    #    X: list, whose elements are tuples of the form
    #    (time, value).

    #Returns:
    #    list of points on the plane, the lead-lag
    #    transformed stream of X
        l=[]
        for j in range(2*(len(X))-1):
            i1=j//2
            i2=j//2
            if j%2!=0:
                i1+=1
            l.append((X[i1][1], X[i2][1]))
        return l
    
    def timejoined(self, Y):  #定义time-joined transformation函数
        Y.append(Y[-1])
        l1=[] 
        
        for j in range(2*(len(Y))+1+2):
            if j==0:
                l1.append((Y[j][0], 0))
                continue
            for i in range(len(Y)-1):
                if j==2*i+1:
                    l1.append((Y[i][0], Y[i][1]))
                    break
                if j==2*i+2:
                    l1.append((Y[i+1][0], Y[i][1]))
                    break
        return l1
        
    def min_max_normalization(self, X):  # do the min-max nomalisation based on the time seires
        X_n = np.array([(float(i)-np.min(X))/float(np.max(X)-np.min(X)) for i in X])
        return X_n
        
    def construct_multi_feature_time_series(self, market_info): # compute the maxtrix of the p-lag value time series
        market_info = self.trend
        n_lagged_time_steps = self.n_lagged_time_steps
        X_n = self.min_max_normalization(self.trend)
        X_ts = np.zeros((len(self.trend), n_lagged_time_steps))
        for i in range(X_ts.shape[0]):
            d = i - n_lagged_time_steps
            if d  < 0:
                X_ts[i][:-d] = market_info[0]
                X_ts[i][-d:] = market_info[1:i+1]
            else:
                X_ts[i] = market_info[i - n_lagged_time_steps + 1: i+1] 
        return X_ts
        
    def signature_matrix(self, market_info ): # compute the signature matrix
        X_ts = self.construct_multi_feature_time_series(self.trend)
        sig_m = []
        stream = []
        for i in range(X_ts.shape[0]):
            stream = list(enumerate(X_ts[i]))
            path = self.leadlag(stream)  
            sig = iisignature.sig(path,3)   # change the dimension of signature
            sig = sig.tolist()
            sig_m.append(sig)
        return sig_m
        
    def signature_normalisation(self, market_info): # do the min_max normalisation on the signature matrix
        sig_matrix = self.signature_matrix(self.trend)
        min_max_scaler = preprocessing.MinMaxScaler()
        sig_min_max = min_max_scaler.fit_transform(sig_matrix)
        return sig_min_max
        
    
    def get_state(self, t): #某t时刻的状态
        
        window_size = self.window_size + 1
        d = t - window_size + 1
		#早期天数不够窗口打小，用0时刻来凑，即填补相应个数
        block = []
        if d<0:
            for i in range(-d):
                block.append(self.trend[0])
            for i in range(t+1):
                block.append(self.trend[i])
        else:
            block = self.trend[d : t + 1]
        #由上面的命令得到一个p-lag值=window_size的序列
        
        #由block该序列计算original_input
        original_input = []
        for i in range(window_size - 1):
            original_input.append((block[i + 1] - block[i])/(block[i]+0.0001)) #每步收益
        original_input = np.array(original_input)
        
        #由block该序列计算signature
        #stream = list(enumerate(block))
        #self.stream=stream
        #path = self.leadlag(self.stream)       # 使用lead-lag transformation
        #path = self.timejoined(self.stream)     # 使用time-joined transformation
        #signature = iisignature.sig(path,5)     # iisignature的结果为np.array
        
        #将original_input（array）与signature横向合并得到新的状态向量
        signature = self.signature_normalisation(self.trend)
        state = np.hstack((original_input, self.signature[t]))

        return state #作为状态编码
        #return original_input  #原来的状态输入
        #return signature       #新的状态输入
        
    def buy_stock(self):       
        # 买入股票
        #self.buy_num = self.hold_money // self.trend[self.t] // 100 # 买入手数
        #self.buy_num = self.buy_num * 100 #买入股票数量
        self.buy_num = self.min_buy
        
        # 计算手续费等
        tmp_money = self.trend[self.t] * self.buy_num #买入股票的总价格
        #service_change = tmp_money * self.buy_rate  #券商的佣金
        #if service_change < self.buy_min: #不足5元，则以5元收取
        #    service_change = self.buy_min
        
        # 如果手续费不够，就少买100股(这里更改了策略，不需要这样判断，但购买时的条件需要补充为hold_money>=手续费+买进的价格)
        #if service_change + tmp_money > self.hold_money: #买股票支出=股票总价格+手续费
        #    self.buy_num = self.buy_num - 100
        #tmp_money = self.trend[self.t] * self.buy_num  #由于可能因为余额无法支付手续费，而少买100股，这里股票价格进行更新
        #service_change = tmp_money * self.buy_rate     #券商的佣金=买入股票总价格*0.0003（万分之三）
        #if service_change < self.buy_min:              #同上，可能因为if语句的执行，进行手续费的更新,不足5元
        #    service_change = self.buy_min
         
        # 购买后，账户情况的更新   
        self.hold_num += self.buy_num                              #拥有的股票数量，仓位更新
        self.stock_value += self.trend[self.t] * self.hold_num      #股票的总价格
        self.hold_money = self.hold_money - self.trend[self.t] * self.buy_num  #续行符号 \,代表拥有的钱=原来的-单股价格*股票数量-手续费
        self.states_buy.append(self.t)                             #记录买入的天数
    
    def sell_stock(self): #,sell_num):
        #卖出股票
        self.sell_num = self.min_sell
        
        #计算手续费等
        tmp_money = self.sell_num * self.trend[self.t]  #卖出股票的总价格（这里sell-num在后面使用时会赋值为hold-num）
        #service_change = tmp_money * self.sell_rate     #券商的佣金=卖出股票的总价格*万分之三
        #if service_change < self.sell_min:              #若佣金小于5元，以5元收取
        #    service_change = self.sell_min
        #stamp_duty = self.stamp_duty * tmp_money        #印花税仅在卖出时收取，且=卖出股票的总价格*千分之一
        
        #卖出后，账号情况的更新
        self.hold_money = self.hold_money + tmp_money   #拥有的钱=原来的+卖出股票总价格-佣金-印花税
        self.hold_num -= self.sell_num                  #因为全部卖出和买入，故此时拥有股票数为0(这里更改了梭哈的策略，故不一定为0)
        self.stock_value -= self.trend[self.t] * self.sell_num      #跟上面一样，全部卖出后，股票价格为0
        self.states_sell.append(self.t)                 #记录卖出的天数


    
    def step(self, action, show_log=False):
        
                
        if action == 1 and self.hold_money >= self.trend[self.t]*100  and self.t < (len(self.trend) - self.half_window):
            #判断拥有的钱是否>购买的股票价格+max(5元，股票总价格*万分之三)
            buy_ = True
            #if my_trick and not self.trick(): 
                # 如果使用自己的触发器并不能出发买入条件，就不买
                #仅当确定自己要使用自己的触发器（训练时候不执行），且满足触发器结果为false时，即当日收盘价小于21日均线，有风险，不购买
            #    buy_ = False
            if buy_ : 
                self.buy_stock()
                self.reward= self.trend[self.t]/self.trend[self.t-self.n]

        elif action == 2 and self.hold_num > 0:
            # 卖出股票         
            self.sell_stock()
            self.reward = (2*self.trend[self.t-1]-self.trend[self.t])/self.trend[self.t-self.n]
        
        elif action == 0:
            self.reward = self.trend[self.t-1]/self.trend[self.t-self.n]
            
        #else:#如果模型发出不操作指令（即1，2之外的动作0），但收盘价小于21日均线，卖出。
        #    if my_trick and self.hold_num>0 and not self.trick():
        #        self.sell_stock(self.hold_num)
        #        if show_log:
        #            print(
        #                'day:%d, sell price:%f, total balance %f,'
        #                % (self.t, self.trend[self.t], self.hold_money)
        #            )
                    
        self.stock_value = self.trend[self.t] * self.hold_num #拥有的股票的总价格（用于下一部计算市场价值）
        self.maket_value = self.stock_value + self.hold_money #市场价值=股票总价格+剩下的拥有的钱
        self.total_profit = self.maket_value - self.init_money#总利润=市场价值-刚开始的钱
    
        #reward = (self.trend[self.t + 1] - self.trend[self.t]) / self.trend[self.t] #reward定义：（St+1-St）/St
        #if np.abs(reward)<=0.015:
        #    self.reward = reward * 0.2
        #elif np.abs(reward)<=0.03:
        #    self.reward = reward * 0.7
        #elif np.abs(reward)>=0.05:
        #    if reward < 0 :
        #        self.reward = (reward+0.05) * 0.1 - 0.05
        #    else:
        #        self.reward = (reward-0.05) * 0.1 + 0.05
        
        # reward = (self.trend[self.t + 1] - self.trend[self.t]) / self.trend[self.t]
        
        #对于股票的买入卖出或不操作等策略，都会影响reward。当股票卖出，r如果未来上涨了会给一个负向的reward，迫使它下次要买入；类似如果卖出，第二天下跌了就给一个正向的reward。如果股票不进行操作同样会给一个reward，但是这个reward比较小，因为我这边认为模型难以判断未来情况。
        #if self.hold_num > 0 or action == 2:                            
        #    self.reward = reward    
        #    if action == 2:
        #        self.reward = -self.reward
        #else:
        #    self.reward = -self.reward * 0.1
            # self.reward = 0
        
        self.profit_rate_account.append((self.maket_value - self.init_money) / self.init_money)  #获得账户利率
        self.profit_rate_stock.append((self.trend[self.t] - self.trend[0]) / self.trend[0])      #获得股票利率
        self.daily_return.append(self.maket_value)
        done = False
        self.t = self.t + 1
        if self.t == len(self.trend) - 2:
            done = True
        s_ = self.get_state(self.t)
        reward = self.reward
        
        return s_, reward, done
    
    def get_info(self):
        return self.states_sell, self.states_buy, self.profit_rate_account, self.profit_rate_stock, self.daily_return, self.total_profit 
    #获得信息函数：卖出天数、买入天数、账户利率、股票利率
    def draw(self,name):
        # 绘制结果
        states_sell, states_buy, profit_rate_account, profit_rate_stock, daily_return, total_profit = self.get_info()
        invest = profit_rate_account[-1]
        total_gains = self.total_profit
        close = self.trend
        fig = plt.figure(figsize = (15,5))
        plt.plot(close, color='r', lw=2.)
        plt.plot(close, 'v', markersize=8, color='k', label = 'selling signal', markevery = states_sell)
        plt.plot(close, '^', markersize=8, color='m', label = 'buying signal', markevery = states_buy)        
        plt.title('total gains %f, total investment %f%%'%(total_gains, invest))
        plt.legend()
        plt.savefig(name+'-trade.png')
        ###plt.close()
        
        #fig = plt.figure(figsize = (15,5))
        #plt.plot(profit_rate_account, label='my account')
        #plt.plot(profit_rate_stock, label='stock')
        #plt.legend()
        #plt.savefig(name+'-profit.png')
        ###plt.close()
        
        
        