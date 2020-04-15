import numpy as np
from keras.models import Sequential,load_model
from keras.layers import Dense,Conv2D,Flatten,Dropout
from keras.optimizers import Adam
import gym
from collections import deque
import cv2
import random
import os


class Agent:
    def __init__(self):
        self.n_actions=6
        self.batch_size=32
        self.gamma=0.95
        self.epsilon=1.0
        self.epsilon_decay_linear=1e-5
        self.learning_rate=0.0001
        self.eps_min=0.02
        self.model=self.build_model(self.n_actions,(4,80,80))
        self.target_model=self.build_model(self.n_actions,(4,80,80))
        self.experience_memory=deque(maxlen=25000)
        self.n_times_model_updated=0
    def build_model(self,n_actions,input_dims):
        model=Sequential()
        model.add(Conv2D(filters=32,input_shape=(*input_dims,),kernel_size=8,strides=4,activation='relu',data_format="channels_first"))
        model.add(Conv2D(filters=64,kernel_size=4,strides=2,activation='relu',data_format="channels_first"))
        model.add(Conv2D(filters=64,kernel_size=3,strides=1,activation='relu',data_format="channels_first"))
        model.add(Flatten())
        model.add(Dense(512,activation='relu'))
        model.add(Dense(n_actions))
        model.compile(loss='mean_squared_error',optimizer=Adam(lr=self.learning_rate))
        return model
    def memorize(self,state,action,reward,next_state,done):
        self.experience_memory.append((state,action,reward,next_state,done))
    def get_move(self,cur_state):
        if np.random.rand()<self.epsilon:
            return np.random.randint(0,6)
        quality_value=self.model.predict(np.expand_dims(cur_state,axis=0))
        return np.argmax(quality_value)

    def random_experiences(self,batch_size):
        random_idxs=np.random.randint(0,len(self.experience_memory),self.batch_size)
        cur_state,action,reward,next_state,done=[],[],[],[],[]
        for idx in random_idxs:
            cur_state.append(self.experience_memory[idx][0])
            action.append(self.experience_memory[idx][1])
            reward.append(self.experience_memory[idx][2])
            next_state.append(self.experience_memory[idx][3])
            done.append(self.experience_memory[idx][4])
        return np.array(cur_state),np.array(action),np.array(reward),np.array(next_state),np.array(done)



    def replace_target_network(self):
        if self.n_times_model_updated % 1000==0:
            self.target_model.set_weights(self.model.get_weights())


    def train_on_experiences(self):
        if len(self.experience_memory)<self.batch_size:
            return

        else:
            idxs=range(32)
            cur_state,action,reward,new_state,done=self.random_experiences(self.batch_size)
            self.replace_target_network()
            q_cur=self.model.predict(cur_state)
            q_next=self.target_model.predict(new_state)

            q_next[done]=0.0
            q_target=q_cur.copy()
            q_target[idxs,action]=reward+self.gamma*np.max(q_next,axis=1)
            self.model.train_on_batch(cur_state,q_target)
            self.epsilon=self.epsilon-self.epsilon_decay_linear if self.epsilon >self.eps_min else self.eps_min
            self.n_times_model_updated+=1
    def save_models(self):
        self.model.save('model.h5')
        self.target_model.save('target_model.h5')
    def load_model(self):
        self.model=load_model('model.h5')
        self.target_model=load_model('target_model.h5')
