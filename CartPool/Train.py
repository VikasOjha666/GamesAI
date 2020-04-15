#Trains a Deep Q Agent to play the cartpool.
import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Conv2D,Flatten,Dropout
from keras.optimizers import Adam
import gym
from collections import deque
import random


#Parameters.
learning_rate=0.001
n_episodes=100000
training=True
n_played=0
batch_size=32



class Agent:
    def __init__(self,obs,n_actions):
        self.n_actions=n_actions
        self.model=self.build_model(obs,n_actions)
        self.experience_memory=deque(maxlen=5000)
        self.epsilon=0
        self.epsilon_decay_linear=1/75
        self.gamma=0.9

    def build_model(self,obs,n_actions):
        model=Sequential()
        model.add(Dense(24,input_shape=(obs,),activation='relu'))
        model.add(Dense(24,activation='relu'))
        model.add(Dense(n_actions,activation='linear'))
        model.compile(loss='mse',optimizer=Adam(lr=learning_rate))
        return model
    def memorize(self,state,action,reward,next_state,done):
        self.experience_memory.append((state,action,reward,next_state,done))
    def get_move(self,cur_state):
        if np.random.rand()<self.epsilon:
            return random.randrange(self.n_actions)
        quality_val=self.model.predict(cur_state)
        return np.argmax(quality_val[0])
    def get_reward(self,r,done):
        if not done:
            return r
        else:
            return -r

    def train_on_experiences(self):
        if len(self.experience_memory)<batch_size:
            return
        else:
            minibatch=random.sample(self.experience_memory,batch_size)

            for cur_state,action,reward,next_state,done in minibatch:
                updated_target=reward
                if not done:
                    updated_target=updated_target+self.gamma*np.amax(self.model.predict(next_state)[0])
                updated_target_b=self.model.predict(cur_state)
                updated_target_b[0][action]=updated_target
                self.model.fit(cur_state,updated_target_b,verbose=0)
            self.epsilon*=self.epsilon_decay_linear
            self.epsilon=max(0.01,self.epsilon)


#Setup environment and make the Agent object.
env=gym.make("CartPole-v1")
DQNAgent=Agent(env.observation_space.shape[0],env.action_space.n)

#Lets play and train.



while n_played<n_episodes:
        n_played+=1
        old_state=env.reset()
        old_state=np.reshape(old_state,[1,env.observation_space.shape[0]])
        score=0
        while True:
            score+=1
            env.render()
            action=DQNAgent.get_move(old_state)
            next_state,reward,done,info=env.step(action)
            reward=DQNAgent.get_reward(reward,done)
            next_state=np.reshape(next_state,[1,env.observation_space.shape[0]])
            DQNAgent.memorize(old_state,action,reward,next_state,done)
            old_state=next_state
            if done:
                print("No of times the game was played:"+str(n_played),'score:'+str(score))
                break
        DQNAgent.train_on_experiences()

            if n_played%500:
                DQNAgent.model.save_weights('weights.hdf5')
