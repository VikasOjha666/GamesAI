import gym
from keras.models import Sequential
from keras.layers import Dense,Conv2D,Flatten,Dropout
from keras.optimizers import Adam
import numpy as np

#Setup environment and make the Agent object.
env=gym.make("CartPole-v1")



model=Sequential()
model.add(Dense(24,input_shape=(env.observation_space.shape[0],),activation='relu'))
model.add(Dense(24,activation='relu'))
model.add(Dense(env.action_space.n,activation='linear'))
model.compile(loss='mse',optimizer=Adam(lr=0.001))


model.load_weights('weights.hdf5')



done=False
while True:
    old_state=env.reset()
    old_state=np.reshape(old_state,[1,env.observation_space.shape[0]])
    score=0
    while not done:
        score+=1
        env.render()
        quality_val=model.predict(old_state)
        action=np.argmax(quality_val[0])
        next_state,reward,done,info=env.step(action)
        old_state=next_state
        if done:
            print('Score:'+str(score))
