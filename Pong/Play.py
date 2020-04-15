import numpy as np
from utils import make_env
from keras.models import load_model
import time

env=make_env('PongNoFrameskip-v4')

epsilon=0.02
action_space=[i for i in range(6)]

model=load_model('q_val.h5')
def choose_action(observation):
    if np.random.random()<epsilon:
        action=np.random.choice(action_space)
    else:
        state=np.array([observation],copy=False,dtype=np.float32)
        actions=model.predict(state)
        action=np.argmax(actions)
    return action


while True:
    state=env.reset()
    done=False
    while not done:
        time.sleep(0.01)
        env.render()
        action=choose_action(state)
        next_state,reward,done,info=env.step(action)
        state=next_state
