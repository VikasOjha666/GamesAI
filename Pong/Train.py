import numpy as np
from Train import Agent
from utils import  make_env

if __name__=='__main__':
    env=make_env('PongNoFrameskip-v4')
    num_games=250
    load_checkpoint=False
    best_score=-21

    agent=Agent()
    if load_checkpoint:
        agent.load_models()

    scores,eps_history=[],[]
    n_steps=0

    for i in range(num_games):
        score=0
        observation=env.reset()
        done=False
        while not done:
            action=agent.get_move(observation)
            observation_,reward,done,info=env.step(action)
            n_steps+=1
            score+=reward
            if not load_checkpoint:
                agent.memorize(observation,action,reward,observation_,int(done))
                agent.train_on_experiences()
            else:
                env.render()
            observation=observation_
        scores.append(score)
        avg_score=np.mean(scores[-100:])
        print('episode',i,'score ',score,
        'average score %.2f'%avg_score,'epsilon %.2f'%agent.epsilon,
        'steps',n_steps)

        if avg_score>best_score:
            agent.save_models()
            print('avg score %.2f better than best_score %.2f' % (avg_score,best_score))
            best_score=avg_score
