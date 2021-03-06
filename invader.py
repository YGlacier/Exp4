import gym
import numpy as np
import model
import chainer
import chainerrl
import os.path
from chainer import cuda

env = gym.make('SpaceInvaders-ram-v0')
observation=env.reset()
total_reward=0

output_size=env.action_space.n
input_size=env.observation_space.shape[0]

q_func_name="QFunction/qfunc.model"
q_func=model.QFunction(input_size,output_size,256)
#cuda.get_device(0).use()
#q_func.to_gpu(0)

if(os.path.isfile(q_func_name)):
    q_func.load_model(q_func_name)
optimizer=chainer.optimizers.Adam(eps=1e-2)
optimizer.setup(q_func)

gamma = 0.99
explorer=chainerrl.explorers.ConstantEpsilonGreedy(epsilon=0.3,random_action_func=env.action_space.sample)
replay_buffer=chainerrl.replay_buffer.ReplayBuffer(capacity=10**6)
phi=lambda x:x.astype(np.float32,copy=False)

agent=chainerrl.agents.DQN(q_func,optimizer,replay_buffer,gamma,explorer,replay_start_size=500,phi=phi)

epoch_num=100
epoch_max_length=100000
reward_list_name="data/reward.txt"
if(os.path.isfile(reward_list_name)):
    reward_list=np.loadtxt(reward_list_name)
else:
    reward_list=np.array([])
agent_path="agent"
#print(os.path.isfile(agent_path+"/model.npz"))
if(os.path.isfile(agent_path+"/model.npz")):
    agent.load(agent_path)
epoch_log_name="data/epoch.txt"
print(os.path.isfile(epoch_log_name))
if(os.path.isfile(epoch_log_name)):
    total_epoch=np.loadtxt(epoch_log_name)
    print(total_epoch)
else:
    total_epoch=0

agent.gamma=0.99
explorer.epsilon=0.5
agent.episodic_update=True
death_penalty=200

#for epoch in range(epoch_num):
while True:
    observation=env.reset()
    reward=0
    total_reward=0
    done=False
    life=3
    for _ in range(epoch_max_length):
        #env.render()
        action=agent.act_and_train(observation,reward)
        observation,reward,done,info=env.step(action)
        total_reward+=reward
        if info["ale.lives"] < life:
            reward-=death_penalty
            life=info["ale.lives"]
        if done:
            break
    total_epoch+=1
    agent.stop_episode_and_train(observation,reward,done)
    q_func.save_model(q_func_name)
    agent.save(agent_path)
    reward_list=np.append(reward_list,[total_reward])
    np.savetxt(reward_list_name,reward_list)
    np.savetxt(epoch_log_name,[total_epoch])
    print("Epoch:",total_epoch," Reward=", total_reward)

env.close()

#for _ in range(10000):
#    env.render()
#    action=env.action_space.sample()
#    observation,reward,done,info=env.step(action)
#    total_reward+=reward
#    if done:
#        break
#env.close()
#print total_reward
