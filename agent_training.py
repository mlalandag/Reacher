
from unityagents import UnityEnvironment
import numpy as np
import torch
import matplotlib.pyplot as plt
from agent import Agent

num_agents = 1

# please do not modify the line below
env = UnityEnvironment(file_name="Reacher_Windows_x86_64/Reacher.exe")

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]


# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents in the environment
print('Number of agents:', len(env_info.agents))

# number of actions
action_size = brain.vector_action_space_size
print('Number of actions:', action_size)

# examine the state space 
state = env_info.vector_observations[0]
print('States look like:', state)
state_size = len(state)
print('States have length:', state_size)


max_num_episodes = 5000

episode_scores = []

agent = Agent(state_size, action_size, 1)

print("loop over episodes")

for episode in range(1, max_num_episodes+1):
     
    env_info = env.reset(train_mode=False)[brain_name]     # reset the environment
    state  = env_info.vector_observations[0]               # get the current state
    score = 0                                              # initialize the score

    #for step in range(1000):
    while True:

        action = agent.act(state)                          # select an action (for each agent)
        action = np.clip(action, -1, 1)                   # all actions between -1 and 1
        env_info = env.step(action)[brain_name]            # send all actions to tne environment
        next_state = env_info.vector_observations[0]       # get next state (for each agent)
        reward = env_info.rewards[0]                       # get reward (for each agent)
        done = env_info.local_done[0]                      # see if episode finished

        agent.step(state, action, reward, next_state, done)

        score += reward                                    # update the score (for each agent)
        state = next_state                                 # roll over states to next time step
        if np.any(done):                                   # exit loop if episode finished
            break


    episode_scores.append(score)

    mean_last_100 = np.mean(episode_scores[episode - 100:])

    print('Total score for episode {} : {:.2f} - Mean last 100 episodes {:.2f}'.format(episode, score, mean_last_100))

    if mean_last_100 >= 30:
        # Save weights
        torch.save(agent.actor_local.state_dict(), 'actor_weights.pth')
        torch.save(agent.critic_local.state_dict(), 'critic_weights.pth')
        break

#Plot scores and save to image file
graph = plt.figure()
plt.plot([score for score in range(len(episode_scores))], episode_scores)
plt.ylabel('scores')
plt.xlabel('episodes')
plt.show()
graph.savefig('scores.jpg')
        
#When finished, you can close the environment.
env.close()
        
