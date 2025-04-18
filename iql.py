import random
import numpy as np
from tqdm import tqdm

def train(
        env, 
        params:dict, 
        qtable, 
        actions_dict:dict, 
        action_dict:dict, 
        reward_dict:dict,
        obs_dict:dict,
        obs_action_dict:dict, 
        train_episodes:int, 
        train_log_every, 
        alpha:float, 
        gamma:float, 
        decay_type:str,
        decay:float,
        epsilon:float,
        epsilon_min:float,
        logger,
        visualizer=None
    ):
    """
    Train agent employing Q-learning algorithm
    """
    print("Start training...\n")
    
    n_actions = env.actions_n()
    old_s = {} 
    old_a = {}

    # Restricted actions
    #actions_has_food = [3, 7]
    #actions_looking_for_food = [0, 2]

    ticks_per_episode = {}

    for ep in tqdm(range(1, train_episodes + 1), desc="EPISODES", colour='red', position=0, leave=False):
        env.reset()
        ticks_per_episode[str(ep)] = {}
        for tick in tqdm(range(1, params['episode_ticks'] + 1), desc="TICKS", colour='green', position=1, leave=False):
            for agent in env.agent_iter(max_iter=params['learner_population']):

                if env.all_learners_done:
                    break

                cur_state, reward, termination, truncation, _ = env.last(agent)

                agent_has_food = cur_state[0]
                patch_has_food = cur_state[1]
                agent_in_nest = cur_state[2]

                if not (termination or truncation):
                    if agent_has_food == 1:
                        if agent_in_nest == 1:
                            obs_dict[str(ep)]["has-food-in-nest"] += 1
                        else:
                            obs_dict[str(ep)]["has-food-out-nest"] += 1
                    else:
                        if patch_has_food == 1:
                            if agent_in_nest == 1:
                                obs_dict[str(ep)]["food-available-in-nest"] += 1
                            else:
                                obs_dict[str(ep)]["food-available-out-nest"] += 1
                        else:
                            obs_dict[str(ep)]["food-not-available"] += 1
                

                cur_s = env.convert_observation(cur_state)

                # Restricted behaviour starts here
                # if ep == 1 and tick == 1:
                #     action = actions_looking_for_food[np.random.randint(0, len(actions_looking_for_food))]
                # else:
                #     old_value = qtable[int(agent), old_s[agent], old_a[agent]]
                #     if agent_has_food == 1:
                #         if agent_in_nest:
                #             action = 5 # drop food
                #             new_value = (1 - alpha) * old_value + alpha * (reward + gamma * qtable[int(agent)][cur_s][action])
                #             qtable[int(agent), old_s[agent], old_a[agent]] = new_value
                #         else:
                #             next_max = np.max(qtable[int(agent)][cur_s][actions_has_food])
                #             new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
                #             qtable[int(agent), old_s[agent], old_a[agent]] = new_value
                #             if random.uniform(0, 1) < epsilon:
                #                 action = actions_has_food[np.random.randint(0, len(actions_has_food))]
                #             else:
                #                 action = actions_has_food[np.argmax(qtable[int(agent)][cur_s][actions_has_food])]
                #             # action = 7 # lay food pheromone and head back to nest
                #     else:
                #         if patch_has_food == 1 and agent_in_nest == 0:
                #             action = 4 # take food
                #             new_value = (1 - alpha) * old_value + alpha * (reward + gamma * qtable[int(agent)][cur_s][action])
                #             qtable[int(agent), old_s[agent], old_a[agent]] = new_value
                #         else:
                #             next_max = np.max(qtable[int(agent)][cur_s][actions_looking_for_food])
                #             new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
                #             qtable[int(agent), old_s[agent], old_a[agent]] = new_value
                #             if random.uniform(0, 1) < epsilon:
                #                 action = actions_looking_for_food[np.random.randint(0, len(actions_looking_for_food))]
                #             else:
                #                 action = actions_looking_for_food[np.argmax(qtable[int(agent)][cur_s][actions_looking_for_food])]
                #             # action = 2 # follow food pheromone 
                # Restricted behaviour ends here
                    
                if ep == 1 and tick == 1:
                    action = np.random.randint(0, n_actions)
                else:
                    old_value = qtable[int(agent), old_s[agent], old_a[agent]]
                    next_max = np.max(qtable[int(agent), cur_s])
                    new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
                    qtable[int(agent), old_s[agent], old_a[agent]] = new_value
                    if random.uniform(0, 1) < epsilon:
                        action = np.random.randint(0, n_actions)
                    else:
                        action = np.argmax(qtable[int(agent)][cur_s])

                if not (termination or truncation):
                    if agent_has_food == 1:
                        if agent_in_nest == 1:
                            obs_action_dict[str(ep)]["has-food-in-nest"][str(action)] += 1
                        else:
                            obs_action_dict[str(ep)]["has-food-out-nest"][str(action)] += 1
                    else:
                        if patch_has_food == 1:
                            if agent_in_nest == 1:
                                obs_action_dict[str(ep)]["food-available-in-nest"][str(action)] += 1
                            else:
                                obs_action_dict[str(ep)]["food-available-out-nest"][str(action)] += 1
                        else:
                            obs_action_dict[str(ep)]["food-not-available"][str(action)] += 1

                    old_s[agent] = cur_s
                    old_a[agent] = action

                    actions_dict[str(ep)][str(action)] += 1
                    action_dict[str(ep)][str(agent)][str(action)] += 1

                if termination or truncation:
                    action = None

                env.step(action)

                if str(agent) in ticks_per_episode[str(ep)].keys():
                    ticks_per_episode[str(ep)][str(agent)] += 1
                else:
                    ticks_per_episode[str(ep)][str(agent)] = 1

                reward_dict[str(ep)][str(agent)] += round(reward, 2)

            if env.all_learners_done:
                    break

            if visualizer != None:
                visualizer.render(
                    env.patches,
                    env.learners,
                    env.turtles
                )
        
        if decay_type == "log":
            epsilon = max(epsilon * decay, epsilon_min)
        elif decay_type == "linear":
            epsilon = max(epsilon - (1 - decay), epsilon_min)
        
        if ep % train_log_every == 0:
            avg_rew = round((sum({agent: rew/ticks_per_episode[str(ep)][str(agent)] for agent, rew in reward_dict[str(ep)].items()}.values()) / params["learner_population"]), 2)
            eps = round(epsilon, 4)
            value = [ep, tick * ep, avg_rew]
            value.extend(list(actions_dict[str(ep)].values()))
            for s in env.states:
                value.append(obs_dict[str(ep)][s])
                for i, a in enumerate(env.actions):
                    value.append(obs_action_dict[str(ep)][s][str(i)])
            value.append(eps)
            logger.load_value(value)

    logger.empty_table()
    env.close()
    if visualizer != None:
        visualizer.close()
    print("Training finished!\n")

    return qtable

def eval(
        env,
        params:dict,
        actions_dict,
        action_dict,
        reward_dict,
        obs_dict,
        obs_action_dict,
        situation_action_agent_dict,
        test_episodes:int,
        qtable,
        test_log_every:int,
        logger,
        visualizer=None
    ):
    """
    DOC Evaluate agent's performance after Q-learning
    """

    print("Start testing...\n")

    # Restricted actions
    #actions_has_food = [3, 7]
    #actions_looking_for_food = [0, 2]

    ticks_per_episode = {}
    
    for ep in tqdm(range(1, test_episodes + 1), desc="EPISODES", colour='red', leave=False):
        env.reset()
        ticks_per_episode[str(ep)] = {}
        for tick in tqdm(range(1, params['episode_ticks'] + 1), desc="TICKS", colour='green', leave=False):
            for agent in env.agent_iter(max_iter=params['learner_population']):

                if env.all_learners_done:
                    break

                state, reward, termination, truncation, _ = env.last(agent)

                if termination or truncation:
                    action = None
                else:

                    agent_has_food = state[0]
                    patch_has_food = state[1]
                    agent_in_nest = state[2]

                    if agent_has_food == 1:
                        if agent_in_nest == 1:
                            obs_dict[str(ep)]["has-food-in-nest"] += 1
                        else:
                            obs_dict[str(ep)]["has-food-out-nest"] += 1
                    else:
                        if patch_has_food == 1:
                            if agent_in_nest == 1:
                                obs_dict[str(ep)]["food-available-in-nest"] += 1
                            else:
                                obs_dict[str(ep)]["food-available-out-nest"] += 1
                        else:
                            obs_dict[str(ep)]["food-not-available"] += 1

                    s = env.convert_observation(state)
                    action = np.argmax(qtable[int(agent)][s])

                    # Restricted behavior starts here
                    # if agent_has_food == 1:
                    #     if agent_in_nest:
                    #         action = 5 # drop food
                    #     else:
                    #         action = actions_has_food[np.argmax(qtable[int(agent)][s][actions_has_food])]
                    #         #action = 7 # lay food pheromone and head back to nest
                    # else:
                    #     if patch_has_food == 1 and agent_in_nest == 0:
                    #         action = 4 # take food
                    #     else:
                    #         action = actions_looking_for_food[np.argmax(qtable[int(agent)][s][actions_looking_for_food])]
                    #         #action = 2 # follow food pheromone
                    # Restricted behavior ends here

                    if agent_has_food == 1:
                        if agent_in_nest == 1:
                            obs_action_dict[str(ep)]["has-food-in-nest"][str(action)] += 1
                            situation_action_agent_dict[str(ep)][str(agent)]["has-food-in-nest"][str(action)] += 1
                        else:
                            obs_action_dict[str(ep)]["has-food-out-nest"][str(action)] += 1
                            situation_action_agent_dict[str(ep)][str(agent)]["has-food-out-nest"][str(action)] += 1
                    else:
                        if patch_has_food == 1:
                            if agent_in_nest == 1:
                                obs_action_dict[str(ep)]["food-available-in-nest"][str(action)] += 1
                                situation_action_agent_dict[str(ep)][str(agent)]["food-available-in-nest"][str(action)] += 1
                            else:
                                obs_action_dict[str(ep)]["food-available-out-nest"][str(action)] += 1
                                situation_action_agent_dict[str(ep)][str(agent)]["food-available-out-nest"][str(action)] += 1
                        else:
                            obs_action_dict[str(ep)]["food-not-available"][str(action)] += 1
                            situation_action_agent_dict[str(ep)][str(agent)]["food-not-available"][str(action)] += 1

                    actions_dict[str(ep)][str(action)] += 1
                    action_dict[str(ep)][str(agent)][str(action)] += 1

                env.step(action)
                
                if str(agent) in ticks_per_episode[str(ep)].keys():
                    ticks_per_episode[str(ep)][str(agent)] += 1
                else:
                    ticks_per_episode[str(ep)][str(agent)] = 1

                reward_dict[str(ep)][str(agent)] += round(reward, 2)
            
            if env.all_learners_done:
                break

            if visualizer != None:
                visualizer.render(
                    env.patches,
                    env.learners,
                    env.turtles
                )
        
        if ep % test_log_every == 0:
            avg_rew = round((sum({agent: rew/ticks_per_episode[str(ep)][str(agent)] for agent, rew in reward_dict[str(ep)].items()}.values()) / params["learner_population"]), 2)

            value = [ep, tick * ep, avg_rew]
            value.extend(list(actions_dict[str(ep)].values()))
            for s in env.states:
                value.append(obs_dict[str(ep)][s])
                for i, a in enumerate(env.actions):
                    value.append(obs_action_dict[str(ep)][s][str(i)])

            logger.load_value(value)
    
    logger.save_pickle(situation_action_agent_dict)
    logger.empty_table()
    env.close()
    if visualizer != None:
        visualizer.close()
    print("Testing finished!")