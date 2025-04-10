from pettingzoo import AECEnv
from pettingzoo.utils.env import ObsType
from pettingzoo.utils import agent_selector
import pygame
import numpy as np
import random
from gymnasium.spaces import Discrete, Box
from scipy.ndimage import gaussian_filter
from tqdm import tqdm
import time
import sys
import math
from typing import Optional

class Ants(AECEnv):
    def observe(self, agent: str) -> ObsType:
        self.agent = self.agent_name_mapping[agent]
        #self.observations[agent] = self.process_agent()
        self.observations[agent] = self._get_obs2(self.learners[self.agent])
        return np.array(self.observations[agent])

    def observation_space(self, agent):
        return self._observation_spaces[agent]
    
    def action_space(self, agent):
        return self._action_spaces[agent]
      
    def observations_n(self, same_obs=True):
        #if same_obs:
        #    return self.observation_space('0').shape[0]
        if self.sniff_patches > 3:
            return pow(2, 9)
        else:
            return pow(2, 7)

    def actions_n(self, same_actions=True):
        if same_actions:
            return self.action_space('0').n.item()
    
    metadata = {"render_modes": ["human", "server"]}
    
    def __init__(self, seed, render_mode: Optional[str] = None, **kwargs):

        np.random.seed(seed)
        random.seed(seed)

        assert render_mode is None or render_mode in self.metadata["render_modes"]

        self.W = kwargs['W']
        self.H = kwargs['H']
        self.patch_size = kwargs['PATCH_SIZE']
        self.turtle_size = kwargs['TURTLE_SIZE']

        self.population = kwargs["population"]
        self.learner_population = kwargs['learner_population']

        self.sniff_threshold = kwargs['sniff_threshold']
        self.sniff_limit = kwargs['sniff_limit']
        #self.smell_area = kwargs['smell_area']
        self.follow_food_pheromone_mode = kwargs['follow_food_pheromone_mode']
        self.take_drop_mode = kwargs['take_drop_mode']
        
        self.lay_area = kwargs['lay_area']
        self.lay_amount = kwargs['lay_amount']
    
        self.diffuse_area = kwargs['diffuse_area']
        self.diffuse_mode = kwargs['diffuse_mode']

        self.evaporation = kwargs['evaporation']

        self.sniff_patches = kwargs['sniff_patches']
        self.wiggle_patches = kwargs['wiggle_patches'] 
        assert (
            self.sniff_patches in (1, 3, 5, 7, 8)
        ), "Error! sniff_patches admitted values are: 1, 3, 5, 7, 8."
        assert (
            self.wiggle_patches in (1, 3, 5, 7, 8)
        ), "Error! wiggle_patches admitted values are: 1, 3, 5, 7, 8."

        self.actions = kwargs['actions']
        self.states = kwargs['states']
        self.penalty = kwargs["penalty"]
        self.reward = kwargs["reward"]
        self.reward_type = kwargs["reward_type"]

        # Used to calculate the agent's directions.
        self.N_DIRS = 8
        self.movements = np.array([
            (0, -self.patch_size),                  # dir 0
            (self.patch_size, -self.patch_size),    # dir 1
            (self.patch_size, 0),                   # dir 2
            (self.patch_size, self.patch_size),     # dir 3
            (0, self.patch_size),                   # dir 4
            (-self.patch_size, self.patch_size),    # dir 5
            (-self.patch_size, 0),                  # dir 6
            (-self.patch_size, -self.patch_size),   # dir 7
        ])

        # List of the coordinates of the "center" of each patch, which also represent its identifier
        self.coords = []
        self.offset = self.patch_size // 2
        self.W_pixels = self.W * self.patch_size
        self.H_pixels = self.H * self.patch_size
        for x in range(self.offset, (self.W_pixels - self.offset) + 1, self.patch_size):
            for y in range(self.offset, (self.H_pixels - self.offset) + 1, self.patch_size):
                self.coords.append((x, y))
        n_coords = len(self.coords)

        pop_tot = self.population + self.learner_population
        self.possible_agents = [str(i) for i in range(self.population, pop_tot)]  
        self._agent_selector = agent_selector(self.possible_agents)
        self.agent = self._agent_selector.reset()

        # Create patches
        self.patches = {
            self.coords[i]: {
                            "id": i,
                            "food_pheromone": 0.0,
                            "nest_pheromone": 0.0,
                            "turtles": [],
                            "food": 0,
                            "nest": 0,
                            "food_pile" : 0
                            } for i in range(n_coords)
        }

        # Setup nest
        if self.W % 2 == 0:
            self.nest_pos = self.coords[(len(self.coords) // 2) - (self.H // 2)]
        else:
            self.nest_pos = self.coords[(len(self.coords) // 2)]

        self.patches = self._setup_nest(self.patches, self.nest_pos)
        #print(self.nest_pos)
        #print(self.patches)

        # Setup food piles
        self.food_piles_pos = (
            (self.coords[0][0] + self.patch_size * int(self.W * 1/5), self.coords[0][1] + self.patch_size * int(self.H * 1/5)), 
            (self.coords[0][0] + self.patch_size * int(self.W * 1/6), self.coords[0][1] + self.patch_size * int(self.H * 3/4)), 
            (self.coords[n_coords - 1][0] - self.patch_size * int(self.W * 1/6), self.nest_pos[1])
        )

        self.patches, self.total_food_amount = self._setup_food_piles(self.patches, self.food_piles_pos)
        #print("Total food: ", self.total_food_amount)

        #Create learner turtles
        self.learners = {
            i: {
                "pos": self.nest_pos,
                "dir": np.random.randint(self.N_DIRS),
                "food": 0 
            } for i in range(self.population, pop_tot)
        }

        # Test added stuf
        self.learners_view_of_nest = {
            i: 0 for i in range(self.population, pop_tot)
        }
        self.food_individually_retrieved = {
            i: 0 for i in range(self.population, pop_tot)
        }
        self.learners_done = 0
        self.all_learners_done = False

        # Create NON learner turtles
        self.turtles = {
            i: {
                "pos": self.coords[np.random.randint(n_coords)]
            } for i in range(self.population)
        }

        # Add learners and turtles to the respective patches
        for l in self.learners:
            self.patches[self.learners[l]['pos']]['turtles'].append(l)
        for t in self.turtles:
            self.patches[self.turtles[t]['pos']]['turtles'].append(t)
        
        self.lay_patches = self._find_neighbours(self.lay_area)

        if self.diffuse_mode == "cascade":
            assert isinstance(self.diffuse_area, int), "Error: diffuse_area must be an int"
            self.diffuse_patches = self._find_neighbours_cascade(self.diffuse_area)
        elif self.diffuse_mode in ("rng", "sorted", "filter", "rng-filter"):
            assert isinstance(self.diffuse_area, int), "Error: diffuse_area must be an int"
            self.diffuse_patches = self._find_neighbours(self.diffuse_area)

        self.fov = self._field_of_view(self.wiggle_patches)
        self.ph_fov = self._field_of_view(self.sniff_patches)

        # Create an action space for each agent
        self._action_spaces = {
            a: Discrete(len(self.actions))
            for a in self.possible_agents
        }

        # Create an observation space for each agent
        self._observation_spaces = {
                a: Box(low=0.0, high=np.inf, shape=(self.sniff_patches*2+3,), dtype=np.float32)
                for a in self.possible_agents
            }
        
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(self.population, pop_tot)))
        )

    def _field_of_view(self, n_patches):
        # Pre-compute every possible agent's direction
        fov = {}
        
        if n_patches < self.N_DIRS:
            central = n_patches // 2
            sliding_window = []
            
            for i in range(self.N_DIRS):
                tmp = []
                for j in range(n_patches):
                    tmp.append((i + j) % self.N_DIRS)
                sliding_window.append(tmp)
            sliding_window = sorted(sliding_window, key=lambda x: x[central])
            
            for c in self.coords:
                tmp_fov = self.movements + c
                tmp_fov[:, 0] %= self.W_pixels 
                tmp_fov[:, 1] %= self.H_pixels 
                fov[c] = tmp_fov[sliding_window, :]
        else:
            for c in self.coords:
                tmp_fov = self.movements + c
                tmp_fov[:, 0] %= self.W_pixels 
                tmp_fov[:, 1] %= self.H_pixels 
                fov[c] = tmp_fov

        return fov
    
    def _get_new_positions(self, possible_patches, agent):
        pos = agent["pos"]
        direction = agent["dir"]
        if len(possible_patches[pos].shape) > 2:
            return possible_patches[pos][direction], direction
        else:
            return possible_patches[pos], direction
    
    def _get_new_direction(self, n_patches, old_dir, idx_dir):
        start = (old_dir - (n_patches // 2)) % self.N_DIRS 
        new_dirs = np.array([(i + start) % self.N_DIRS for i in range(n_patches)])
        return new_dirs[idx_dir]
    
    def _get_obs2(self, agent): # obs = [agent_has_food, patch_has_food, agent_in_nest, food_pheromone_in_ph_fov, nest_pheromone_in_ph_fov]
        f, _ = self._get_new_positions(self.ph_fov, agent)
        obs = [int(agent["food"]>0), int(self.patches[agent["pos"]]["food"]>0), self.patches[agent["pos"]]["nest"]]
        obs.extend([self.patches[tuple(i)]["food_pheromone"] for i in f])
        obs.extend([self.patches[tuple(i)]["nest_pheromone"] for i in f])
        obs = np.array(obs)
        return obs
    
    def convert_observation(self, obs):
        agent_has_food = int(obs[0].item())
        patch_has_food = int(obs[1].item())
        agent_in_nest = int(obs[2].item())
        food_pheromone_obs = obs[3:3+self.sniff_patches]
        nest_pheromone_obs = obs[3+self.sniff_patches:]

        if np.unique(food_pheromone_obs).shape[0] == 1:
            food_pheromone_obs_index = np.random.randint(self.sniff_patches)
        else:
            food_pheromone_obs_index = food_pheromone_obs.argmax().item()

        if np.unique(nest_pheromone_obs).shape[0] == 1:
            nest_pheromone_obs_index = np.random.randint(self.sniff_patches)
        else:
            nest_pheromone_obs_index = nest_pheromone_obs.argmax().item()

        #food_pheromone_obs_index = food_pheromone_obs.argmax().item()
        #nest_pheromone_obs_index = nest_pheromone_obs.argmax().item()
        
        food_pheromone_obs_bin = bin(food_pheromone_obs_index).split('b')[1]
        nest_pheromone_obs_bin = bin(nest_pheromone_obs_index).split('b')[1]

        if self.sniff_patches > 3:
            if len(food_pheromone_obs_bin) < 3:
                food_pheromone_obs_bin = "0"*(3-len(food_pheromone_obs_bin))+food_pheromone_obs_bin
            if len(nest_pheromone_obs_bin) < 3:
                nest_pheromone_obs_bin = "0"*(3-len(nest_pheromone_obs_bin))+nest_pheromone_obs_bin
        else:
            if len(food_pheromone_obs_bin) < 2:
                food_pheromone_obs_bin = "0"*(2-len(food_pheromone_obs_bin))+food_pheromone_obs_bin
            if len(nest_pheromone_obs_bin) < 2:
                nest_pheromone_obs_bin = "0"*(2-len(nest_pheromone_obs_bin))+nest_pheromone_obs_bin

        obs_bin = str(agent_has_food) + str(patch_has_food) + str(agent_in_nest) + food_pheromone_obs_bin + nest_pheromone_obs_bin
        return int(obs_bin, 2)
    
    def _find_neighbours(self, area: int):
        """
        For each patch, find neighbouring patches within square radius 'area'
        """

        neighbours = {}
        
        for p in self.patches:
            neighbours[p] = []
            for x in range(p[0], p[0] + (area * self.patch_size) + 1, self.patch_size):
                for y in range(p[1], p[1] + (area * self.patch_size) + 1, self.patch_size):
                    x, y = self._wrap(x, y)
                    neighbours[p].append((x, y))
            for x in range(p[0], p[0] - (area * self.patch_size) - 1, -self.patch_size):
                for y in range(p[1], p[1] - (area * self.patch_size) - 1, -self.patch_size):
                    x, y = self._wrap(x, y)
                    neighbours[p].append((x, y))
            for x in range(p[0], p[0] + (area * self.patch_size) + 1, self.patch_size):
                for y in range(p[1], p[1] - (area * self.patch_size) - 1, -self.patch_size):
                    x, y = self._wrap(x, y)
                    neighbours[p].append((x, y))
            for x in range(p[0], p[0] - (area * self.patch_size) - 1, -self.patch_size):
                for y in range(p[1], p[1] + (area * self.patch_size) + 1, self.patch_size):
                    x, y = self._wrap(x, y)
                    neighbours[p].append((x, y))
            neighbours[p] = list(set(neighbours[p]))

        return neighbours

    def _find_neighbours_cascade(self, area: int):
        """
        For each patch, find neighbouring patches within square radius 'area', 1 step at a time
        (visiting first 1-hop patches, then 2-hops patches, and so on)
        """

        neighbours = {}
        
        for p in self.patches:
            neighbours[p] = []
            for ring in range(area):
                for x in range(p[0] + (ring * self.patch_size), p[0] + ((ring + 1) * self.patch_size) + 1,
                               self.patch_size):
                    for y in range(p[1] + (ring * self.patch_size), p[1] + ((ring + 1) * self.patch_size) + 1,
                                   self.patch_size):
                        if (x, y) not in neighbours[p]:
                            neighbours[p].append((x, y))
                for x in range(p[0] + (ring * self.patch_size), p[0] - ((ring + 1) * self.patch_size) - 1,
                               -self.patch_size):
                    for y in range(p[1] + (ring * self.patch_size), p[1] - ((ring + 1) * self.patch_size) - 1,
                                   -self.patch_size):
                        if (x, y) not in neighbours[p]:
                            neighbours[p].append((x, y))
                for x in range(p[0] + (ring * self.patch_size), p[0] + ((ring + 1) * self.patch_size) + 1,
                               self.patch_size):
                    for y in range(p[1] + (ring * self.patch_size), p[1] - ((ring + 1) * self.patch_size) - 1,
                                   -self.patch_size):
                        if (x, y) not in neighbours[p]:
                            neighbours[p].append((x, y))
                for x in range(p[0] + (ring * self.patch_size), p[0] - ((ring + 1) * self.patch_size) - 1,
                               -self.patch_size):
                    for y in range(p[1] + (ring * self.patch_size), p[1] + ((ring + 1) * self.patch_size) + 1,
                                   self.patch_size):
                        if (x, y) not in neighbours[p]:
                            neighbours[p].append((x, y))
            neighbours[p] = [self._wrap(x, y) for (x, y) in neighbours[p]]

        return neighbours

    def _wrap(self, x: int, y: int):
        """
        Wrap x,y coordinates around the torus

        :param x: the x coordinate to wrap
        :param y: the y coordinate to wrap
        :return: the wrapped x, y
        """
        return x % self.W_pixels, y % self.H_pixels

    def process_agent(self):
        """
        In this methods we compute the agent's reward and it's observation.
        """

        observations = self._get_obs2(self.learners[self.agent])

        return observations
    
    def step(self, action: int):

        # if(self.terminations[self.agent_selection] or self.truncations[self.agent_selection]):
        #     self._was_dead_step(action)
        #     return
        
        self.agent = self.agent_name_mapping[self.agent_selection]  # ID of agent

        #self.observations[str(self.agent)] = self.process_agent()

        if not (self.terminations[self.agent_selection] or self.truncations[self.agent_selection]):

            agent_has_food = self.observations[str(self.agent)][0]
            patch_has_food = self.observations[str(self.agent)][1]
            agent_in_nest = self.observations[str(self.agent)][2]
            food_pheromone_obs = self.observations[str(self.agent)][3:3+self.sniff_patches]
            nest_pheromone_obs = self.observations[str(self.agent)][3+self.sniff_patches:]
            
            if action == 0:     # Walk
                self.patches, self.learners[self.agent] = self.walk2(self.patches, self.learners[self.agent])
            elif action == 1:   # Lay food pheromone
                self.patches = self.lay_pheromone(self.patches, self.learners[self.agent]['pos'])
            elif action == 2:   # Follow food pheromone
                max_pheromone, max_coords, max_ph_dir = self._find_max_pheromone2(
                    self.learners[self.agent],
                    food_pheromone_obs   
                )

                if max_pheromone >= self.sniff_threshold:

                    if self.follow_food_pheromone_mode == "basic":
                        self.patches, self.learners[self.agent] = self.follow_pheromone2(
                                self.patches,
                                max_coords,
                                max_ph_dir,
                                self.learners[self.agent]
                            ) 
                    elif self.follow_food_pheromone_mode == "turn_away":

                        max_pheromone_nest, max_coords_nest, max_ph_dir_nest = self._find_max_pheromone2(
                        self.learners[self.agent],
                        nest_pheromone_obs   
                        )

                        if max_ph_dir == max_ph_dir_nest:
                            self.learners[self.agent]["dir"] = (self.learners[self.agent]["dir"] + 4) % self.N_DIRS
                            #self.patches, self.learners[self.agent] = self.step_forward(self.patches, self.learners[self.agent])
                        else:
                            self.patches, self.learners[self.agent] = self.follow_pheromone2(
                                self.patches,
                                max_coords,
                                max_ph_dir,
                                self.learners[self.agent]
                            )
                    elif self.follow_food_pheromone_mode == "clip":
                        if max_pheromone <= self.sniff_limit:
                            self.patches, self.learners[self.agent] = self.follow_pheromone2(
                                self.patches,
                                max_coords,
                                max_ph_dir,
                                self.learners[self.agent]
                            )
                        else:
                            self.patches, self.learners[self.agent] = self.walk2(self.patches, self.learners[self.agent])
                else:
                    self.patches, self.learners[self.agent] = self.walk2(self.patches, self.learners[self.agent])
            elif action == 3:   # Follow nest pheromone
                max_pheromone, max_coords, max_ph_dir = self._find_max_pheromone2(
                    self.learners[self.agent],
                    nest_pheromone_obs   
                )
                if max_pheromone >= self.sniff_threshold:
                    self.patches, self.learners[self.agent] = self.follow_pheromone2(
                        self.patches,
                        max_coords,
                        max_ph_dir,
                        self.learners[self.agent]
                    )
            elif action == 4:   # Take food
                #print("agent", self.agent, "patch_has_food: ", patch_has_food)
                if agent_has_food == 0 and patch_has_food == 1 and agent_in_nest == 0: # Ant can't take nest food now
                    self.patches, self.learners[self.agent] = self.take_food(self.patches, self.learners[self.agent])
                else:
                    self.patches, self.learners[self.agent] = self.walk2(self.patches, self.learners[self.agent])
            elif action == 5:   # Drop food
                if agent_has_food == 1:
                    self.patches, self.learners[self.agent] = self.drop_food(self.patches, self.learners[self.agent])
                    if agent_in_nest == 1:
                        self.food_individually_retrieved[self.agent] += 1
                else:
                    self.patches, self.learners[self.agent] = self.walk2(self.patches, self.learners[self.agent])
            elif action == 6:   # Lay food pheromone and walk
                self.patches, self.learners[self.agent] = self.walk2(self.patches, self.learners[self.agent])
                self.patches = self.lay_pheromone(self.patches, self.learners[self.agent]['pos'])
            elif action == 7:   # Lay food pheromone and follow nest pheromone
                max_pheromone, max_coords, max_ph_dir = self._find_max_pheromone2(
                    self.learners[self.agent],
                    nest_pheromone_obs,     
                )
                if max_pheromone >= self.sniff_threshold:
                    self.patches, self.learners[self.agent] = self.follow_pheromone2(
                        self.patches,
                        max_coords,
                        max_ph_dir,
                        self.learners[self.agent]
                    )
                else:
                    self.patches, self.learners[self.agent] = self.walk2(self.patches, self.learners[self.agent])
                self.patches = self.lay_pheromone(self.patches, self.learners[self.agent]['pos'])
            #elif action == 8: # Step forward test
            #    self.patches, self.learners[self.agent] = self.step_forward(self.patches, self.learners[self.agent])
            else:
                raise ValueError("Action out of range!")
            
            #self.observations[str(self.agent)] = self.process_agent()

            if self.reward_type == "reward_nest_food_punish_piles_food_and_wandering_time":
                self.carrying_food_ticks, self.searching_food_ticks, self.rewards_cust, cur_reward = self.reward_nest_food_punish_piles_food_and_wandering_time(self.carrying_food_ticks, self.searching_food_ticks, self.rewards_cust)
            elif self.reward_type == "reward_nest_food_punish_wandering_time":
                self.carrying_food_ticks, self.searching_food_ticks, self.rewards_cust, cur_reward = self.reward_nest_food_punish_wandering_time(self.carrying_food_ticks, self.searching_food_ticks, self.rewards_cust)
            elif self.reward_type == "reward_relative_food_punish_wandering_time":
                self.carrying_food_ticks, self.searching_food_ticks, self.rewards_cust, cur_reward = self.reward_relative_food_punish_wandering_time(self.carrying_food_ticks, self.searching_food_ticks, self.rewards_cust)
            elif self.reward_type == "reward_nest_food_punish_wandering_time_incremental":
                self.rewards_cust, cur_reward = self.reward_nest_food_punish_wandering_time_incremental(self.rewards_cust)
            elif self.reward_type == "reward_indiviually_retrieved_nest_food_punish_wandering_time_incremental":
                self.rewards_cust, cur_reward = self.reward_indiviually_retrieved_nest_food_punish_wandering_time_incremental(self.rewards_cust)
            elif self.reward_type == "reward_indiviually_retrieved_nest_food_punish_wandering_time":
                self.carrying_food_ticks, self.searching_food_ticks, self.rewards_cust, cur_reward = self.reward_indiviually_retrieved_nest_food_punish_wandering_time(self.carrying_food_ticks, self.searching_food_ticks, self.rewards_cust)
            
            # Terminate the current agent if there is no more food to be found

            food_in_nest = 0
            for coords, info in self.patches.items():
                if info["nest"]:
                    food_in_nest += info["food"]

            if food_in_nest == self.total_food_amount:
                self.terminations[self.agent_selection] = True

            if self._agent_selector.is_last():
                for ag in self.agents:
                    self.rewards[ag] = self.rewards_cust[self.agent_name_mapping[ag]][-1]
                if len(self.turtles) > 0:
                    self.turtles, self.patches = self.move(self.turtles, self.patches)
                if self.diffuse_mode == "gaussian":
                    self.patches = self._diffuse_and_evaporate(self.patches)
                else:
                    self.patches = self._diffuse(self.patches)
                    self.patches = self._evaporate(self.patches)
            else:
                self._clear_rewards()

        else:

            self.learners_done += 1
            self.all_learners_done = self.learners_done == len(self.learners)
            if self._agent_selector.is_last():
                for ag in self.agents:
                    self.rewards[ag] = self.rewards_cust[self.agent_name_mapping[ag]][-1]
            else:
                self._clear_rewards()      
            
        self.agent_selection = self._agent_selector.next()
        self._cumulative_rewards[str(self.agent)] = 0
        self._accumulate_rewards()

    def lay_pheromone(self, patches, pos):
        """
        Lay 'amount' pheromone in square 'area' centred in 'pos'
        """
        for p in self.lay_patches[pos]:
            patches[p]['food_pheromone'] += self.lay_amount
        
        return patches

    def _diffuse(self, patches):
        """
        Diffuses pheromone from each patch to nearby patches controlled through self.diffuse_area patches in a way
        controlled through self.diffuse_mode:
            'simple' = Python-dependant (dict keys "ordering")
            'rng' = random visiting
            'sorted' = diffuse first the patches with more pheromone
            'filter' = do not re-diffuse patches receiving pheromone due to diffusion
        """
        n_size = len(self.diffuse_patches[list(patches.keys())[0]])  # same for every patch
        patch_keys = list(patches.keys())
        
        if self.diffuse_mode == 'rng':
            random.shuffle(patch_keys)
        elif self.diffuse_mode == 'sorted':
            patch_list = list(patches.items())
            patch_list = sorted(patch_list, key=lambda t: t[1]['food_pheromone'], reverse=True)
            patch_keys = [t[0] for t in patch_list]
        elif self.diffuse_mode == 'filter':
            patch_keys = [k for k in patches if patches[k]['food_pheromone'] > 0]
        elif self.diffuse_mode == 'rng-filter':
            patch_keys = [k for k in patches if patches[k]['food_pheromone'] > 0]
            random.shuffle(patch_keys)
        
        for patch in patch_keys:
            p = patches[patch]['food_pheromone']
            ratio = p / n_size
            
            if p > 0:
                diffuse_keys = self.diffuse_patches[patch][:]
                
                for n in diffuse_keys:
                    patches[n]['food_pheromone'] += ratio
                
                patches[patch]['food_pheromone'] = ratio

        return patches
    
    def _evaporate(self, patches):
        """
        Evaporates pheromone from each patch according to param self.evaporation
        """
        for patch in patches.keys():
            patches[patch]['food_pheromone'] *= self.evaporation

        return patches

    def _diffuse_and_evaporate(self, patches):
        """
        This method combine the _diffuse2 and _evaporate methods in one function.
        It is the method currently used.
        """
        # Diffusion
        grid = np.array([patches[p]["food_pheromone"] for p in patches.keys()]).reshape((self.W, self.H))
        grid = gaussian_filter(grid, sigma=self.diffuse_area, mode="wrap")
        grid = grid.flatten()
        # Evaporation
        grid *= self.evaporation
        # Write values
        for p, g in zip(patches, grid):
            patches[p]['food_pheromone'] = g
        
        return patches
    
    def _find_max_pheromone2(self, agent, obs):
        # Det = follow greatest pheromone
        f, direction = self._get_new_positions(self.ph_fov, agent)
    
        idx = obs.argmax().item()

        #weights = obs.tolist()
        #if all([w == 0 for w in obs]):
        #    winner = np.random.choice(len(obs))
        #else:
        #    possible_idx = np.arange(len(obs)).tolist()
        #    weights = (gaussian_filter((obs - obs.min())/(obs.max()-obs.min()), sigma = 1)*100).tolist()
        #    print(weights)
        #    winner = random.choices(possible_idx, weights=weights, k=1)[0]
        #idx = winner

        ph_val = obs[idx]
        ph_pos = tuple(f[idx])
        if self.sniff_patches < self.N_DIRS:
            ph_dir = self._get_new_direction(self.sniff_patches, direction, idx)
        else:
            ph_dir = idx
        return ph_val, ph_pos, ph_dir
    
    def follow_pheromone2(self, patches, ph_coords, ph_dir, turtle):
        patches[turtle['pos']]['turtles'].remove(self.agent)
        turtle["pos"] = ph_coords
        patches[turtle['pos']]['turtles'].append(self.agent)
        turtle["dir"] = ph_dir
        return patches, turtle
    
    def walk2(self, patches, turtle):
        f, direction = self._get_new_positions(self.fov, turtle)
        idx_dir = np.random.randint(f.shape[0])
        patches[turtle['pos']]['turtles'].remove(self.agent)
        turtle["pos"] = tuple(f[idx_dir])
        patches[turtle['pos']]['turtles'].append(self.agent)
        if self.wiggle_patches < self.N_DIRS:
            turtle["dir"] = self._get_new_direction(self.wiggle_patches, direction, idx_dir)
        else:
            turtle["dir"] = idx_dir
        return patches, turtle
    
    def step_forward(self, patches, turtle):
        f, direction = self._get_new_positions(self.fov, turtle)
        if self.wiggle_patches < self.N_DIRS:
            idx_dir = f.shape[0] // 2
        else:
            idx_dir = direction
        patches[turtle['pos']]['turtles'].remove(self.agent)
        turtle["pos"] = tuple(f[idx_dir])
        patches[turtle['pos']]['turtles'].append(self.agent)
        return patches, turtle
    
    def reset(self, seed=None, return_info=True, options=None):
        """
        Reset env.
        """
        # empty stuff
        pop_tot = self.population + self.learner_population
        self.rewards_cust = {i: [] for i in range(self.population, pop_tot)}
        self.carrying_food_ticks = {i: 0 for i in range(self.population, pop_tot)}
        self.searching_food_ticks = {i: 0 for i in range(self.population, pop_tot)}
        
        #Initialize attributes for PettingZoo Env
        self.agents = self.possible_agents[:]
        self._agent_selector.reinit(self.agents)
        self.agent_selection = self._agent_selector.next()
        
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.state = {agent: None for agent in self.agents}
        
        # re-position learner turtle
        for l in self.learners:
            self.patches[self.learners[l]['pos']]['turtles'].remove(l)
            self.learners[l]['pos'] = self.nest_pos
            self.patches[self.learners[l]['pos']]['turtles'].append(l)  # DOC id of learner turtle
            self.learners[l]['food'] = 0
            self.learners_view_of_nest[l] = 0
            self.food_individually_retrieved[l] = 0
        # re-position NON learner turtles
        for t in self.turtles:
            self.patches[self.turtles[t]['pos']]['turtles'].remove(t)
            self.turtles[t]['pos'] = self.coords[np.random.randint(len(self.coords))]
            self.patches[self.turtles[t]['pos']]['turtles'].append(t)
        # patches-own [food_pheromone] - amount of pheromone in the patch
        for p in self.patches:
            self.patches[p]['food_pheromone'] = 0.0
            self.patches[p]['nest_pheromone'] = 0.0
            self.patches[p]['food'] = 0

        self.learners_done = 0
        self.all_learners_done = False

        self.patches, self.total_food_amount = self._setup_food_piles(self.patches, self.food_piles_pos)
        self.patches = self._setup_nest(self.patches, self.nest_pos)
        #print("Total food reset: ", self.total_food_amount)

        self.observations = {
            a: np.zeros(self.wiggle_patches*2+3, dtype=np.float32)
            for a in self.agents
        }

        for agent, obs in self.observations.items():
            obs[2] = 1.0
        
        self._agent_selector.reinit(self.agents)
        self.agent_selection = self._agent_selector.next()

    def _setup_nest(self, patches, nest_pos):

        nest_coords = []
        pos = (nest_pos[0] - self.patch_size * 2, nest_pos[0] + self.patch_size * 2)
        start = nest_pos[1] - self.patch_size
        finish = nest_pos[1] + self.patch_size * 2
        nest_coords.extend([
            (p, j) for p in pos for j in range(start, finish, self.patch_size)
        ]) 

        pos = (nest_pos[0] - self.patch_size, nest_pos[0], nest_pos[0] + self.patch_size)
        start = nest_pos[1] - self.patch_size * 2
        finish = nest_pos[1] + self.patch_size * 3
        nest_coords.extend([
            (p, i) for p in pos for i in range(start, finish, self.patch_size)
        ])

        for coords in nest_coords:
            patches[coords]["nest"] = 1

        for patch in patches:
            patches[patch]["nest_pheromone"] = 100 - (
                ((abs(patch[0]-nest_pos[0])) + (abs(patch[1]-nest_pos[1]))) / self.patch_size
            )
            #patches[patch]["nest_pheromone"] = 100 - math.sqrt(pow((patch[0]-nest_pos[0])/self.patch_size, 2) + pow((patch[1]-nest_pos[1])/self.patch_size, 2))
            
        return patches
    
    def _setup_food_piles(self, patches, food_piles_pos):

        food_piles_coords = []
        for food_pile in food_piles_pos:
            pos = (food_pile[0] - self.patch_size * 2, food_pile[0] + self.patch_size * 2)
            start = food_pile[1] - self.patch_size
            finish = food_pile[1] + self.patch_size * 2
            food_piles_coords.extend([
                (p, j) for p in pos for j in range(start, finish, self.patch_size)
            ]) 

            pos = (food_pile[0] - self.patch_size, food_pile[0], food_pile[0] + self.patch_size)
            start = food_pile[1] - self.patch_size * 2
            finish = food_pile[1] + self.patch_size * 3
            food_piles_coords.extend([
                (p, i) for p in pos for i in range(start, finish, self.patch_size)
            ])

        for coords in food_piles_coords:
            patches[coords]["food"] = np.random.randint(1, 3)
            patches[coords]["food_pile"] = 1

        total_food = 0
        for patch, info in patches.items():
            if info["food_pile"]:
                total_food += info["food"]

        return patches, total_food

    def drop_food(self, patches, ant):
        ant["food"] -= 1
        patches[ant["pos"]]["food"] += 1
        if self.take_drop_mode == "turn_away":
            ant["dir"] = (ant["dir"] + 4) % self.N_DIRS
        return patches, ant
    
    def take_food(self, patches, ant):
        ant["food"] += 1
        patches[ant["pos"]]["food"] -= 1
        if self.take_drop_mode == "turn_away":
            ant["dir"] = (ant["dir"] + 4) % self.N_DIRS
        return patches, ant

    def reward_nest_food_punish_piles_food_and_wandering_time(self, carrying_food_ticks, searching_food_ticks, rewards_cust):
        if self.learners[self.agent]["food"] > 0:
            carrying_food_ticks[self.agent] += 1
        else:
            searching_food_ticks[self.agent] += 1

        food_nest = 0
        food_piles = 0
        for coords, info in self.patches.items():
            if info["nest"]:
                food_nest += info["food"]
            else:
                food_piles += info["food"]

        cur_reward = carrying_food_ticks[self.agent]*self.penalty + \
                     searching_food_ticks[self.agent]*self.penalty + \
                     food_nest * self.reward + \
                     food_piles * self.penalty

        rewards_cust[self.agent].append(cur_reward)
        return carrying_food_ticks, searching_food_ticks, rewards_cust, cur_reward
    
    def reward_nest_food_punish_wandering_time(self, carrying_food_ticks, searching_food_ticks, rewards_cust):

        # Test part

        # food_piles = 0
        # for coords, info in self.patches.items():
        #     if info["food_pile"]:
        #         food_piles += info["food"]

        # if food_piles:
        #     if self.learners[self.agent]["food"] > 0:
        #         carrying_food_ticks[self.agent] += 1
        #     else:
        #         searching_food_ticks[self.agent] += 1


        # End test part

        if self.learners[self.agent]["food"] > 0:
            carrying_food_ticks[self.agent] += 1
        else:
            searching_food_ticks[self.agent] += 1

        food_nest = 0
        for coords, info in self.patches.items():
            if info["nest"]:
                food_nest += info["food"]

        cur_reward = carrying_food_ticks[self.agent]*self.penalty + \
                     searching_food_ticks[self.agent]*self.penalty + \
                     food_nest * self.reward

        rewards_cust[self.agent].append(cur_reward)
        return carrying_food_ticks, searching_food_ticks, rewards_cust, cur_reward
    
    def reward_relative_food_punish_wandering_time(self, carrying_food_ticks, searching_food_ticks, rewards_cust):
        if self.learners[self.agent]["food"] > 0:
            carrying_food_ticks[self.agent] += 1
        else:
            searching_food_ticks[self.agent] += 1

        food_nest = 0
        food_piles = 0
        for coords, info in self.patches.items():
            if info["nest"]:
                food_nest += info["food"]
            else:
                food_piles += info["food"]

        cur_reward = carrying_food_ticks[self.agent]*self.penalty + \
                     searching_food_ticks[self.agent]*self.penalty + \
                     food_nest/max(food_piles, 0.5) * self.reward
        
        rewards_cust[self.agent].append(cur_reward)
        return carrying_food_ticks, searching_food_ticks, rewards_cust, cur_reward
    
    def reward_nest_food_punish_wandering_time_incremental(self, rewards_cust):

        food_nest = 0
        for coords, info in self.patches.items():
            if info["nest"]:
                food_nest += info["food"]

        cur_reward = self.penalty + (food_nest - self.learners_view_of_nest[self.agent]) * self.reward
        
        self.learners_view_of_nest[self.agent] = food_nest

        rewards_cust[self.agent].append(cur_reward)
        return rewards_cust, cur_reward
    
    def reward_indiviually_retrieved_nest_food_punish_wandering_time_incremental(self,rewards_cust):

        cur_reward = self.penalty + self.food_individually_retrieved[self.agent] * self.reward
        
        self.food_individually_retrieved[self.agent] = 0

        rewards_cust[self.agent].append(cur_reward)
        return rewards_cust, cur_reward
    
    def reward_indiviually_retrieved_nest_food_punish_wandering_time(self, carrying_food_ticks, searching_food_ticks, rewards_cust):

        if self.learners[self.agent]["food"] > 0:
            carrying_food_ticks[self.agent] += 1
        else:
            searching_food_ticks[self.agent] += 1

        cur_reward = carrying_food_ticks[self.agent]*self.penalty + \
                     searching_food_ticks[self.agent]*self.penalty + \
                     self.food_individually_retrieved[self.agent] * self.reward

        rewards_cust[self.agent].append(cur_reward)
        return carrying_food_ticks, searching_food_ticks, rewards_cust, cur_reward
    

class AntsVisualizer:
    def __init__(
        self,
        W_pixels,
        H_pixels,
        **kwargs
    ):
        self.fps = kwargs['FPS']
        self.shade_strength = kwargs['SHADE_STRENGTH']
        self.show_chem_text = kwargs['SHOW_CHEM_TEXT']
        self.show_food_text = kwargs['SHOW_FOOD_TEXT']
        self.food_font_size = kwargs['FOOD_FONT_SIZE']
        self.chemical_font_size = kwargs['CHEMICAL_FONT_SIZE']
        self.sniff_threshold = kwargs['sniff_threshold']
        self.patch_size = kwargs['PATCH_SIZE']
        self.turtle_size = kwargs['TURTLE_SIZE']

        self.W_pixels = W_pixels
        self.H_pixels = H_pixels
        self.offset = self.patch_size // 2
        self.screen = pygame.display.set_mode((self.W_pixels, self.H_pixels))
        self.clock = pygame.time.Clock()
        pygame.font.init()
        self.food_font = pygame.font.SysFont("arial", self.food_font_size)
        self.chemical_font = pygame.font.SysFont("arial", self.chemical_font_size)
        self.first_gui = True

        self.BLACK = (0, 0, 0)
        self.BLUE = (0, 0, 255)
        self.WHITE = (255, 255, 255)
        self.RED = (190, 0, 0)
        self.ORANGE = (222, 89, 28)
        self.YELLOW = (255, 255, 0)
        self.GREEN = (0, 190, 0)
        self.VIOLET = (151, 89, 154)
        self.LIGHT_BLUE = (0,98,196),
        self.BROWN = (101, 67, 33)

    def render(
        self,
        patches,
        learners,
        turtles
    ):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:  # window closed -> program quits
                pygame.quit()

        if self.first_gui:
            self.first_gui = False
            pygame.init()
            pygame.display.set_caption("ANTS")

        self.screen.fill(self.BLACK)

        # draw patches
        for p in patches:
            chem = round(patches[p]['food_pheromone']) * self.shade_strength
            pygame.draw.rect(
                self.screen,
                (0, chem if chem <= 255 else 255, 0),
                pygame.Rect(p[0] - self.offset,
                    p[1] - self.offset,
                    self.patch_size,
                    self.patch_size
                )
            )
            if self.show_chem_text and (not sys.gettrace() is None or
                                        patches[p][
                                            'food_pheromone'] >= self.sniff_threshold):  # if debugging show text everywhere, even 0
                text = self.chemical_font.render(str(round(patches[p]['food_pheromone'], 1)), True, self.GREEN)
                self.screen.blit(text, text.get_rect(center=p))

        # draw nest and food
        for patch_coords, patch_info in patches.items():
            if patch_info["nest"]:
                pygame.draw.rect(
                    self.screen,
                    self.VIOLET,
                    pygame.Rect(patch_coords[0] - self.offset,
                        patch_coords[1] - self.offset,
                        self.patch_size,
                        self.patch_size
                    )
                )
            #text = self.food_font.render(str(patch_info["nest_pheromone"]), True, self.WHITE)
            #self.screen.blit(text, text.get_rect(center=patch_coords))
            #text = self.food_font.render(str(patch_info["nest"]), True, self.WHITE)
            #self.screen.blit(text, text.get_rect(center=patch_coords))
            if patch_info["food"] > 0:
                pygame.draw.rect(
                self.screen,
                self.LIGHT_BLUE,
                pygame.Rect(patch_coords[0] - self.offset,
                    patch_coords[1] - self.offset,
                    self.patch_size,
                    self.patch_size
                    )
                )
                if self.show_food_text:
                    text = self.food_font.render(str(patch_info["food"]), True, self.WHITE)
                    self.screen.blit(text, text.get_rect(center=patch_coords))
            else:
                if patch_info["food"] < 0:
                    pygame.draw.rect(
                    self.screen,
                    self.BROWN,
                    pygame.Rect(patch_coords[0] - self.offset,
                        patch_coords[1] - self.offset,
                        self.patch_size,
                        self.patch_size
                        )
                    )
                    if self.show_food_text:
                        text = self.food_font.render(str(patch_info["food"]), True, self.WHITE)
                        self.screen.blit(text, text.get_rect(center=patch_coords))
            #    else:
            #        pygame.draw.rect(
            #        self.screen,
            #        self.ORANGE,
            #        pygame.Rect(patch_coords[0] - self.offset,
            #            patch_coords[1] - self.offset,
            #            self.patch_size,
            #            self.patch_size
            #            )
            #        )
             #       if self.show_food_text:
             #           text = self.food_font.render(str(patch_info["food"]), True, self.WHITE)
             #           self.screen.blit(text, text.get_rect(center=patch_coords))


        # draw learners
        for learner in learners.values():
            if learner['food'] == 1:
                pygame.draw.circle(
                    self.screen,
                    self.YELLOW,
                    (learner['pos'][0], learner['pos'][1]),
                    self.turtle_size // 2
                )
            else:
                pygame.draw.circle(
                    self.screen,
                    self.RED,
                    (learner['pos'][0], learner['pos'][1]),
                    self.turtle_size // 2
                )

        # draw NON learners
        for turtle in turtles.values():
            pygame.draw.circle(self.screen, self.BLUE, (turtle['pos'][0], turtle['pos'][1]), self.turtle_size // 2)

        #for p in patches:
        #    if len(patches[p]['turtles']) > 1:
        #        text = self.cluster_font.render(
        #            str(len(patches[p]['turtles'])),
        #            True,
        #            self.RED if -1 in patches[p]['turtles'] else self.WHITE
        #        )
        #        self.screen.blit(text, text.get_rect(center=p))

        self.clock.tick(self.fps)
        pygame.display.flip()

        return pygame.surfarray.array3d(self.screen)

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()


def main():
    params = {
        "population": 0,
        "diffuse_area" : 1,
        "diffuse_mode" : "gaussian",
        "follow_food_pheromone_mode" : "clip", # either "basic", "turn_away" or "clip"
        "take_drop_mode": "keep_direction", # either "keep_direction" or "turn_away"
        "lay_area" : 1,
        "evaporation" : 0.90,
        "PATCH_SIZE" : 20,
        "TURTLE_SIZE" : 10,

        "H" : 40,
        "W" : 40,
        "learner_population" : 50,
        "sniff_threshold": 0.9,
        "sniff_limit": 2,
        "wiggle_patches": 3,
        "sniff_patches": 3,
        #"smell_area": 1,
        "lay_amount": 5,
        "actions": [
            "random-walk",
            "lay-food-pheromone",
            "follow-food-pheromone",
            "follow-nest-pheromone",
            "take-food",
            "drop-food",
            "random-walk-and-lay-food-pheromone",
            "follow-nest-pheromone-and-lay-food-pheromone"
        ],
        "states": [
            "has-food-in-nest",
            "has-food-out-nest",
            "food-available-in-nest",
            "food-available-out-nest",
            "food-not-available"
        ],
        "episode_ticks" : 500,
        "penalty": -1,
        "reward" : 100,
        "reward_type": "reward_nest_food_punish_wandering_time_incremental"
    }

    params_visualizer = {
      "FPS": 10,
      "SHADE_STRENGTH": 10,
      "SHOW_CHEM_TEXT": True,
      "SHOW_FOOD_TEXT": True,
      "CLUSTER_FONT_SIZE": 12,
      "CHEMICAL_FONT_SIZE": 8,
      "FOOD_FONT_SIZE" : 8,
      "gui": True,
      
      "sniff_threshold": 0.9,
      "PATCH_SIZE": 20,
      "TURTLE_SIZE": 10,
    }

    EPISODES = 2
    SEED = 42

    np.random.seed(SEED)
    env = Ants(SEED, **params)
    env_vis = AntsVisualizer(env.W_pixels, env.H_pixels, **params_visualizer)

    rew_d = {}
    ticks_per_episode = {}

    start_time = time.time()
    for ep in tqdm(range(1, EPISODES + 1), desc="Episode"):
        with open('out.txt', 'a') as f:
            print("Episode: ", ep, file = f)
        env.reset()
        rew_d[ep] = {}
        ticks_per_episode[ep] = {}
        for tick in tqdm(range(1, params['episode_ticks'] + 1), desc="Tick", leave=False):
            with open('out.txt', 'a') as f:
                print("Tick: ", tick, file = f)

            for agent in env.agent_iter(max_iter=params["learner_population"]):

                if env.all_learners_done:
                    break

                observation, reward, termination, truncation, info = env.last(agent)

                with open('out.txt', 'a') as f:
                        print("Agent:", agent, "Reward: ", reward, file = f)

                with open("prova.txt", 'a') as f1:
                    print(ep, tick, agent, reward, sep = ",", file = f1)
                
                if agent in ticks_per_episode[ep].keys():
                    ticks_per_episode[ep][agent] += 1
                else:
                    ticks_per_episode[ep][agent] = 1

                if agent in rew_d[ep].keys():
                    rew_d[ep][agent] += reward
                else:
                    rew_d[ep][agent] = reward

                agent_has_food = observation[0]
                patch_has_food = observation[1]
                agent_in_nest = observation[2]

                if termination or truncation:
                    action = None
                else:
                    if agent_has_food == 1:
                        if agent_in_nest:
                            action = 5 # drop food
                        else:
                            #action = 7 # lay food pheromone and head back to nest
                            action = 3 # 3 7
                    else:
                        if patch_has_food == 1 and agent_in_nest == 0:
                            action = 4 # take food
                        else:
                            #action = 2 # follow food pheromone
                            action = 0 # 0 2
                
                env.step(action)

            if env.all_learners_done:
                break
                
            env_vis.render(
                env.patches,
                env.learners,
                env.turtles
            )

        sum_nest = 0
        sum_piles = 0
        for coords, info in env.patches.items():
            if info["nest"]:
                sum_nest += info["food"]
            else:
                sum_piles += info["food"]
        #print(f"nest: {sum_nest} piles: {sum_piles}")
        
        print("Average reward:", round((sum({agent: rew/ticks_per_episode[ep][agent] for agent, rew in rew_d[ep].items()}.values()) / params["learner_population"]), 2))
        #print("Average reward:", round((sum(rew_d[ep].values()) / params["episode_ticks"]) / params["learner_population"], 2))
        with open('out.txt', 'a') as f:
            #print("Average reward:", round((sum(rew_d[ep].values()) / params["episode_ticks"]) / params["learner_population"], 2), file = f)
            print("Average reward:", round((sum({agent: rew/ticks_per_episode[ep][agent] for agent, rew in rew_d[ep].items()}.values()) / params["learner_population"]), 2), file = f)
            print({key : value + env.searching_food_ticks[key] for key, value in env.carrying_food_ticks.items()})
            print(ticks_per_episode)

    print("Total time = ", time.time() - start_time)
    env.close()
    env_vis.close()

if __name__ == '__main__':
    main()