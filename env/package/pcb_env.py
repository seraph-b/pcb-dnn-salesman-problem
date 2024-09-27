import gym
import numpy as np
from .pcb_utils import generate_board
from .pcb_utils import generate_display_model
from math import sqrt
from gym.envs.classic_control.rendering import SimpleImageViewer

class PCBEnv(gym.Env):
    metadata = {'render.modes' : ['human','rgb_array']}
    def __init__(self,
                 size=(100,100),
                 scale=8,
                 n_obstacles=6,
                 obstacle_variance=(7,12),
                 size_variance=(0,10)):
        # Check inputs
        assert obstacle_variance[0] <= obstacle_variance[1] and obstacle_variance[0] > 0
        assert size_variance[0] <= size_variance[1] and size_variance[0] >= 0
        assert size[0] > 0 and size[1] > 0
        
        # Space-defining variables
        self.board             = None
        self.size              = size
        self.size_variance     = size_variance
        self.n_obstacles       = n_obstacles
        self.obstacle_variance = obstacle_variance

        # gym.Env required variables
        self.action_space      = gym.spaces.discrete.Discrete(len(ACTIONS))
        self.observation_space = gym.spaces.Box(low=0,
                                                high=5,
                                                shape=(self.size[0],self.size[1]),
                                                dtype=np.uint8)

        # Rendering
        self.window = None
        self.scale  = scale
        
        # Rewards
        self.reward_max          = 100.0
        self.reward_step_empty   = -1.1
        self.reward_collision    = -100.0
        self.reward_range        = (self.reward_collision,self.reward_max)

        # Utilities
        self.agent_position = None
        self.goal_position  = None
        self.last_direction = None
        self.n_turns        = 0

        # Set up the board for the first time
        self.reset()

    def reset(self):
        """

        Generates a random board with n obstacles of varying size on a board
        of varying dimensions as defined in __init__ and randomly places the
        agent and goal.

        :return: A 2D numpy array containing the environment model.
        
        """
        self.last_direction  = None
        self.n_turns         = 0
        self.board,self.agent_position,self.goal_position = generate_board(self.size,
                                                                           self.size_variance,
                                                                           self.n_obstacles,
                                                                           self.obstacle_variance)
        return self.board

    def step(self,action):
        """

        Takes a step in a specified direction.

        :param action: Integer in range of the length of ACTIONS.
        :return: The updated mode, reward, done status, and info.

        """
        done = self._is_final_action(action)
        reward = self._act(action)
        return self.board,reward,done,None

    def render(self,mode='human'):
        """

        Reshapes the environment model to be scaled up and contain color
        information or opens a window and displays the scaled up model.

        :param mode: Render mode from metadata.render.modes.
        :return:
        
        """
        display_model = generate_display_model(self.board,self.size,self.scale)
        if mode == 'rgb_array':
            return display_model
        elif mode == 'human':
            if not self.window:
                self.window = SimpleImageViewer(maxwidth=1000)
            self.window.imshow(display_model)
            return self.window.isopen
        else:
            super(PCBEnv,self).render(mode=mode)

    def _act(self,action):
        """

        Evaluates whether the agent ran into an obstacle, reached the goal,
        or stepped onto an empty space. This function updates the
        environment and returns the appropriate reward.

        :param action: Integer in range of the length of ACTIONS.
        :return: Reward for the last action.
        
        """
        next_position = tuple(sum(x) for x in zip(self.agent_position,ACTION_MAP[action]))
        if next_position[0] == 100 or next_position[1] == 100 or self.board[next_position] in [1,2,3]:
            return self.reward_collision*self._reward_proximity_scale(next_position)

        self.board[self.agent_position] = 3
        self.agent_position = next_position
        
        if self.board[next_position] == 4:
            self.board[next_position] = 3
            return self._calculate_goal_reward()

        self.board[next_position] = 5
        return self.reward_step_empty+self._reward_proximity_offset(next_position)

    def _reward_proximity_offset(self,next_position):
        """

        Calculates the distance from goal and returns an offset to reward 
        to direct agents.

        :param next_position: A tuple of integers representing the next
        position of agent.
        :return: Reward offset.

        """
        distance = sqrt((self.goal_position[0]-next_position[0])**2+(self.goal_position[1]-next_position[1])**2)
        reward_offset = 0
        if distance <= 50:
            reward_offset = 1-(distance/50)
        return reward_offset
    
    def _reward_proximity_scale(self,next_position):
        """

        Calculates the distance from goal and returns a scale to reward 
        to direct agents.

        :param next_position: A tuple of integers representing the next
        position of agent.
        :return: Reward scale.

        """
        distance = sqrt((self.goal_position[0]-next_position[0])**2+(self.goal_position[1]-next_position[1])**2)
        reward_scale = 1
        if distance <= 50:
            reward_scale = 0.5+(distance/100)
        return reward_scale

    def _is_final_action(self,action):
        """

        Evaluates whether the agent is about step on an occupied space.

        :param action: Integer in range of the length of ACTIONS.
        :return: True if next space is occupied, False otherwise.
        
        """
        next_position = tuple(sum(x) for x in zip(self.agent_position,ACTION_MAP[action]))
        if next_position[0] == 100 or next_position[0] == -1 or next_position[1] == 100 or next_position[1] == -1:
            return True
        next_space = self.board[next_position]
        return next_space in [1,2,3,4]

    def _calculate_goal_reward(self):
        """

        Calculates the final reward as a fraction of the maximum reward
        minus one and the number of turns made on the route.

        :return: Reward for reaching the goal.
        
        """
        if self.n_turns == 0:
            return self.reward_max
        else:
            return (self.reward_max-1)/self.n_turns

# Global dictionaries
ACTIONS = {
    0 : 'north',
    1 : 'east',
    2 : 'south',
    3 : 'west'
}

ACTION_MAP = {
    0 : (-1, 0),
    1 : ( 0, 1),
    2 : ( 1, 0),
    3 : ( 0,-1)
}
