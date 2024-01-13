import gym
from gym import spaces
import numpy as np

class ShootingAirplaneEnv(gym.Env):
    metadata = {"render_modes": ["text"], "render_fps": 4}

    def __init__(self, render_mode=None, size=8):
        self.size = size  # The size of the square grid

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # Observation: for each cell in size x size matrix, three states involving  
        # { unseen (0), hit (1), miss (2) }
        self.observation_space = spaces.Box(low=0, high=255, shape=(size, size, 1), dtype=np.uint8)

        # We have size x size actions, (row, column) index to shoot
        self.action_space = spaces.MultiDiscrete([size, size])

        # Direction: 0
        #                 (0, -1)
        # (-2, 0) (-1, 0) (0,  0) (1, 0) (2, 0)
        #                 (0,  1)
        #         (-1, 2) (0,  2) (1, 2)
        # Direction: 1
        #        (-1, -2) (0, -2) (1, -2)
        #                 (0, -1)
        # (-2, 0) (-1, 0) (0,  0) (1,  0) (2, 0)
        #                 (0,  1)
        # Direction: 2
        #                 (0, -2)
        #                 (0, -1)        (2, -1)
        #         (-1, 0) (0,  0) (1, 0) (2,  0)
        #                 (0,  1)        (2,  1)
        #                 (0,  2)
        # Direction: 3
        #                 (0, -2)
        #(-2, -1)         (0, -1)       
        #(-2,  0) (-1, 0) (0,  0) (1, 0)
        #(-2,  1)         (0,  1)       
        #                 (0,  2)

        self._relative_pos = np.array([
            [(0, -1), (-2, 0), (-1, 0), (0, 0), (1, 0), (2, 0), (0, 1), (-1, 2), (0, 2), (1, 2)],
            [(0, 1), (-2, 0), (-1, 0), (0, 0), (1, 0), (2, 0), (0, -1), (-1, -2), (0, -2), (1, -2)],
            [(0, -2), (0, -1), (2, -1), (-1, 0), (0, 0), (1, 0), (2, 0), (0, 1), (2, 1), (0, 2)],
            [(0, -2), (0, -1), (-2, -1), (1, 0), (0, 0), (-1, 0), (-2, 0), (0, 1), (-2, 1), (0, 2)],
        ], dtype='int32')


    def _get_obs(self):
        return self._board

    def _get_info(self):
        return { 'prob': 1.0, 'action_mask': np.array((self._board != 0) == False, dtype='int32').reshape((self.size,self.size))}

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # initialize board (observation)
        self._board = np.zeros([self.size, self.size, 1], dtype=np.uint8)

        while True:
            self._hidden_airplane = np.ones([self.size, self.size, 1], dtype=np.uint8) * 255

            # the direction of airplain from 0 to 3
            direc = self.np_random.integers(0, 4)

            # Choose the airplain's center uniformly at random
            center = self.np_random.integers(1, self.size-1, size=(2,))

            out_of_board = False
            for rel_x, rel_y in self._relative_pos[direc]:
                if center[0] + rel_x < 0 or center[0] + rel_x >= self.size:
                    out_of_board = True
                    break

                if center[1] + rel_y < 0 or center[1] + rel_y >= self.size:
                    out_of_board = True
                    break

                self._hidden_airplane[center[0] + rel_x, center[1] + rel_y, 0] = 1

            if not out_of_board:
                break

        return self._get_obs(), self._get_info()

    def step(self, action):
        assert action[0] >= 0 and action[0] < self.size
        assert action[1] >= 0 and action[1] < self.size

        if self._board[action[0], action[1], 0] == 0:
            # if the cell on the airplane,
            if self._hidden_airplane[action[0], action[1], 0] == 1:
                self._board[action[0], action[1], 0] = 1
                reward = 1
            # missed
            else:
                self._board[action[0], action[1], 0] = 2
                reward = -1

        # should not fall on here, but..
        else:
            reward = -1

        # An episode is done iff all ten cells of airplain hit
        hits = np.sum(self._board == self._hidden_airplane)
        terminated = True if hits == 10 else False

        observation = self._get_obs()
        info = self._get_info()

        # observation, reward, if terminated, if truncated, info
        # truncated: true if episode truncates due to a time limit
        return observation, reward, terminated, False, info

    def render(self):
        if self.render_mode == "text":
            return self._render_board()

    def _render_board(self):
        str = ''
        chars = [' ', 'H', 'M']
        for row in range(self.size):
            for col in range(self.size):
                str += chars[self._board[row, col, 0]]
            str += ' | '
            for col in range(self.size):
                str += 'H' if self._hidden_airplane[row, col, 0] == 1 else ' '
            str += "\n"

        print(str)
