import gym
from gym import spaces
import numpy as np

class OmokSinglePlayerEnv(gym.Env):
    metadata = {"render_modes": ["text"], "render_fps": 4, "max_steps": 100 }

    def __init__(self, render_mode=None, size=15):
        self.size = size  # The size of the square grid

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # Observation: for each cell in size x size matrix, three states involving  
        # 0: empty, 1: player 1, 2: player 2
        self.observation_space = spaces.Box(low=0, high=255, shape=(size, size, 1), dtype=np.uint8)

        # We have 2 x size x size actions, representing stone's color x row idx x col idx
        # the action results in putting a stone of the color at (row, column) position
        self.action_space = spaces.MultiDiscrete([size, size])

        # to examine n-mok
        self.offsets = np.array([-4, -3, -2, -1, 0, 1, 2, 3, 4], dtype=np.int32)
        self.neighbors = [
            (-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)
        ]

        # count the number of steps (i.e., stones on the board)
        self.n_step = 0

    def _get_obs(self):
        return self._board

    def _get_info(self):
        return {# deterministic state transition
                'prob': 1.0,
                # step counting
                'steps': self.n_step,
                # return an ndarray of shape (size, size) each elem of which denotes if empty
                'action_mask': np.array(
                    (self._board != 0) == False, dtype='int32').reshape((self.size, self.size))
               }

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # initialize board (observation)
        self._board = np.zeros([self.size, self.size, 1], dtype=np.uint8)
        
        # player 1 = env, player 2 = xternal agent
        self.n_step = 1
        self.player = 1
        self.enemy = 2

        # to return the locations for the env player to put a stone
        self.next_move = list()
        
        # start a game by putting a stone at the center for player 1
        self._board[7, 7, 0] = 1

        return self._get_obs(), self._get_info()
    
    # return True if player wins by locating a stone at (row, col)
    def wins(self, player, row, col):
        return self.mok(player, 5, row, col)

    # when player put a stone at (row, col),
    #   examine if there exists a n-mok including the last stone
    #   and return true
    #   note: you should call it right after updating board about the last stone
    def mok(self, player, n, row, col, left_free=False, right_free=False):
        assert (not (left_free and right_free)) or (n <= 3), \
            "n should be at most 3 to check if both sizes are free"
        assert (not ((not left_free and right_free) or (left_free and not right_free))) or (n <= 4), \
            "n should be at most 4 to check if a single size only is free"

        if left_free:
            n += 1
        if right_free:
            n += 1
        
        board_arr = self._board.reshape((-1,))
        idx = row*self.size + col

        player_num = np.ones((n,)) * player
        if left_free:
            player_num[0] = 0
        if right_free:
            player_num[n-1] = 0

        if_any = False

        for l in range(n):
            # horizontal
            if (col + l+1-n < 0) or (col + l+1 >= self.size):
                continue

            if all([ board_arr[idx + offset] == p for p, offset in zip(player_num, range(l+1-n, l+1)) ]):
                if left_free:
                    self.next_move.append(idx + l+1-n)
                if right_free:
                    self.next_move.append(idx + l)
                
                if_any = True
            
        for l in range(n):    
            # vertical
            if (row + l+1-n < 0) or (row + l+1 >= self.size):
                continue

            if all([board_arr[idx + offset*self.size] == p for p, offset in zip(player_num, range(l+1-n, l+1)) ]):
                if left_free:
                    self.next_move.append(idx + (l+1-n)*self.size)
                if right_free:
                    self.next_move.append(idx + l*self.size)
                
                if_any = True
            
        for l in range(n):    
            # left to right diagonal
            if (row + l+1-n < 0) or (row + l+1 >= self.size):
                continue

            if (col + l+1-n < 0) or (col + l+1 >= self.size):
                continue

            if all([board_arr[idx + offset*self.size - offset] == p for p, offset in zip(player_num, range(l+1-n, l+1)) ]):
                if left_free:
                    self.next_move.append(idx + (l+1-n)*self.size - (l+1-n))
                if right_free:
                    self.next_move.append(idx + l*self.size - l)
                
                if_any = True
            
        for l in range(n):    
            # right to left diagonal
            if (row + l+1-n < 0) or (row + l+1 >= self.size):
                continue

            if (col + l+1-n < 0) or (col + l+1 >= self.size):
                continue

            if all([board_arr[idx + offset*self.size + offset] == p for p, offset in zip(player_num, range(l+1-n, l+1)) ]):
                if left_free:
                    self.next_move.append(idx + (l+1-n)*self.size + (l+1-n))
                if right_free:
                    self.next_move.append(idx + l*self.size + l)
                
                if_any = True
            
        return if_any
    
    # priority 2: if no defence, the env player will lose for sure
    def defence(self, row, col):
        self.next_move = list()

        if self.mok(self.enemy, 3, row, col, left_free=True, right_free=True):
            self._board[self.next_move[0] // self.size, self.next_move[0] % self.size] = self.player
            return True

        if self.mok(self.enemy, 4, row, col, left_free=True, right_free=False):
            self._board[self.next_move[0] // self.size, self.next_move[0] % self.size] = self.player
            return True

        if self.mok(self.enemy, 4, row, col, left_free=False, right_free=True):
            self._board[self.next_move[0] // self.size, self.next_move[0] % self.size] = self.player
            return True
        return False
    
    # priority 1: the env player can win for sure
    def attack_for_5mok(self):
        self.next_move = list()

        for i in range(self.size):
            for j in range(self.size):
                if self._board[i, j] == self.player:
                    if self.mok(self.player, 4, i, j, left_free=True, right_free=False) or self.mok(self.player, 4, i, j, left_free=False, right_free=True):
                        self._board[self.next_move[0] // self.size, self.next_move[0] % self.size] = self.player
                        return True
        return False
    
    # priority 3: the env player can win for sure with the subsequent two moves
    def attack_for_4mok(self):
        self.next_move = list()

        for i in range(self.size):
            for j in range(self.size):
                if self._board[i, j] == self.player:
                    if self.mok(self.player, 3, i, j, left_free=True, right_free=True):
                        self._board[self.next_move[0] // self.size, self.next_move[0] % self.size] = self.player
                        return True
        return False
    
    # priority 4: simply, put a stone adjacent
    def attack_adj(self):
        self.next_move = list()

        for i in range(self.size):
            for j in range(self.size):
                if self._board[i, j] == self.player:
                    for rr, cc in self.neighbors:
                        if i + rr < 0 or i + rr >= self.size:
                            continue
                        if j + cc < 0 or j + cc >= self.size:
                            continue
                        if self._board[i+rr, j+cc] == 0:
                            self.next_move.append((i+rr)*self.size + j+cc)
        
        if len(self._board) > 0:
            idx = np.random.choice(self.next_move, size=1)[0]
            self._board[idx // self.size, idx % self.size] = self.player
            return True
        return False
    
    def attach_rand(self):
        self.next_move = list()

        for i in range(self.size):
            for j in range(self.size):
                if self._board[i, j] == 0:
                    self.next_move.append((i)*self.size + j)
        
        if len(self._board) > 0:
            idx = np.random.choice(self.next_move, size=1)[0]
            self._board[idx // self.size, idx % self.size] = self.player
            return True
        return False

    # take an action of enemy (i.e., external agent):
    # input -> a tuple of (row idx, col idx) to locate a stone
    def step(self, action):
        assert action[0] >= 0 and action[0] < self.size
        assert action[1] >= 0 and action[1] < self.size
        assert self._board[action[0], action[1]] == 0
        
        # put a stone of the current player
        self._board[action[0], action[1]] = self.enemy
        
        reward = 0.
        terminated = self.wins(self.enemy, action[0], action[1])
        # the agent wins -> reward = 10
        if terminated:
            reward = 10.
        else:
            if self.mok(self.enemy, 4, action[0], action[1]):
                reward = 1.
            elif self.mok(self.enemy, 3, action[0], action[1]):
                reward = 1.

            while True:
                # priority 1: attack to make 5mok
                #if self.attack_for_5mok():
                #    reward = -10.
                #    terminated = True
                #    break

                # priority 2: defence enemy's 3mok and 4mok -> immediately, put a stone to defence
                if self.defence(action[0], action[1]):
                    reward += -1.
                    break

                # priority 3: attack to make 4mok
                #if self.attack_for_4mok():
                #    reward = -5.
                #    break

                # priority 4
                if self.attack_adj():
                    reward += -1.
                    break

                # priority 5
                if self.attack_rand():
                    reward += -1.
                    break
        
        self.n_step += 1

        truncated = (self.n_step >= self.metadata['max_steps'])
        terminated = (truncated or terminated)
        observation = self._get_obs()
        info = self._get_info()

        # observation, reward, if terminated, if truncated, info
        # truncated: true if episode truncates due to a time limit
        return observation, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "text":
            return self._render_board()

    def _render_board(self):
        ret = ' '
        colnum = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0', '1', '2', '3', '4']
        chars = [' ', 'O', '@']
        
        # first, column numbers
        for col in range(self.size):
            ret += ' ' + colnum[col]
        ret += "\n"
        
        ret += " " + "+-" * self.size + "+\n"
        for row in range(self.size):
            ret += colnum[row] + "+"
            for col in range(self.size):
                ret += chars[self._board[row, col, 0]] + "|"
            ret += "\n"
            
            ret += " " + "+-" * self.size + "+\n"
            
        print(ret)
