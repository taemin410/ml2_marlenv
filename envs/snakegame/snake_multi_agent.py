
# from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ..utils.multienv import MultiAgentEnv

from snakegame.common import Point, Cell, Direction
from snakegame.field import Field
from snakegame.python import Python


class SnakeGameMultiEnv(MultiAgentEnv):
    def __init__(self, init_map, players=None, full_observation=False, vision_range=10, num_agents=2):

    	#number of agent & observability initialized
 		super().__init__(num_agents=2, full_observation = full_observation)


 		#Store map and players data
        self.init_map = init_map
        self.players = players
        self.num_players = len(self.field.players)

        #Draw map on field object
        self.field = Field(init_map, players)
        self.visits = np.zeros((self.num_players, np.prod(self.field.size)))

        if self.full_observation:
        #give the whole field to the observation space
            self.observation_space = Box(
            low=0,
            high=2,
            shape=(self.num_players, 4, self.field.size[0], self.field.size[1])
        )
        else:
            # Set vision range to vision_range(nxn)
            self.observation_space = Box(
                low=0,
                high=2,
                shape=(self.num_players, 4, vision_range, vision_range)
            )
        
        self.action_space = Discrete(5)

        self.reset()

    def reset(self):
        
        #TODO: Reseting multiplayer mode with randomized input 

        #randomize snake's initial position 
        players = [
            Python(Point(np.random.randint(3, self.field.size[1]-5, 1)[0],np.random.randint(3, self.field.size[0]-5, 1)[0]), random.choice(Direction.DIRECTIONLIST), 1)
        ]

        self.field = Field(self.init_map, players)

        #initialize dones and infos 
        self.dones = np.zeros(self.num_players, dtype=bool)
        self.epinfos = {
            'step': 0,
            'scores': np.zeros(self.num_players),
            'fruits': np.zeros(self.num_players),
            'kills': np.zeros(self.num_players)
        }

        #Create fruit on one cell 
        self.fruit = self.field.get_empty_cell()
        self.field[self.fruit] = Cell.FRUIT

        return self.full_observation()

    def step(self, actions):
        assert len(actions) == self.num_players
        self.epinfos['step'] += 1
        rewards = np.zeros(self.num_players)
        
        for idx, action in enumerate(actions):
            python = self.field.players[idx]
            if not python.alive:
                continue

            # Choose action
            if python.direction == Direction.NORTH:
                if action == Action.LEFT:
                    python.turn_left()
                elif action == Action.RIGHT:
                    python.turn_right()
            elif python.direction == Direction.EAST:
                if action == Action.DOWN:
                    python.turn_right()
                elif action == Action.UP:
                    python.turn_left()
            elif python.direction == Direction.SOUTH:
                if action == Action.RIGHT:
                    python.turn_left()
                elif action == Action.LEFT:
                    python.turn_right()
            elif python.direction == Direction.WEST:
                if action == Action.UP:
                    python.turn_right()
                elif action == Action.DOWN:
                    python.turn_left()

            # Eat fruit
            if self.field[python.next] == Cell.FRUIT:
                # python.grow()
                rewards[idx] += Reward.FRUIT
                self.epinfos['fruits'][idx] += 1
                if python.head == self.fruit:
                    self.fruit = self.field.get_empty_cell()
                    self.field[self.fruit] = Cell.FRUIT

            # Or just starve
            else:
                self.field[python.tail] = Cell.EMPTY
                python.move()
                rewards[idx] += Reward.TIME

            self.field.players[idx] = python


        # Resolve conflicts
        conflicts = self.field.update_cells()
        for conflict in conflicts:
            idx = conflict[0]
            python = self.field.players[idx]
            python.alive = False
            rewards[idx] += Reward.LOSE
            self.dones[idx] = True

            # If collided with another player
            if len(conflict) > 1:
                idx = conflict[1]
                if idx != conflict[0]:
                    other = self.field.players[idx]
                    # Head to head
                    if self.field[python.head] in Cell.HEAD:
                        other.alive = False
                        rewards[idx] += Reward.LOSE
                        self.dones[idx] = True
                    # Head to body
                    else:
                        rewards[idx] += Reward.KILL
                        self.epinfos['kills'][idx] += 1

        # Check if done and calculate scores
        if self.num_players > 1 and np.sum(~self.dones) == 1:
            idx = list(self.dones).index(False)
            self.dones[idx] = True
            rewards[idx] += Reward.WIN
        
        self.epinfos['scores'] += rewards
    
        return self.full_observation(), rewards, self.dones, self.epinfos

    def full_observation(self):
        self.field.clear()
        body = np.zeros((self.num_players, *self.field.size))
        for idx in range(self.num_players):
            head_cell = np.isin(
                self.field._cells,
                Cell.HEAD[idx]
            ).astype(np.float32)
            body_cell = np.isin(
                self.field._cells,
                Cell.BODY[idx]
            ).astype(np.float32)
            body[idx] = head_cell + body_cell

        fruit = np.isin(self.field._cells, Cell.FRUIT).astype(np.float32)
        wall = np.isin(self.field._cells, Cell.WALL).astype(np.float32)

        state = np.zeros(self.observation_space.shape)
        for idx in range(self.num_players):
            if self.field.players[idx].alive:
                state[idx][0] = body[idx]
                state[idx][1] = body[idx]
                state[idx][2] = fruit
                state[idx][3] = wall
                
        return state

    def encode(self):
        self.field.clear()
        body = np.zeros((self.num_players, *self.field.size))
        for idx in range(self.num_players):
            head_cell = np.isin(
                self.field._cells,
                Cell.HEAD[idx]
            ).astype(np.float32)
            body_cell = np.isin(
                self.field._cells,
                Cell.BODY[idx]
            ).astype(np.float32)
            body[idx] = head_cell + body_cell

        fruit = np.isin(self.field._cells, Cell.FRUIT).astype(np.float32)
        wall = np.isin(self.field._cells, Cell.WALL).astype(np.float32)

        state = np.zeros(self.observation_space.shape)
        for idx in range(self.num_players):
            if self.field.players[idx].alive:
                state[idx][0] = self.get_vision(idx, body[idx])
                state[idx][1] = self.get_vision(idx, np.sum(body, axis=0) - body[idx])
                # state[idx][1] = fruit 
                state[idx][2] = self.get_vision(idx, fruit)
                state[idx][3] = self.get_vision(idx, wall)
        
        return state
    
    def get_vision(self, idx, arr):
        head = np.where(np.isin(self.field._cells, Cell.HEAD[idx]))
        h, w = self.observation_space.shape[2:]
        x1 = int(head[1] - w//2)
        x2 = int(head[1] + w//2)
        y1 = int(head[0] - h//2)
        y2 = int(head[0] + h//2)
        
        arr = arr[max(y1,0):y2, max(0, x1):x2]

        if y1 < 0:
            arr = np.pad(arr, ((-y1, 0), (0, 0),), mode="constant")
        elif y2 > self.field.size[0]:
            arr = np.pad(arr, ((0, y2 - self.field.size[0]), (0, 0),),mode="constant")
        if x1 < 0:
            arr = np.pad(arr, ((0, 0), (-x1, 0)),mode="constant")
        elif x2 > self.field.size[1]:
            arr = np.pad(arr, ((0, 0), (0,x2 - self.field.size[1])),mode="constant")

        return arr
 
    
    def render(self):
        print(self.field)
