"""modified from YuriyGuts/snake-ai-reinforcement"""
import numpy as np
import torch
import pygame
from envs.snakegame.common import Cell, Direction
from envs.snakegame.snake_multi_agent import Action


class Color:
    BACKGROUND = (0, 0, 0)
    GRID = (56, 56, 56)
    CELL = {
        Cell.EMPTY: BACKGROUND,
        Cell.FRUIT: (223, 7, 22),
        Cell.WALL: (32, 32, 32),

        Cell.HEAD[0]: (104, 255, 0),
        Cell.BODY[0]: (104, 255, 0),

        Cell.HEAD[1]: (255, 191, 0),
        Cell.BODY[1]: (255, 191, 0),

        Cell.HEAD[2]: (255, 0, 92),
        Cell.BODY[2]: (255, 0, 92),

        Cell.HEAD[3]: (0, 111, 255),
        Cell.BODY[3]: (0, 111, 255),
    }


class Timer:
    FPS_LIMIT = 10

    def __init__(self):
        self.reset()

    def reset(self):
        self.fps_clock = pygame.time.Clock()
        self.start_time = pygame.time.get_ticks()

    def tick(self):
        self.fps_clock.tick(self.FPS_LIMIT)

    @property
    def time(self):
        return pygame.time.get_ticks() - self.start_time


class ML2PythonGUI:

    SNAKE_CONTROL_KEYS = [
        pygame.K_UP,
        pygame.K_LEFT,
        pygame.K_DOWN,
        pygame.K_RIGHT
    ]

    SNAKE_DIRECTIONS = [
        Direction.NORTH,
        Direction.EAST,
        Direction.SOUTH,
        Direction.WEST,
    ]

    def __init__(self, env, args):
        pygame.init()
        
        self.env = env
        self.human = args.human
        self.cell_size = args.cell_size
        self.device = args.device
        self.human = True if 'human' in args.mode else False
        self.mode = args.mode 
    
        self.screen = pygame.display.set_mode((
            self.env.field.size[0]*self.cell_size,
            self.env.field.size[1]*self.cell_size
        ))
        self.screen.fill(Color.BACKGROUND)
        pygame.display.set_caption('ML2 Python')

        self.timer = Timer()
        self.reset()

    def reset(self):
        self.timer.reset()
        return self.env.reset()

    def run(self, policy):
        done = False
        obs = self.reset()
        prev_obs = obs

        while not done:
            self.render()
            pygame.display.update()

            action = Action.IDLE
            for event in pygame.event.get():
                if self.human and event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT:
                        action = Action.LEFT
                    elif event.key == pygame.K_RIGHT:
                        action = Action.RIGHT
                    elif event.key == pygame.K_UP:
                        action = Action.UP
                    elif event.key == pygame.K_DOWN:
                        action = Action.DOWN
                    elif event.key == pygame.K_ESCAPE:
                        raise
                elif event.type == pygame.QUIT:
                    break

            timed_out = self.timer.time >= self.timer.FPS_LIMIT
            made_move = self.human and action != Action.IDLE
            if timed_out or made_move:
                self.timer.reset()

                state = np.concatenate([obs, prev_obs], axis=1)
                state = torch.tensor(state).to(self.device)
                
                q = policy(state.float())

                actions = torch.argmax(q, dim=1)
                actions = actions.cpu().numpy()
                if self.human:
                    actions[0] = action
                
                prev_obs = obs
                obs, _, dones, scores = self.env.step(actions)
                done = all(dones)
                self.timer.tick()

        print (scores)

        
    def render(self):
        for x in range(self.env.field.size[0]):
            for y in range(self.env.field.size[1]):
                # Draw grid lines
                surface = pygame.display.get_surface()
                pygame.draw.line(
                    self.screen,
                    Color.GRID,
                    (x*self.cell_size, 0),
                    (x*self.cell_size, surface.get_height())
                )
                pygame.draw.line(
                    self.screen,
                    Color.GRID,
                    (0, y*self.cell_size),
                    (surface.get_width(), y*self.cell_size)
                )

                # Draw cells
                rect = pygame.Rect(
                    x*self.cell_size,
                    y*self.cell_size,
                    self.cell_size,
                    self.cell_size
                )

                cell = self.env.field[x, y]
                color = Color.CELL[cell]
                if cell == Cell.WALL:
                    pygame.draw.rect(self.screen, color, rect, 1)
                    padding = self.cell_size // 5
                    rect = rect.inflate((-padding, -padding))
                    pygame.draw.rect(self.screen, color, rect)
                else:
                    pygame.draw.rect(self.screen, color, rect)
