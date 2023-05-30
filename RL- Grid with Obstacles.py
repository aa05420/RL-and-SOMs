import pygame
import numpy as np
import random

#grid parameters
N = 10
cell_size = 50
grid = np.zeros((N, N))

#hyper parameters
gamma = 0.9
max_steps = 100
num_episodes = 1000
temperature = 0.5


# function to create obstacles and reward cells 
def initialize_grid():
    
    for i in range(random.randint(1,10)):
        x = random.randint(0,N-1)
        y = random.randint(0,N-1)
        grid[x,y] = -100

    grid[N-1,N-1] = 100
    return grid

grid = initialize_grid()

#colors for visualization
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLACK = (0,0,0)


#initializing value function
values = np.zeros((N, N, 4))


#action selection using Boltzmann distribution
def boltzmann_action(probs, temperature):
    scaled_probs = probs / temperature
    exp_probs = np.exp(scaled_probs)
    exp_sum = np.sum(exp_probs)
    action_probs = exp_probs / exp_sum
    action = np.random.choice(range(4), p = action_probs)
    return action

# Define function for running an episode
def run_episode(grid, values, gamma, max_steps, temperature):
    
    x, y = 0, 0
    episode_reward = 0
    step = 0
    
    # running episode until termination
    while (grid[x,y]!= -1 and grid[x,y] != 100) and step < max_steps:
        # selecting action using Boltzmann distribution
        probs = np.array([values[x, y, 0], values[x, y, 1], values[x, y, 2], values[x, y, 3]])
        action = boltzmann_action(probs, temperature)

        # observe reward for the next state based on action
        
        if action == 0 and y > 0:
            x_next, y_next = x, y-1     #up 
        elif action == 1 and y < N-1:
            x_next, y_next = x, y+1     #down
        elif action == 2 and x > 0:
            x_next, y_next = x-1, y     #left
        elif action == 3 and x < N-1:
            x_next, y_next = x+1, y     #right
        else:
            x_next, y_next = x, y   


        reward = grid[x_next,y_next]

        # updating the value function
        target = reward + gamma * np.max([values[x_next, y_next, 0], values[x_next, y_next, 1], values[x_next, y_next, 2], values[x_next, y_next, 3]])
        values[x, y, action] += 0.1 * (target - values[x, y, action])

        
        episode_reward += reward
        x, y = x_next, y_next
        step += 1
    return episode_reward


#pygame initialization
pygame.init()
screen = pygame.display.set_mode((N*cell_size, N*cell_size))

#arrow constants
ARROW_SIZE = 15
ARROW_THICKNESS = 3
ARROW_COLOR = BLACK

#arrow angles for each action
UP_ANGLE = 90
DOWN_ANGLE = 270
LEFT_ANGLE = 180
RIGHT_ANGLE = 0

#draw arrows in each cell
def draw_arrows(values):
    for i in range(N):
        for j in range(N):
            if grid[i,j] != -100 and grid[i,j]!=100:
                x = i * cell_size + cell_size // 2
                y = j * cell_size + cell_size // 2

                #action with the highest value
                max_action = np.argmax(values[i,j,:])
                print(max_action)
                #draw an arrow in the corresponding direction
                if max_action == 0 and j > 0: # up
                    pygame.draw.line(screen, ARROW_COLOR, (x, y), (x, y-ARROW_SIZE), ARROW_THICKNESS)
                    pygame.draw.polygon(screen, ARROW_COLOR, [(x, y-ARROW_SIZE), (x-ARROW_SIZE/2, y-ARROW_SIZE/2), (x+ARROW_SIZE/2, y-ARROW_SIZE/2)])
                elif max_action == 1 and j < N-1: # down
                    pygame.draw.line(screen, ARROW_COLOR, (x, y), (x, y+ARROW_SIZE), ARROW_THICKNESS)
                    pygame.draw.polygon(screen, ARROW_COLOR, [(x, y+ARROW_SIZE), (x-ARROW_SIZE/2, y+ARROW_SIZE/2), (x+ARROW_SIZE/2, y+ARROW_SIZE/2)])
                elif max_action == 2 and i > 0: # left
                    pygame.draw.line(screen, ARROW_COLOR, (x, y), (x-ARROW_SIZE, y), ARROW_THICKNESS)
                    pygame.draw.polygon(screen, ARROW_COLOR, [(x-ARROW_SIZE, y), (x-ARROW_SIZE/2, y-ARROW_SIZE/2), (x-ARROW_SIZE/2, y+ARROW_SIZE/2)])
                elif max_action == 3 and i < N-1: # right
                    pygame.draw.line(screen, ARROW_COLOR, (x, y), (x+ARROW_SIZE, y), ARROW_THICKNESS)
                    pygame.draw.polygon(screen, ARROW_COLOR, [(x+ARROW_SIZE, y), (x+ARROW_SIZE/2, y-ARROW_SIZE/2), (x+ARROW_SIZE/2, y+ARROW_SIZE/2)])

running = True
while running:
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    screen.fill(WHITE)
    for episode in range(num_episodes): 
        reward = run_episode(grid, values, gamma, max_steps, temperature)
        if episode == num_episodes -1:
            running = False
        # drawing the grid
        for i in range(N):
            for j in range(N):
                cell_color = WHITE
                if grid[i, j] == -100:
                    cell_color = RED

                elif grid[i, j] == 100:
                    cell_color = GREEN
                pygame.draw.rect(screen, cell_color, (i*cell_size, j*cell_size, cell_size, cell_size))
                pygame.draw.rect(screen, (0,0,0), (i*cell_size, j*cell_size, cell_size, cell_size), 1) #borders
        
        # draw arrows pointing to the best action in each cell
        draw_arrows(values)
        # update the display
        pygame.display.update()



while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
           