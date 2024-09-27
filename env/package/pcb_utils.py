import numpy as np
from random import choice
from random import randint

def get_random_position(board):
    rand_position = choice(np.argwhere(board==0))
    return (rand_position[0],rand_position[1])

def fill_obstacle(board,
                  size,
                  position,
                  variance):
    offset = (randint(variance[0],variance[1]),randint(variance[0],variance[1]))
    if offset[0] >= size[0]:
        offset[0] = size[0]-1
    if offset[1] >= size[1]:
        offset[1] = size[1]-1
        
    board[position[0]:
          position[0]+offset[0],
          position[1]:
          position[1]+offset[1]] = 2

def fill_empty_space(board,
                     size,
                     variance):
    board[randint(variance[0],variance[1]):
          size[0]-randint(variance[0],variance[1]),
          randint(variance[0],variance[1]):
          size[1]-randint(variance[0],variance[1])] = 0

def generate_board(size,
                   size_var,
                   n_obstacles,
                   ob_var):
    board = np.ones(size,dtype=np.uint8)
    fill_empty_space(board,size,size_var)
    for _ in range(n_obstacles):
        position = get_random_position(board)
        fill_obstacle(board,size,position,ob_var)
    goal = get_random_position(board)
    board[goal] = 4
    agent = get_random_position(board)
    board[agent] = 5
    return board,agent,goal

def generate_display_model(board,size,scale):
    model = np.zeros(shape=(size[0]*scale,size[1]*scale,3),dtype=np.uint8)
    for y in range(model.shape[0]):
        for x in range(model.shape[1]):
            original_position = (int(y/scale),int(x/scale))
            model[y,x,:] = np.array(ITEM_RGB_MAP[board[original_position]])
    return model

# Global Dictionaries
ITEM_RGB_MAP = {
    0 : [101,101,101],
    1 : [  0,  0,  0],
    2 : [ 44, 44, 44],
    3 : [204, 88,  0],
    4 : [  7,162,255],
    5 : [ 64,198,  0]
}
