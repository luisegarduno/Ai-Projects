####################################################################
# Packages & Libraries #############################################
####################################################################

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from numpy import random
from matplotlib import colors
# %matplotlib inline
# %config InlineBackend.figure_format = 'retina'

####################################################################
# Helper Functions ################################################# 
####################################################################

# -------------------------------------------------------------------------
""" Read & Load Maze File """
# -------------------------------------------------------------------------
def selectMaze(maze):
    myFile = open(maze, "r")
    maze_str = myFile.read()
    return maze_str

# -------------------------------------------------------------------------
""" Convert a maze as a string into a 2D numpy array """
# -------------------------------------------------------------------------
def convert_Maze2Array(maze_str):
    maze = maze_str.split('\n')
    maze = np.array([[tile for tile in row] for row in maze if len(row) > 0])
    return maze

# -------------------------------------------------------------------------
# https://stackoverflow.com/questions/43971138/python-plotting-colored-grid-based-on-values
""" Display a maze (Numpy Array) """
# -------------------------------------------------------------------------
def show_maze(maze, fontsize = 10):
    cmap = colors.ListedColormap(['white', 'black', 'blue', 'green', 'red', 'gray', 'orange'])

    # make a deep copy first so the original maze is not changed
    maze = np.copy(maze)

    goal = find_pos(maze, 'G')
    start = find_pos(maze, 'S')

    # Converts all tile types to integers
    maze[maze == ' '] = 0
    maze[maze == 'X'] = 1 # Wall
    maze[maze == 'S'] = 2 # Start
    maze[maze == 'G'] = 3 # Goal
    maze[maze == 'P'] = 4 # Position/Final path
    maze[maze == '.'] = 5 # Visited Squares
    maze[maze == 'F'] = 6 # Frontier

    # Converts all string values to integers
    maze = maze.astype(np.int)

    fig, ax = plt.subplots()
    ax.imshow(maze, cmap = cmap, norm = colors.BoundaryNorm(list(range(cmap.N + 1)), cmap.N))

    plt.text(start[1], start[0], "S", fontsize=fontsize, color="white",
             horizontalalignment = 'center', verticalalignment = 'center')

    plt.text(goal[1], goal[0], "G", fontsize=fontsize, color="white",
             horizontalalignment = 'center', verticalalignment = 'center')

    plt.show()

# -------------------------------------------------------------------------
""" Find Start/Goal in Maze. Caution: there's no error checking
    [Parameters]: {maze: 2D Array w/ Char's | what: Objective/Letter to find {S start, G goal}
    [Returns]: a tupple (x, y) for the found position. """
# -------------------------------------------------------------------------
def find_pos(maze, what="S"):
    position = np.where(maze == what)
    return(tuple([position[0][0], position[1][0]]))

# -------------------------------------------------------------------------
""" Look @ label of a square w/ the position as an array; Form [x, y] """
# -------------------------------------------------------------------------
def look(maze, position):
    return(maze[position[0], position[1]])

####################################################################
# Create Node Class Objects ########################################
####################################################################

class NODE:

    # Agent Actions
    move_NESW = {'N': (-1, 0), 'E': (0, 1), 'S': ( 1, 0)  , 'W': ( 0,-1)}
    move_ESWN = {'E': ( 0, 1), 'S': (1, 0), 'W': ( 0,-1)  , 'N': (-1, 0)}
    move_SENW = {'S': ( 1, 0), 'E': (0, 1), 'N': (-1, 0)  , 'W': ( 0,-1)}
    move_WSEN = {'W': ( 0,-1), 'S': (1, 0), 'E': ( 0, 1)  , 'N': (-1, 0)}
    dirs = ["nesw", "eswn", "senw", "wsen"]

    def  __init__(self, position, parent, action, cost):
        self.position = position     # The STATE; positions = (row, column) 
        self.parent   = parent       # Reference to parent node (none == root node)
        self.action   = action       # Action used in transition func (root.action = null)
        self.cost     = cost         # Depth within Uniform Cost (also g(n) for A*)

    def __str__(self):
        return f"NODE - position = {self.position}; action = {self.action}; cost = {self.cost}"

    # Returns path from Root Node to current Node
    def getPathFromRoot(self):
        node = self
        path = [node]

        # Parent/Current Node = Starting point of path
        while not node.parent is None:
            node = node.parent
            path.append(node)
        path.reverse()

        return (path)

    def INITIAL(self):
        return find_pos(self, what="S", parent=None, action=None, cost=0)

    def direction(location, speed = 1.0, rand = False):
        if rand == False:
            dx, dy = NODE.move_NESW[location]
            return (dx * speed, dy * speed)

        if rand == True:
            move_order = random.choice(NODE.dirs)

            if move_order == 'nesw': dx, dy = NODE.move_NESW[location]
            if move_order == 'eswn': dx, dy = NODE.move_ESWN[location]
            if move_order == 'wsen': dx, dy = NODE.move_WSEN[location]
            if move_order == 'senw': dx, dy = NODE.move_SENW[location]
            return (dx * speed, dy * speed)

####################################################################
# EXPAND Function ##################################################
####################################################################

# Page 167
def EXPAND(problem, node):
    # s <-- node.STATE (node.position)
    s = []
    #s = node.position

    r = random.choice(NODE.dirs)
    if r == 'nesw': dir_ = NODE.move_NESW
    if r == 'eswn': dir_ = NODE.move_ESWN
    if r == 'senw': dir_ = NODE.move_SENW
    if r == 'wsen': dir_ = NODE.move_WSEN

    # for each action in problem.ACTIONS(s) do
    for action in dir_:
        x, y = node.position
        dx, dy = dir_[action]
        dx2, dy2 = int(x + dx), int(y + dy)
        if look(problem, ([dx2],[dy2])) != 'X':
            # s2 or s' <-- problem.RESULT(s, action)
            s2 = (dx2, dy2)

            # cost <-- node.PATH-COST + problem.ACTION-COST(s, action, s')
            cost = lambda s2: 1

            # yield NODE(STATE=s', PARENT=node, ACTION=action, PATH-COST=cost)
            s.append(NODE(position=s2, parent=node, action=action, cost=cost))
            #s.append((s2, action, 1))
    return s

####################################################################
# Search Algorithms ################################################
####################################################################


# -------------------------------------------------------------------
# Best First-Search -------------------------------------------------
# -------------------------------------------------------------------
import queue
def BestFirstSearch(problem, f=1):

    # node <-- NODE(STATE=problem.INITIAL)
    node = NODE(position=find_pos(problem,"S"))

    # frontier <-- a priority queue ordered by f, with a node as an element
    frontier = queue.Queue
    frontier.put(node)

    # reached <-- a lookup table, with one entry with key problem.INITIAL & value node
    reached = {}
    reached[node.position] = [(node.position, " ")]


    # while not is-empty(frontier) do
    while not frontier.empty():

        # node <-- pop(frontier)
        node = frontier.get()

        # if problem.IS-GOAL(node.STATE) then return node
        if node.position == find_pos(problem, "G"): return node

        # for each child in EXPAND(problem, node) do
        for child in EXPAND(problem, node):

            # s <-- child.STATE
            s = child.position

            # if s is not in reached or child.PATH-COST < reached[s].PATH-COST then
            if ( (s not in reached) or (child.cost < reached[s].cost)):

                # reached[s] <-- child
                reached[(s)] = child

                # add child to frontier
                frontier.append(child)

    return -1

# -------------------------------------------------------------------
# Breadth First-Search (pg. 176) ------------------------------------
# -------------------------------------------------------------------
def BreadthFirstSearch(problem, debug=False, visualize=False):

    # node <-- NODE(problem.INITIAL) then return node
    node = NODE(find_pos(problem, what="S"), parent = None, action = None, cost = 0)

    # if problem.IS-GOAL(node.STATE) then return node
    if find_pos(problem, what='G') == node.position:
        if debug: print("Arrived @ Goal: Returning...")
        return node

    # frontier <-- a FIFO queue, w/ node as an element
    frontier = queue.Queue()
    frontier.put(node)

    # reached <-- {problem.INITIAL}
    #reached = [(node.position, " ")]
    reached = [(node.position)]

    # while not IS-EMPTY(frontier) do
    path_counter = 0
    while not frontier.empty():
        if debug: print("Frontier NOT empty")

        # node <-- POP(frontier)
        node = frontier.get()
        if find_pos(problem, what='G') == node: break
        #head = frontier.get()
        # if find_pos(problem, what='G') == head: break
        #head = NODE(position=head.position, parent=head.parent, action=head.action, cost=head.cost)
        
        path_counter += 1
        
        # root.position = node.position
        #root = root.position

        # for each child in EXPAND(problem, node) do
        for child in EXPAND(problem, node):
            
            # s <-- child.STATE
            s = child.position
            if debug: print(s,"--> :", problem[s])

            # if problem.IS-GOAL(s) then return child
            if find_pos(problem, what='G') == s:
                print("Path length:", len(child.getPathFromRoot()))
                print("Reached squares:", path_counter)
                break
                #print("HERE: ", child)
                #return child

            # if s is not in reached then
            if s not in reached:
                if debug: print("Current Position: ", s)
                
                # add s to reached
                reached.append(s)

                # add child to frontier 
                frontier.put(child)
                
                if find_pos(problem, what='G') != s:
                    if problem[s] == ' ' and problem[s] != '.': problem[s] = 'F'
                    if debug: print('Adding to frontier', s)
                    
            elif s in reached and problem[s] != 'S':
                problem[s] = '.'

    #x = child.getPathFromRoot()
    #print("Positions:", [n.position for n in x])
    #for n in x:
        #problem[n.position] = 'P'
    #x = reached[node]
    #path_length = x.parent
    #print("Path Length: ", path_length)
    #while len(reached) != 0:
    #    x = reached.pop()
    #    print("Adding: ", x)
    #    problem[x] = 'P'
    #    if debug: print()
        #frontier.put(0,x[1])
        #x = reached[x]
        
    #frontier.insert(0,[0,1])
    #print("Path length:", len(frontier))
        
        # print(x)

                
    return -1

# -------------------------------------------------------------------
# Uniform Cost-Search / Dijkstra's Algorithm ------------------------
# -------------------------------------------------------------------
def UniformCostSearch(problem):
    return BestFirstSearch(problem, PATH_COST)

# -------------------------------------------------------------------
# Depth First-Search ------------------------------------------------
# -------------------------------------------------------------------
def DepthFirstSearch():
    print("DepthFirstSearch")

# -------------------------------------------------------------------
# Greedy Best First-Search ------------------------------------------
# -------------------------------------------------------------------
def GreedyBestFirstSearch():
    print("GreedyBestFirstSearch")

# -------------------------------------------------------------------
# A* Search ---------------------------------------------------------
# -------------------------------------------------------------------
def A_Search():
    print("A_Search")