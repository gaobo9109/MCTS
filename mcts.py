
import numpy as np
import random
from math import sqrt, cos, sin, log
import pprint

class Node:
    def __init__(self, state, action, action_space, terminal, parent):
        self.parent = parent
        self.state = state
        self.action = action
        self.terminal = terminal
        self.state_visits = 0
        self.action_space = action_space
        self.untried_actions = list(action_space)
        self.children = {action: {} for action in action_space}
        self.UCB = {action: float("inf") for action in action_space}
        self.action_visits = {action: 0 for action in action_space}
        self.q_value = {action: 0 for action in action_space}
        self.exploration_constant = 1 / sqrt(2)

    def pick_best_action(self):
        if self.untried_actions:
            action = random.choice(self.untried_actions)
            self.untried_actions.remove(action)
        else:
            for action in self.action_space:
                self.UCB[action] = self.q_value[action] + self.exploration_constant \
                    * sqrt(log(self.state_visits) / self.action_visits[action])
            action = max(self.UCB, key=self.UCB.get)
        return action

    def get_next_state_node(self, action, next_state):
        return self.children[action].get(next_state, None)

    def set_next_state_node(self, action, next_state, terminal):  
        node = Node(next_state, action, self.action_space, terminal, self)
        self.children[action][next_state] = node

    def get_all_possible_next_states(self):
        children = []
        for _, next_states_dict in self.children.items():
            children.extend(next_states_dict.values())
        return children

    def update(self, action, reward):
        self.state_visits += 1
        self.action_visits[action] += 1
        self.q_value[action] = 1/(self.action_visits[action]+1) \
            * (self.action_visits[action] * self.q_value[action] + reward)
        

class MCTS:
    def __init__(self, world, action_space, start, goal):
        self.world = world
        self.world_height = len(self.world)
        self.world_width = len(self.world[0])
        self.goal = goal
        self.action_space = action_space
        self.root = Node(start, None, action_space, False, None)
        self.state_nodes = set()

    def compute_next_state(self, state, action):
        row, col, theta = state
        v, w = action
        if w != 0:
            ratio = v / w
            new_row = int(round(row + ratio * cos(theta) - ratio * cos(theta + w)))
            new_col = int(round(col - ratio * sin(theta) + ratio * sin(theta + w)))
            new_theta = (theta + w + 2 * np.pi) % (2 * np.pi)
        else:
            new_row = int(round(row + v * sin(theta)))
            new_col = int(round(col + v * cos(theta)))
            new_theta = theta
        
        return (new_row, new_col, new_theta)

    def stochastic_state_update(self, state, action):
        v, w = action
        if w == 0:
            prob = random.random()
            if prob > 0.95:
                action = (v * np.pi/2, np.pi/2)
            elif prob > 0.90:
                action = (v * np.pi/2, -np.pi/2)

        next_state = self.compute_next_state(state, action)
        reward = self.reward(next_state)
        done = reward != 0
        return next_state, reward, done

    def reward(self, state):
        row, col, _ = state
        if (row, col) == self.goal:
            reward = 1
        elif row < 0 or row >= self.world_height \
                or col < 0 or col >= self.world_width \
                or self.world[row][col] == 1:
            reward = -1
        else:
            reward = 0
        return reward

    def default_policy(self, node):
        state = node.state
        done = node.terminal
        reward = -1
        while not done:
            action = random.choice(self.action_space)
            next_state, reward, done = self.stochastic_state_update(state, action)
            # print(state, action, next_state, reward, done)
            state = next_state
        return reward

    def tree_policy(self):
        current = self.root
        while not current.terminal:
            state = current.state
            action = current.pick_best_action()
            next_state, _, done = self.stochastic_state_update(state, action)
            next_state_node = current.get_next_state_node(action, next_state)
            if next_state_node:
                current = next_state_node
            else:
                current.set_next_state_node(action, next_state, done)
                current = current.get_next_state_node(action, next_state)
                self.state_nodes.add(current)
                break
        return current


    def backpropagate(self, node, reward):
        action = node.action
        current = node.parent

        while current:
            current.update(action, reward)
            action = current.action
            current = current.parent

    def generate_action_table(self):
        action_table = {}
        print(len(self.state_nodes))
        for node in self.state_nodes:
            if not node.terminal:
                action_table[node.state] = node.pick_best_action()
        return action_table


    def run(self, sim_count):
        for _ in range(sim_count):
            node = self.tree_policy()
            # print(node.state, node.terminal)
            reward = self.default_policy(node)
            self.backpropagate(node, reward)
        # print()

        action_table = self.generate_action_table()
        return action_table


if __name__ == "__main__":
    world = [[0,0,0,0,0,0,0,0,0,0],
             [1,1,1,1,1,1,1,1,1,0],
             [0,0,0,0,0,0,0,0,1,0],
             [0,0,0,0,0,0,0,0,1,0],
             [0,0,1,1,1,1,1,0,1,0],
             [0,0,1,0,0,0,1,0,1,0],
             [0,0,1,0,0,0,1,0,1,0],
             [0,0,1,0,0,0,0,0,1,0],
             [0,0,1,1,1,1,1,1,1,0],
             [0,0,0,0,0,0,0,0,0,0]]

    world = [[0,0,0,0],
             [0,1,0,1],
             [0,0,0,1],
             [1,0,0,0]]

    action_space = [(1,0), (0, np.pi/2), (0, -np.pi/2)]
    start = (0,0,0)
    goal = (3,3)
    mcts = MCTS(world, action_space, start, goal)
    action_table = mcts.run(10000)
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(action_table)
