# my_team.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import random
import contest.util as util
from contest.capture_agents import CaptureAgent
from contest.game import Directions
from contest.util import nearest_point

#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='OffensiveReflexAgent', second='DefensiveReflexAgent', num_training=0):
    return [eval(first)(first_index), eval(second)(second_index)]

##########
# Agents #
##########

class BaseAgent(CaptureAgent):
    """
    A base agent with shared utilities for pathfinding and stuck detection.
    """
    def register_initial_state(self, game_state):
        CaptureAgent.register_initial_state(self, game_state)
        self.start = game_state.get_agent_position(self.index)
        self.history = []  # For stuck detection
        self.distancer.get_maze_distances() 

    def get_successor(self, game_state, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearest_point(pos):
            # Only half a grid position was covered
            return successor.generate_successor(self.index, action)
        else:
            return successor

    def is_stuck(self, pos):
        """
        Checks if the agent is oscillating or stuck in a loop.
        """
        self.history.append(pos)
        if len(self.history) > 10:
            self.history.pop(0)
        # If we visited the same spot 4 times in the last 10 moves, we are likely stuck
        return self.history.count(pos) > 3

    def get_random_legal_action(self, game_state):
        """
        Returns a random action that is not STOP (unless necessary).
        """
        actions = game_state.get_legal_actions(self.index)
        if Directions.STOP in actions:
            actions.remove(Directions.STOP)
        if not actions:
            return Directions.STOP
        return random.choice(actions)

    def a_star_search(self, game_state, targets, avoid_positions=None, avoid_radius=0):
        """
        Finds the shortest path to ANY of the target positions using A* search.
        Avoids specific positions within a certain radius.
        """
        if not targets: return None
        if avoid_positions is None: avoid_positions = []

        my_pos = game_state.get_agent_position(self.index)
        
        # Priority Queue: (priority, (position, path))
        # Priority = Cost so far + Heuristic (Manhattan distance to nearest target)
        pq = util.PriorityQueue()
        pq.push((my_pos, []), 0)
        
        visited = set()
        visited.add(my_pos)

        # Optimization: Pre-compute danger zones for O(1) lookup
        danger_zone = set()
        for avoid in avoid_positions:
            x, y = int(avoid[0]), int(avoid[1])
            danger_zone.add((x, y))
            if avoid_radius > 0:
                for i in range(1, avoid_radius + 1):
                    # Add diamond shape or box around ghost
                    for dx in range(-i, i + 1):
                        for dy in range(-i, i + 1):
                            if abs(dx) + abs(dy) <= i: # Manhattan radius
                                danger_zone.add((x + dx, y + dy))

        while not pq.is_empty():
            curr_pos, path = pq.pop()

            if curr_pos in targets:
                if len(path) > 0:
                    return path[0] # Return the first step
                return Directions.STOP

            # Expand neighbors
            x, y = int(curr_pos[0]), int(curr_pos[1])
            for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
                dx, dy = 0, 0
                if action == Directions.NORTH: dy = 1
                elif action == Directions.SOUTH: dy = -1
                elif action == Directions.EAST: dx = 1
                elif action == Directions.WEST: dx = -1
                
                next_pos = (x + dx, y + dy)
                
                if not game_state.has_wall(next_pos[0], next_pos[1]):
                    if next_pos not in visited and next_pos not in danger_zone:
                        visited.add(next_pos)
                        
                        # Heuristic: Min dist to any target
                        h_cost = min([self.get_maze_distance(next_pos, t) for t in targets])
                        g_cost = len(path) + 1
                        
                        pq.push((next_pos, path + [action]), g_cost + h_cost)
        
        return None # No path found

class OffensiveReflexAgent(BaseAgent):
    """
    Target-Based Offensive Agent.
    Strategies:
    1. If 0 Food: Pathfinds to nearest food, ignoring ghosts unless they are colliding.
    2. If Food > 0: Pathfinds to food with safety buffer.
    3. If Food > Threshold: Pathfinds to Home with safety buffer.
    """

    def choose_action(self, game_state):
        my_pos = game_state.get_agent_position(self.index)
        
        # 1. LOOP BREAKER
        if self.is_stuck(my_pos):
            self.history = [] # Reset
            return self.get_random_legal_action(game_state)

        my_state = game_state.get_agent_state(self.index)
        carrying = my_state.num_carrying
        food_list = self.get_food(game_state).as_list()
        capsules = self.get_capsules(game_state)
        
        # Get Defenders (Ghosts)
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        ghosts = [a for a in enemies if not a.is_pacman and a.get_position() is not None]
        ghost_positions = [g.get_position() for g in ghosts]
        
        # Check if ghosts are scared
        scared_ghosts = [g for g in ghosts if g.scared_timer > 5]
        active_ghosts = [g.get_position() for g in ghosts if g.scared_timer <= 5]

        # --- STRATEGY LOGIC ---

        # DECISION 1: Should we go home?
        target_list = []
        go_home = False
        
        # Dynamic threshold based on remaining food
        threshold = 6
        if len(food_list) <= 2: threshold = 1
        
        if carrying >= threshold or len(food_list) <= 2:
            go_home = True
            
        # If we have any food and are near the border, just deposit it
        dist_to_home = self.get_maze_distance(my_pos, self.start)
        if carrying > 0 and dist_to_home < 5:
            go_home = True

        # DECISION 2: Determine Safety Buffer
        # 0 Food = 0 Buffer (Brave)
        # Carrying Food = 2 Buffer (Cautious)
        avoid_radius = 1 # Default
        if carrying == 0:
            avoid_radius = 0 # No fear, just avoid direct collision
        elif go_home:
            avoid_radius = 2 # Very careful when bringing points home
        
        # If ghosts are scared, ignore them (radius 0)
        if len(scared_ghosts) == len(ghosts) and len(ghosts) > 0:
            avoid_radius = 0
            active_ghosts = [] # Treat them as empty space

        # EXECUTION
        action = None
        
        # PRIORITY A: SURVIVAL
        # If a ghost is literally next to us (dist=1) and not scared, RUN to any safe spot
        if active_ghosts:
            dists = [self.get_maze_distance(my_pos, g) for g in active_ghosts]
            if min(dists) <= 1:
                # Emergency Logic: Find ANY valid neighbor that increases distance to ghost
                best_move = Directions.STOP
                max_dist = -1
                for act in game_state.get_legal_actions(self.index):
                    succ = self.get_successor(game_state, act)
                    pos = succ.get_agent_position(self.index)
                    d = min([self.get_maze_distance(pos, g) for g in active_ghosts])
                    if d > max_dist:
                        max_dist = d
                        best_move = act
                return best_move

        # PRIORITY B: CAPSULES (Power Pellets)
        if capsules and not go_home:
            closest_cap_dist = min([self.get_maze_distance(my_pos, c) for c in capsules])
            if closest_cap_dist < 10: # Only divert if reasonably close
                action = self.a_star_search(game_state, capsules, active_ghosts, avoid_radius)
                if action: return action

        # PRIORITY C: PRIMARY OBJECTIVE (Home or Food)
        if go_home:
            # Target: Nearest border crossing (using start as proxy)
            action = self.a_star_search(game_state, [self.start], active_ghosts, avoid_radius)
        else:
            # Target: Nearest Food (top 3)
            if food_list:
                sorted_food = sorted(food_list, key=lambda x: self.get_maze_distance(my_pos, x))
                action = self.a_star_search(game_state, sorted_food[:3], active_ghosts, avoid_radius)

        # FAILSAFE: FLANKING
        # If path is blocked, ignore buffer and try safest food
        if action is None and len(food_list) > 0:
            safest_food = None
            max_ghost_dist = -1
            for f in food_list:
                d = min([self.get_maze_distance(f, g) for g in active_ghosts]) if active_ghosts else 0
                if d > max_ghost_dist:
                    max_ghost_dist = d
                    safest_food = f
            
            if safest_food:
                action = self.a_star_search(game_state, [safest_food], active_ghosts, 0)

        if action is None:
            return self.get_random_legal_action(game_state)
        
        return action

class DefensiveReflexAgent(BaseAgent):
    """
    Patrol & Intercept Defender.
    """
    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.patrol_points = []
        self.patrol_index = 0
    
    def register_initial_state(self, game_state):
        super().register_initial_state(game_state)
        
        # Calculate Patrol Points (Border Line)
        layout = game_state.data.layout
        mid_x = layout.width // 2
        if self.red: mid_x -= 1
        else: mid_x += 1
        
        valid_y = [y for y in range(1, layout.height - 1) if not game_state.has_wall(mid_x, y)]
        if valid_y:
            self.patrol_points = [
                (mid_x, max(valid_y)),            # Top
                (mid_x, valid_y[len(valid_y)//2]), # Mid
                (mid_x, min(valid_y))             # Bot
            ]

    def choose_action(self, game_state):
        my_pos = game_state.get_agent_position(self.index)
        
        # 1. IDENTIFY INVADERS
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        
        # 2. CHASE MODE
        if invaders:
            invader_pos = [i.get_position() for i in invaders]
            action = self.get_defensive_action(game_state, invader_pos)
            if action: return action
            
        # 3. PATROL MODE
        if self.patrol_points:
            target = self.patrol_points[self.patrol_index]
            if self.get_maze_distance(my_pos, target) <= 2:
                self.patrol_index = (self.patrol_index + 1) % len(self.patrol_points)
                target = self.patrol_points[self.patrol_index]
            
            action = self.get_defensive_action(game_state, [target])
            if action: return action
            
        return self.get_random_legal_action(game_state)

    def get_defensive_action(self, game_state, targets):
        """
        BFS that strictly avoids entering enemy territory.
        """
        my_pos = game_state.get_agent_position(self.index)
        layout = game_state.data.layout
        mid_x = layout.width // 2
        
        queue = [(my_pos, [])]
        visited = set([my_pos])
        
        while queue:
            curr, path = queue.pop(0)
            if curr in targets:
                return path[0] if path else Directions.STOP
            
            x, y = int(curr[0]), int(curr[1])
            for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
                dx, dy = 0, 0
                if action == Directions.NORTH: dy = 1
                elif action == Directions.SOUTH: dy = -1
                elif action == Directions.EAST: dx = 1
                elif action == Directions.WEST: dx = -1
                
                next_pos = (x + dx, y + dy)
                
                if not game_state.has_wall(next_pos[0], next_pos[1]):
                    # Check Boundary
                    is_valid_side = (next_pos[0] < mid_x) if self.red else (next_pos[0] >= mid_x)
                    if is_valid_side and next_pos not in visited:
                        visited.add(next_pos)
                        queue.append((next_pos, path + [action]))
        return None
