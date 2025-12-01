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
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --red_opts and --blue_opts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """
    return [eval(first)(first_index), eval(second)(second_index)]


##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
    """
    A base class for reflex agents that choose score-maximizing actions
    """

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None
        # Cache for this turn's computations
        self.current_turn_cache = {}

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)
    
    def choose_action(self, game_state):
        """Clear cache at start of each turn"""
        self.current_turn_cache = {}
        return self._choose_action_impl(game_state)
    
    def _choose_action_impl(self, game_state):
        """Base implementation of choose_action"""
        actions = game_state.get_legal_actions(self.index)
        values = [self.evaluate(game_state, a) for a in actions]
        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]
        
        food_left = len(self.get_food(game_state).as_list())
        if food_left <= 2:
            best_dist = 9999
            best_action = None
            for action in actions:
                successor = self.get_successor(game_state, action)
                pos2 = successor.get_agent_position(self.index)
                dist = self.get_maze_distance(self.start, pos2)
                if dist < best_dist:
                    best_action = action
                    best_dist = dist
            return best_action
        
        return random.choice(best_actions)
    
    def get_enemies_info(self, game_state):
        """Cache enemy information for this turn"""
        if 'enemies' not in self.current_turn_cache:
            enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
            ghosts = [a for a in enemies if not a.is_pacman and a.get_position() is not None]
            invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
            self.current_turn_cache['enemies'] = enemies
            self.current_turn_cache['ghosts'] = ghosts
            self.current_turn_cache['invaders'] = invaders
        return self.current_turn_cache['enemies'], self.current_turn_cache['ghosts'], self.current_turn_cache['invaders']

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

    def evaluate(self, game_state, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        return features * weights

    def get_features(self, game_state, action):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        features['successor_score'] = self.get_score(successor)
        return features

    def get_weights(self, game_state, action):
        """
        Normally, weights do not depend on the game state.  They can be either
        a counter or a dictionary.
        """
        return {'successor_score': 1.0}


class OffensiveReflexAgent(ReflexCaptureAgent):
    """
    Aggressive offensive agent that goes deep for food and escapes smartly
    """

    def _choose_action_impl(self, game_state):
        """
        Enhanced with smarter escaping and return logic
        """
        actions = game_state.get_legal_actions(self.index)
        
        # Remove STOP to keep moving
        if Directions.STOP in actions and len(actions) > 1:
            actions.remove(Directions.STOP)
        
        my_state = game_state.get_agent_state(self.index)
        my_pos = my_state.get_position()
        
        # Use cached enemy info
        _, ghosts, _ = self.get_enemies_info(game_state)
        
        # Emergency escape if ghost is close
        if ghosts and my_state.is_pacman:
            ghost_dists = [self.get_maze_distance(my_pos, g.get_position()) for g in ghosts]
            min_ghost_dist = min(ghost_dists)
            
            if min_ghost_dist <= 3:
                closest_ghost = ghosts[ghost_dists.index(min_ghost_dist)]
                # Only flee if ghost isn't scared
                if closest_ghost.scared_timer <= 0:
                    return self.escape_to_safety(game_state, actions)
        
        # Return home with food
        food_left = len(self.get_food(game_state).as_list())
        if my_state.num_carrying >= 3 or food_left <= 2:
            return self.return_home(game_state, actions)
        
        # Normal evaluation
        values = [self.evaluate(game_state, a) for a in actions]
        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]
        return random.choice(best_actions)
    
    def escape_to_safety(self, game_state, actions):
        """Run back to safe zone"""
        best_dist = 9999
        best_action = None
        
        for action in actions:
            successor = self.get_successor(game_state, action)
            pos = successor.get_agent_position(self.index)
            dist = self.get_maze_distance(self.start, pos)
            if dist < best_dist:
                best_action = action
                best_dist = dist
        
        return best_action if best_action else random.choice(actions)
    
    def return_home(self, game_state, actions):
        """Return home with collected food"""
        return self.escape_to_safety(game_state, actions)

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        food_list = self.get_food(successor).as_list()
        features['successor_score'] = -len(food_list)

        my_pos = successor.get_agent_state(self.index).get_position()
        
        # Distance to nearest food (greedy search)
        if len(food_list) > 0:
            min_distance = min([self.get_maze_distance(my_pos, food) for food in food_list])
            features['distance_to_food'] = min_distance
        
        # Capsule priority when ghosts nearby (multiagent minimax concept)
        capsules = self.get_capsules(successor)
        if capsules:
            # Reuse cached ghost info from successor state
            _, ghosts, _ = self.get_enemies_info(successor)
            if ghosts:
                min_ghost_dist = min([self.get_maze_distance(my_pos, g.get_position()) for g in ghosts])
                if min_ghost_dist <= 5:  # Ghost nearby, value capsules
                    capsule_dist = min([self.get_maze_distance(my_pos, cap) for cap in capsules])
                    features['capsule_distance'] = capsule_dist
        
        # Dead-end detection (search concept: avoid states with no good successors)
        legal_actions = successor.get_legal_actions(self.index)
        if len(legal_actions) <= 2:  # Only STOP + one direction = dead end
            features['dead_end'] = 1
        
        # Penalize reversing (avoid oscillation)
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev:
            features['reverse'] = 1
            
        return features

    def get_weights(self, game_state, action):
        return {
            'successor_score': 100,
            'distance_to_food': -3,
            'capsule_distance': -8,  # Prioritize capsules when threatened
            'dead_end': -50,  # Strongly avoid dead ends
            'reverse': -2
        }


class DefensiveReflexAgent(ReflexCaptureAgent):
    """
    Aggressive defensive agent that actively hunts invaders
    """

    def _choose_action_impl(self, game_state):
        """
        Remove STOP to stay active
        """
        actions = game_state.get_legal_actions(self.index)
        
        # Remove STOP to keep moving
        if Directions.STOP in actions and len(actions) > 1:
            actions.remove(Directions.STOP)
        
        values = [self.evaluate(game_state, a) for a in actions]
        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]
        return random.choice(best_actions)

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)

        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        # Computes whether we're on defense (1) or offense (0)
        features['on_defense'] = 1
        if my_state.is_pacman: features['on_defense'] = 0

        # Use cached invader info
        _, _, invaders = self.get_enemies_info(successor)
        features['num_invaders'] = len(invaders)
        
        if len(invaders) > 0:
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
            min_dist = min(dists)
            features['invader_distance'] = min_dist
            
            # Predictive interception (expectimax concept)
            # If invader is far, position between invader and their target food
            if min_dist > 3:
                defending_food = self.get_food_you_are_defending(successor).as_list()
                if defending_food:
                    # Closest invader (already computed)
                    invader_pos = invaders[dists.index(min_dist)].get_position()
                    # Find food closest to invader
                    target_food = min(defending_food, key=lambda f: self.get_maze_distance(invader_pos, f))
                    # Position between invader and their likely target
                    intercept_dist = self.get_maze_distance(my_pos, target_food)
                    features['intercept_position'] = intercept_dist
        else:
            # Patrol middle when no invaders
            center_x = game_state.data.layout.width // 2
            if self.red:
                center_x -= 1
            center_y = game_state.data.layout.height // 2
            patrol_pos = (center_x, center_y)
            features['patrol_distance'] = self.get_maze_distance(my_pos, patrol_pos)

        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        return features

    def get_weights(self, game_state, action):
        return {
            'num_invaders': -1000,
            'on_defense': 100,
            'invader_distance': -100,
            'intercept_position': -15,  # Cut off invader's path
            'patrol_distance': -5,
            'reverse': -2
        }
