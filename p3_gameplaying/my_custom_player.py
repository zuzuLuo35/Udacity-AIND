
from sample_players import DataPlayer
import time
import math
from isolation import DebugState

class CustomPlayer(DataPlayer):
    """ Implement your own agent to play knight's Isolation

    The get_action() method is the only required method for this project.
    You can modify the interface for get_action by adding named parameters
    with default values, but the function MUST remain compatible with the
    default interface.

    **********************************************************************
    NOTES:
    - The test cases will NOT be run on a machine with GPU access, nor be
      suitable for using any other machine learning techniques.

    - You can pass state forward to your agent on the next turn by assigning
      any pickleable object to the self.context attribute.
    **********************************************************************
    """
    def get_action(self, state):
        """ Employ an adversarial search technique to choose an action
        available in the current state calls self.queue.put(ACTION) at least

        This method must call self.queue.put(ACTION) at least once, and may
        call it as many times as you want; the caller will be responsible
        for cutting off the function after the search time limit has expired.

        See RandomPlayer and GreedyPlayer in sample_players for more examples.

        **********************************************************************
        NOTE: 
        - The caller is responsible for cutting off search, so calling
          get_action() from your own code will create an infinite loop!
          Refer to (and use!) the Isolation.play() function to run games.
        **********************************************************************
        """
        # TODO: Replace the example implementation below with your own search
        #       method by combining techniques from lecture
        #
        # EXAMPLE: choose a random move without any search--this function MUST
        #          call self.queue.put(ACTION) at least once before time expires
        #          (the timer is automatically managed for you)

        import random
        if state.ply_count < 2:
            self.queue.put(random.choice(state.actions()))
        else:
            #my_timer = time.time() + float(0.1499)
            best_move = None
            depth_limit = 3
            for depth in range(1, depth_limit + 1):
                best_move = self.alpha_beta_search(state, depth)
            if best_move is None:
                best_move = random.choice(state.actions())
            self.queue.put(best_move)
    
        # Alpha beta pruning
        # Iterative deepening to set bounds
        # Evaluation function: other than (#my_moves - #opponent_moves), partition, symmetry   
###
#    def iterative_deepening(self, state, depth_limit):
#        best_score = float("-inf")
#        best_move = None
#        for depth in range(1, depth_limit + 1):
            #if time.time() >= my_timer:
            #    return best_move
#            best_move = self.alpha_beta_search(state, depth)
        # Check depth achieved 
        #print(" %d", depth)
#        return best_move 
###        
    def alpha_beta_search(self, state, depth):
        alpha = float("-inf")
        beta = float("inf")
        best_score = float("-inf")
        best_move = None
        #score = self.score_NumOfMoves
        score = self.score_Custom

        def min_value(state, alpha, beta, depth):
            if state.terminal_test():
                return state.utility(self.player_id)
            if depth <= 0:
                return score(state)
            v = float("inf")
            for a in state.actions():
                v = min(v, max_value(state.result(a), alpha, beta, depth - 1))
                if v <= alpha:
                    return v
                beta = min(beta, v)
            return v

        def max_value(state, alpha, beta, depth):
            if state.terminal_test():
                return state.utility(self.player_id)
            if depth <= 0:
                return score(state)
            v = float("-inf")
            for a in state.actions():
                v = max(v, min_value(state.result(a), alpha, beta, depth - 1))
                if v >= beta:
                    return v
                alpha = max(alpha, v)
            return v
        
        for a in state.actions():
            #if time.time() >= my_timer:
            #    return best_move
            v = min_value(state.result(a), alpha, beta, depth - 1)
            alpha = max(alpha, v)
            if v > best_score:
                best_score = v
                best_move = a
        return best_move
        
    def score_NumOfMoves(self, state):
        loc_me = state.locs[self.player_id]
        loc_opp = state.locs[1 - self.player_id]
        return (len(state.liberties(loc_me)) - len(state.liberties(loc_opp)))
    
    def score_Custom(self, state):
        # additional heuristic: maximize current player's distance to wall and minimize for the the opponent
        loc_me = state.locs[self.player_id]
        loc_opp = state.locs[1 - self.player_id]
        num_moves_me = len(state.liberties(loc_me))
        num_moves_opp = len(state.liberties(loc_opp))
        center_distance_me = math.sqrt((loc_me%13 - 57%13)**2 + (loc_me//13 - 57//13)**2)
        center_distance_opp = math.sqrt((loc_opp%13 - 57%13)**2 + (loc_opp//13 - 57//13)**2)
        
        # create heuristic with normalized #ofMoves item and distance item, assign weight accordingly
        norm_moves = (num_moves_me - num_moves_opp)#/(num_moves_me + num_moves_opp)
        norm_distance = (center_distance_opp - center_distance_me)#/(center_distance_me + center_distance_opp)
        
        # assign more weight to custom heuristic at beginning of game and more to #ofmoves near the end
        if state.ply_count < 20:
            return (norm_moves + 2 * norm_distance)
        else:
            return (2 * norm_moves + norm_distance)