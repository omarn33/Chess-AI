import math
import chess.lib
from chess.lib.utils import encode, decode
from chess.lib.heuristics import evaluate
from chess.lib.core import makeMove

###########################################################################################
# Utility function: Determine all the legal moves available for the side.
# This is modified from chess.lib.core.legalMoves:
#  each move has a third element specifying whether the move ends in pawn promotion
def generateMoves(side, board, flags):
    for piece in board[side]:
        fro = piece[:2]
        for to in chess.lib.availableMoves(side, board, piece, flags):
            promote = chess.lib.getPromote(None, side, board, fro, to, single=True)
            yield [fro, to, promote]
            
###########################################################################################
# Example of a move-generating function:
# Randomly choose a move.
def random(side, board, flags, chooser):
    '''
    Return a random move, resulting board, and value of the resulting board.
    Return: (value, moveList, boardList)
      value (int or float): value of the board after making the chosen move
      moveList (list): list with one element, the chosen move
      moveTree (dict: encode(*move)->dict): a tree of moves that were evaluated in the search process
    Input:
      side (boolean): True if player1 (Min) plays next, otherwise False
      board (2-tuple of lists): current board layout, used by generateMoves and makeMove
      flags (list of flags): list of flags, used by generateMoves and makeMove
      chooser: a function similar to random.choice, but during autograding, might not be random.
    '''
    moves = [ move for move in generateMoves(side, board, flags) ]
    if len(moves) > 0:
        move = chooser(moves)
        newside, newboard, newflags = makeMove(side, board, move[0], move[1], flags, move[2])
        value = evaluate(newboard)
        print((value, [ move ], { encode(*move): {} }))
        return (value, [ move ], { encode(*move): {} })
    else:
        return (evaluate(board), [], {})

###########################################################################################
# Stuff you need to write:
# Move-generating functions using minimax, alphabeta, and stochastic search.
def minimax(side, board, flags, depth):
    '''
    Return a minimax-optimal move sequence, tree of all boards evaluated, and value of best path.
    Return: (value, moveList, moveTree)
      value (float): value of the final board in the minimax-optimal move sequence
      moveList (list): the minimax-optimal move sequence, as a list of moves
      moveTree (dict: encode(*move)->dict): a tree of moves that were evaluated in the search process
    Input:
      side (boolean): True if player1 (Min) plays next, otherwise False
      board (2-tuple of lists): current board layout, used by generateMoves and makeMove
      flags (list of flags): list of flags, used by generateMoves and makeMove
      depth (int >=0): depth of the search (number of moves)
    '''

    # Store the next possible moves from the current position
    possible_moves = [move for move in generateMoves(side, board, flags)]

    # Base Case
    if (depth == 0) or (len(possible_moves) == 0):
        value = evaluate(board)
        return (value, [], {})

    # Store the best move from the current location
    best_move = None

    # Store the evaluation score of the board after the best move is played
    best_move_score = None

    # Initialize a dictionary to store all the possible moves to a given depth
    move_tree = {}

    # Initialize a list to store the moves containing the optimal set of moves
    move_list = []

    # White's turn (maximizing player)
    if not side:
        # For each possible move, determine max score for white
        for move in possible_moves:

            # Move to next possible position and update board
            newside, newboard, newflags = makeMove(side, board, move[0], move[1], flags, move[2])

            # Evaluate the score at the current position
            current_pos_evaluation, prev_move_list, prev_move_tree = minimax(True, newboard, newflags, depth - 1)

            # Store the newly explored move in move_tree
            move_tree[encode(*move)] = prev_move_tree

            # Determine if the current position score is > maximum possible score at the parent position
            if (best_move_score == None) or (current_pos_evaluation > best_move_score):
                # Update best move
                best_move = move

                # Update best move score
                best_move_score = current_pos_evaluation

                # Store the previous moves list
                move_list = prev_move_list

    # Black's turn (minimizing player)
    else:
        # For each possible move, determine min score for black to concede (min white score)
        for move in possible_moves:

            # Move to next possible position and update board
            newside, newboard, newflags = makeMove(side, board, move[0], move[1], flags, move[2])

            # Evaluate the score at the current position
            current_pos_evaluation, prev_move_list, prev_move_tree = minimax(False, newboard, newflags, depth - 1)

            # Store the moves explored in move_tree
            move_tree[encode(*move)] = prev_move_tree

            # Determine if the current position score is < minimum possible score at the parent position
            if (best_move_score == None) or (current_pos_evaluation < best_move_score):
                # Update best move
                best_move = move

                # Update best move score
                best_move_score = current_pos_evaluation

                # Store the previous moves list
                move_list = prev_move_list

    # Add best_move to the front of the previous move list
    move_list.insert(0, best_move)

    return (best_move_score, move_list, move_tree)


def alphabeta(side, board, flags, depth, alpha=-math.inf, beta=math.inf):
    '''
    Return minimax-optimal move sequence, and a tree that exhibits alphabeta pruning.
    Return: (value, moveList, moveTree)
      value (float): value of the final board in the minimax-optimal move sequence
      moveList (list): the minimax-optimal move sequence, as a list of moves
      moveTree (dict: encode(*move)->dict): a tree of moves that were evaluated in the search process
    Input:
      side (boolean): True if player1 (Min) plays next, otherwise False
      board (2-tuple of lists): current board layout, used by generateMoves and makeMove
      flags (list of flags): list of flags, used by generateMoves and makeMove
      depth (int >=0): depth of the search (number of moves)
    '''
    # Store the next possible moves from the current position
    possible_moves = [move for move in generateMoves(side, board, flags)]

    # Base Case
    if (depth == 0) or (len(possible_moves) == 0):
        value = evaluate(board)
        return (value, [], {})

    # Store the best move from the current location
    best_move = None

    # Store the evaluation score of the board after the best move is played
    best_move_score = None

    # Initialize a dictionary to store all the possible moves to a given depth
    move_tree = {}

    # Initialize a list to store the moves containing the optimal set of moves
    move_list = []

    # White's turn (maximizing player)
    if not side:
        # For each possible move, determine max score for white
        for move in possible_moves:

            # Move to next possible position and update board
            newside, newboard, newflags = makeMove(side, board, move[0], move[1], flags, move[2])

            # Evaluate the score at the current position
            current_pos_evaluation, prev_move_list, prev_move_tree = alphabeta(True, newboard, newflags, depth - 1, alpha, beta)

            # Store the newly explored move in move_tree
            move_tree[encode(*move)] = prev_move_tree

            # Determine if the current position score is > maximum possible score at the parent position
            if (best_move_score is None) or (current_pos_evaluation > best_move_score):
                # Update best move
                best_move = move

                # Update best move score
                best_move_score = current_pos_evaluation

                # Store the previous moves list
                move_list = prev_move_list

            # Update alpha
            if best_move_score > alpha:
                alpha = best_move_score

            # Terminating condition
            if alpha >= beta:
                break

    # Black's turn (minimizing player)
    else:
        # For each possible move, determine min score for black to concede (min white score)
        for move in possible_moves:

            # Move to next possible position and update board
            newside, newboard, newflags = makeMove(side, board, move[0], move[1], flags, move[2])

            # Evaluate the score at the current position
            current_pos_evaluation, prev_move_list, prev_move_tree = alphabeta(False, newboard, newflags, depth - 1, alpha, beta)

            # Store the moves explored in move_tree
            move_tree[encode(*move)] = prev_move_tree

            # Determine if the current position score is < minimum possible score at the parent position
            if (best_move_score is None) or (current_pos_evaluation < best_move_score):
                # Update best move
                best_move = move

                # Update best move score
                best_move_score = current_pos_evaluation

                # Store the previous moves list
                move_list = prev_move_list

            # Update beta
            if best_move_score < beta:
                beta = best_move_score

            # Terminating condition
            if alpha >= beta:
                break

    # Add best_move to the front of the previous move list
    move_list.insert(0, best_move)

    return (best_move_score, move_list, move_tree)


def stochastic(side, board, flags, depth, breadth, chooser):
    '''
    Choose the best move based on breadth randomly chosen paths per move, of length depth-1.
    Return: (value, moveList, moveTree)
      value (float): average board value of the paths for the best-scoring move
      moveLists (list): any sequence of moves, of length depth, starting with the best move
      moveTree (dict: encode(*move)->dict): a tree of moves that were evaluated in the search process
    Input:
      side (boolean): True if player1 (Min) plays next, otherwise False
      board (2-tuple of lists): current board layout, used by generateMoves and makeMove
      flags (list of flags): list of flags, used by generateMoves and makeMove
      depth (int >=0): depth of the search (number of moves)
      breadth: number of different paths
      chooser: a function similar to random.choice, but during autograding, might not be random.
    '''
    # Stores all possible moves
    possible_moves = [move for move in generateMoves(side, board, flags)]

    # Stores the best move score
    best_move_score = None

    # Stores the move list containing optimal path
    move_list = None

    # Stores moves explored
    move_tree = {}

    # Traverse all possible moves
    for move in possible_moves:
        # Determine score of new board
        newside, newboard, newflags = makeMove(side, board, move[0], move[1], flags, move[2])

        # Set current score to 0
        current_move_score = 0

        # Initialize the path list
        depth_move_list = None

        # Initialize move tree
        move_tree[encode(*move)] = {}

        # Traverse each breadth
        for path in range(breadth):
            # Set new side
            breadth_side = newside

            # Set new board
            breadth_board = newboard

            # Set new breadth flags
            breadth_flags = newflags

            # Store first move
            depth_move_list = [move]

            # Traverse through the number of depths given
            for j in range(depth - 1):

                # Calculate possible moves
                depth_possible_moves = [temp_move for temp_move in generateMoves(breadth_side, breadth_board, breadth_flags)]

                # Break loop if the number of possible moves is 0
                if len(depth_possible_moves) == 0:
                    break

                # Choose a random path
                breadth_move = chooser(depth_possible_moves)

                # Append move to path list
                depth_move_list.append(breadth_move)

                # Generate new board based on breadth
                breadth_side, breadth_board, breadth_flags = makeMove(breadth_side, breadth_board, breadth_move[0], breadth_move[1],
                                                                  breadth_flags, breadth_move[2])

            # Update current move score
            current_move_score = current_move_score + evaluate(breadth_board)

            # Initialize a dictionary to store all moves explored
            depth_move_dict = {}

            # Traverse the path list in reverse
            for k in range(len(depth_move_list) - 1, 1, -1):
                # Store depth move dict as temp
                temp = depth_move_dict

                # Initialize depth move dict
                depth_move_dict = {}

                # Update depth move dict with temp value
                depth_move_dict[encode(*(depth_move_list[k]))] = temp

            # Update the move_tree with the explored dict from each depth
            move_tree[encode(*move)][encode(*depth_move_list[1])] = depth_move_dict

        # Calculate the average move score
        current_move_score /= breadth

        # Determine scores for MAX or MIN
        if not side and ((best_move_score == None) or (current_move_score > best_move_score)):
            # Update best score
            best_move_score = current_move_score

            # Update move list
            move_list = depth_move_list

        if side and ((best_move_score == None) or (current_move_score < best_move_score)):
            # Update best score
            best_move_score = current_move_score

            # Update move list
            move_list = depth_move_list

    return (best_move_score, move_list, move_tree)
