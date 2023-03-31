# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        score = 0

        closestGhostPosition = newGhostStates[0].configuration.pos
        closestGhost = manhattanDistance(newPos, closestGhostPosition)

        # Minimize distance from pacman to food
        newFoodPositions = newFood.asList()
        foodDistances = [manhattanDistance(newPos, foodPosition) for foodPosition in newFoodPositions]

        if successorGameState.isWin():
            return float("+inf")

        if len(foodDistances) == 0:
            return 0

        closestFood = min(foodDistances)

        # Stop action would reduce score because of the pacman's timer constraint
        if action == 'Stop':
            score -= 50

        return successorGameState.getScore() + closestGhost / (closestFood * 10) + score

    # cu cat ne aflam mai departe de fantom cu atat scorul este mai bun
    # cu cat suntem mai aproape de mancare cu atat e mai bine si fiind o functie exponentiala e mai bine


def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"

        def max_value(gameState, depth):
            availableActions = gameState.getLegalActions(0)
            #print(availableActions)
            if len(availableActions) == 0 or gameState.isWin() or gameState.isLose() or depth == self.depth:
                return (self.evaluationFunction(gameState), None)
            v = (float("-inf"))
            action = ""
            for nextAction in availableActions:
                currValAct = (v, action)
                nextValAct = (
                    min_value(gameState.generateSuccessor(0, nextAction), 1,
                              depth)[0], nextAction)
                (v, action) = max(currValAct, nextValAct, key=lambda x: x[
                    0])
            return (v, action)

        def min_value(gameState, ghostID, depth):
            availableActions = gameState.getLegalActions(ghostID)
            #print(f"{ghostID} -> {availableActions}")
            if len(availableActions) == 0:
                return (self.evaluationFunction(gameState), None)
            v = float("inf")
            action = ""
            for nextAction in availableActions:
                currValAct = (v, action)
                nextValAct = (
                    max_value(gameState.generateSuccessor(ghostID, nextAction), depth + 1)[0], nextAction) if (
                        ghostID == gameState.getNumAgents() - 1) else \
                    (min_value(gameState.generateSuccessor(ghostID, nextAction), ghostID + 1, depth)[0], nextAction)
                (v, action) = min(currValAct, nextValAct, key=lambda x: x[0])

            return (v, action)

        pacman_start = max_value(gameState, 0)[1]
        print(f"final: {pacman_start} ")
        return pacman_start


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        def max_value(gameState, depth, a, b):
            availableActions = gameState.getLegalActions(0)
            if len(availableActions) == 0 or gameState.isWin() or gameState.isLose() or depth == self.depth:  ###The trvial situations(state)
                return (self.evaluationFunction(gameState), None)
            v = (float("-inf"))  ## maximul  dintre valorile copiilor nodului
            action = ""

            for nextAction in availableActions:  ###In that way that the 2 functions are calling each other is like building the tree(diagrams from tha class)
                currValAct = (v, action)
                nextValAct = (
                    min_value(gameState.generateSuccessor(0, nextAction), 1,  ### prima fantoma urmeaza sa faca miscarea
                              depth, a, b)[0], nextAction)
                (v, action) = max(currValAct, nextValAct, key=lambda x: x[
                    0])  # We have the available moves and we are seeking for the "best" one
                if v > b:
                    return (v, action)
                a = max(a, v)
            return (v, action)

        def min_value(gameState, ghostID, depth, a, b):
            availableActions = gameState.getLegalActions(ghostID)  ##ID ul fantomei
            if len(availableActions) == 0:
                return (self.evaluationFunction(gameState), None)
            v = float("inf")  ###As we see in contrast with max we begin from +infinte
            action = ""
            for nextAction in availableActions:
                currValAct = (v, action)
                nextValAct = (
                    max_value(gameState.generateSuccessor(ghostID, nextAction), depth + 1, a, b)[0], nextAction) if (
                        ghostID == gameState.getNumAgents() - 1) else \
                    (min_value(gameState.generateSuccessor(ghostID, nextAction), ghostID + 1, depth, a, b)[0],
                     nextAction)
                (v, action) = min(currValAct, nextValAct, key=lambda x: x[0])

                if (v < a):
                    return (v, action)
                b = min(b, v)

            return (v, action)

        a = -(float("inf"))
        b = float("inf")
        pacman_start = max_value(gameState, 0, a, b)[1]
        return pacman_start


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"

        def max_value(gameState, depth):
            availableActions = gameState.getLegalActions(0)
            if len(availableActions) == 0 or gameState.isWin() or gameState.isLose() or depth == self.depth:  ###The trvial situations(state)
                return (self.evaluationFunction(gameState), None)
            v = (float("-inf"))  ## maximul  dintre valorile copiilor nodului
            action = ""
            for nextAction in availableActions:  ###In that way that the 2 functions are calling each other is like building the tree(diagrams from tha class)
                currValAct = (v, action)
                nextValAct = (
                    exp_value(gameState.generateSuccessor(0, nextAction), 1,  ### prima fantoma urmeaza sa faca miscarea
                              depth)[0], nextAction)
                (v, action) = max(currValAct, nextValAct, key=lambda x: x[
                    0])  # We have the available moves and we are seeking for the "best" one
                # It is working exactly as the theory of minimax algorithm commands
                # if (nextValue > v):  # Here we have as start -infinite ## se retine maximul
                #     v, action = nextValue, nextAction
            return (v, action)

        def exp_value(gameState, ghostID, depth):
            availableActions = gameState.getLegalActions(ghostID)  ##ID ul fantomei
            if len(availableActions) == 0:
                return (self.evaluationFunction(gameState), None)
            v = 0  ###As we see in contrast with max we begin from +infinte
            action = ""
            for nextAction in availableActions:
                nextValAct = (
                    max_value(gameState.generateSuccessor(ghostID, nextAction), depth + 1)) if (
                        ghostID == gameState.getNumAgents() - 1) else \
                    (exp_value(gameState.generateSuccessor(ghostID, nextAction), ghostID + 1, depth))

                probability = nextValAct[0] / len(availableActions)
                v += probability
            return (v, action)

        pacman_start = max_value(gameState, 0)[1]
        return pacman_start

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"

    posCapsules = currentGameState.getCapsules()
    posPacman = currentGameState.getPacmanPosition()
    posFood = currentGameState.getFood().asList()
    posGhosts = currentGameState.getGhostStates()
    posJump = currentGameState.getJump()

    legalMoves = currentGameState.getLegalActions()

    for move in legalMoves:
        if currentGameState.generatePacmanSuccessor(move).isWin():
            return float("inf")

    if currentGameState.isLose():
        return float("-inf")

    if currentGameState.isWin():
        return float("inf")

    distJump = 0
    if not posJump == None:
        distJump = util.manhattanDistance(posPacman, posJump)

    minFoodDist =min( [util.manhattanDistance(food, posPacman) for food in posFood]) ## bucata cea mai apropiata de mancare

    GhDistList = [util.manhattanDistance(posPacman, ghost.getPosition()) for ghost in posGhosts if ghost.scaredTimer == 0]
    ScGhDistList = [util.manhattanDistance(posPacman, ghost.getPosition()) for ghost in posGhosts if ghost.scaredTimer > 0]

    minGhDist = min(GhDistList) if len(GhDistList) > 0 else -1
    minScGhDist = min(ScGhDistList) if len(ScGhDistList) > 0 else -1
    # As we see they do not hve all the same role-importance in the estimation -evaluation of a state

    score = scoreEvaluationFunction(currentGameState)
    score += distJump + 1.5 * minFoodDist + 2 * (1.0 / minGhDist) + 2 * minScGhDist + 25 * len(posCapsules) + 4 * len(posFood)
    return score






# Abbreviation
better = betterEvaluationFunction
