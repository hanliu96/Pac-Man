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

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
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
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
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

        "*** YOUR CODE HERE ***"
        closestGhostDis = 1e8
        for GhostState in newGhostStates:
            ghost_x, ghost_y = GhostState.getPosition()

            if newScaredTimes == 0:
                closestGhostDis = min(closestGhostDis, manhattanDistance((ghost_x, ghost_y), newPos))

        closestFoodDis = 1e8
        foods = newFood.asList()

        if len(foods) == 0:
            closestFoodDis = 0
        for food in foods:
            closestFoodDis = min(closestFoodDis, manhattanDistance(food, newPos))
        return successorGameState.getScore() - 5 / (closestGhostDis + 1) - closestFoodDis / 5

def scoreEvaluationFunction(currentGameState):
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

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
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
        return self.miniMaxSearch(gameState, agentIdx=0, depth=self.depth)[1]

    def miniMaxSearch(self, gameState, agentIdx, depth):
        if gameState.isLose() or gameState.isWin() or depth == 0:
            action = self.evaluationFunction(gameState), Directions.STOP
        elif agentIdx == 0:
            action = self.Maximizer(gameState, agentIdx, depth)
        else:
            action = self.Minimizer(gameState, agentIdx, depth)

        return action

    def Maximizer(self, gameState, agentIdx, depth):
        actions = gameState.getLegalActions(agentIdx)
        if agentIdx == gameState.getNumAgents() - 1:
            nextAgent, nextDepth = 0, depth - 1
        else:
            nextAgent, nextDepth = agentIdx + 1, depth

        maximizerScore = -1e8
        maximizerAction = Directions.STOP
        for action in actions:
            succGameState = gameState.generateSuccessor(agentIdx, action)
            nextScore = self.miniMaxSearch(succGameState, nextAgent, nextDepth)[0]
            if nextScore > maximizerScore:
                maximizerScore, maximizerAction = nextScore, action

        return maximizerScore, maximizerAction

    def Minimizer(self, gameState, agentIdx, depth):
        actions = gameState.getLegalActions(agentIdx)
        if agentIdx == gameState.getNumAgents() - 1:
            nextAgent, nextDepth = 0, depth - 1
        else:
            nextAgent, nextDepth = agentIdx + 1, depth

        minimizerScore = 1e8
        minimizerAction = Directions.STOP
        for action in actions:
            succGameState = gameState.generateSuccessor(agentIdx, action)
            nextScore = self.miniMaxSearch(succGameState, nextAgent, nextDepth)[0]
            if nextScore < minimizerScore:
                minimizerScore, minimizerAction = nextScore, action

        return minimizerScore, minimizerAction


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        return self.alphaBetaSearch(gameState, agentIdx=0, depth=self.depth,
                                    alpha=-1e8, beta=1e8)[1]

    def alphaBetaSearch(self, gameState, agentIdx, depth, alpha, beta):
        if gameState.isLose() or gameState.isWin() or depth == 0:
            action = self.evaluationFunction(gameState), Directions.STOP
        elif agentIdx == 0:
            action = self.MaxValue(gameState, agentIdx, depth, alpha, beta)
        else:
            action = self.MinValue(gameState, agentIdx, depth, alpha, beta)

        return action

    def MaxValue(self, gameState, agentIdx, depth, alpha, beta):
        actions = gameState.getLegalActions(agentIdx)
        if agentIdx == gameState.getNumAgents() - 1:
            nextAgent, nextDepth = 0, depth - 1
        else:
            nextAgent, nextDepth = agentIdx + 1, depth

        maximizerScore = -1e8
        maximizerAction = Directions.STOP
        for action in actions:
            succGameState = gameState.generateSuccessor(agentIdx, action)
            nextScore = self.alphaBetaSearch(succGameState, nextAgent, nextDepth, alpha, beta)[0]
            if nextScore > maximizerScore:
                maximizerScore, maximizerAction = nextScore, action
            if nextScore > beta:
                return nextScore, action
            alpha = max(alpha, maximizerScore)

        return maximizerScore, maximizerAction

    def MinValue(self, gameState, agentIdx, depth, alpha, beta):
        actions = gameState.getLegalActions(agentIdx)
        if agentIdx == gameState.getNumAgents() - 1:
            nextAgent, nextDepth = 0, depth - 1
        else:
            nextAgent, nextDepth = agentIdx + 1, depth

        minimizerScore = 1e8
        minimizerAction = Directions.STOP
        for action in actions:
            succGameState = gameState.generateSuccessor(agentIdx, action)
            nextScore = self.alphaBetaSearch(succGameState, nextAgent, nextDepth, alpha, beta)[0]
            if nextScore < minimizerScore:
                minimizerScore, minimizerAction = nextScore, action
            if nextScore < alpha:
                return nextScore, action
            beta = min(beta, minimizerScore)

        return minimizerScore, minimizerAction


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        return self.expectMaxSearch(gameState, agentIdx=0, depth=self.depth)[1]

    def expectMaxSearch(self, gameState, agentIdx, depth):
        if gameState.isLose() or gameState.isWin() or depth == 0:
            action = self.evaluationFunction(gameState), Directions.STOP
        elif agentIdx == 0:
            action = self.Maximizer(gameState, agentIdx, depth)
        else:
            action = self.expector(gameState, agentIdx, depth)

        return action

    def Maximizer(self, gameState, agentIdx, depth):
        actions = gameState.getLegalActions(agentIdx)
        if agentIdx == gameState.getNumAgents() - 1:
            nextAgent, nextDepth = 0, depth - 1
        else:
            nextAgent, nextDepth = agentIdx + 1, depth

        maximizerScore = -1e8
        maximizerAction = Directions.STOP
        for action in actions:
            succGameState = gameState.generateSuccessor(agentIdx, action)
            nextScore = self.expectMaxSearch(succGameState, nextAgent, nextDepth)[0]
            if nextScore > maximizerScore:
                maximizerScore, maximizerAction = nextScore, action

        return maximizerScore, maximizerAction

    def expector(self, gameState, agentIdx, depth):
        actions = gameState.getLegalActions(agentIdx)
        if agentIdx == gameState.getNumAgents() - 1:
            nextAgent, nextDepth = 0, depth - 1
        else:
            nextAgent, nextDepth = agentIdx + 1, depth

        expectorScore = 0
        expectorAction = Directions.STOP
        for action in actions:
            succGameState = gameState.generateSuccessor(agentIdx, action)
            expectorScore += self.expectMaxSearch(succGameState, nextAgent, nextDepth)[0]
        expectorScore /= len(actions)
        return expectorScore, expectorAction

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    # Useful information you can extract from a GameState (pacman.py)
    currentPacPos = currentGameState.getPacmanPosition()
    Food = currentGameState.getFood()
    GhostStates = currentGameState.getGhostStates()
    ScaredTimes = [ghostState.scaredTimer for ghostState in GhostStates]

    "*** YOUR CODE HERE ***"
    closestGhostDis = 1e8
    for GhostState in GhostStates:
        ghost_x, ghost_y = GhostState.getPosition()

        if ScaredTimes == 0:
            closestGhostDis = min(closestGhostDis, manhattanDistance((ghost_x, ghost_y), newPos))
        else:
            closestGhostDis = -5

    closestFoodDis = 1e8
    foods = Food.asList()

    if len(foods) == 0:
        closestFoodDis = 0
    for food in foods:
        closestFoodDis = min(closestFoodDis, manhattanDistance(food, currentPacPos))
    return currentGameState.getScore() - 5 / (closestGhostDis + 1) - closestFoodDis / 5

# Abbreviation
better = betterEvaluationFunction
