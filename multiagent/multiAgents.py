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
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

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

        "*** YOUR CODE HERE ***"
        # Example evaluation function that considers distance to food and ghosts
        score = successorGameState.getScore()
        foodList = newFood.asList()
        if foodList:
            closestFoodDistance = min(manhattanDistance(newPos, food) for food in foodList)
            score += 1.0 / closestFoodDistance
        for ghostState in newGhostStates:
            ghostPos = ghostState.getPosition()
            ghostDistance = manhattanDistance(newPos, ghostPos)
            if ghostDistance > 0:
                if ghostState.scaredTimer > 0:
                    score += 1.0 / ghostDistance
                else:
                    score -= 1.0 / ghostDistance

        return score

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

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
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
        legalActions = gameState.getLegalActions(0)
        if not legalActions:
            return Directions.STOP
        bestAction = None
        bestValue = float('-inf')
        for action in legalActions:
            successorState = gameState.generateSuccessor(0, action)
            value = self.value(successorState, self.depth, 1)
            if value > bestValue:
                bestValue = value
                bestAction = action
        return bestAction
        util.rxaiseNotDefined()
    def value(self, gameState: GameState, depth: int, agentIndex: int):
        """
        Returns the value of the game state for the given agent index at the specified depth.
        """
        if gameState.isWin() or gameState.isLose() or depth == 0:
            return self.evaluationFunction(gameState)

        numAgents = gameState.getNumAgents()
        nextAgentIndex = (agentIndex + 1) % numAgents
        nextDepth = depth - 1 if nextAgentIndex == 0 else depth

        legalActions = gameState.getLegalActions(agentIndex)
        if agentIndex == 0:
            # Pacman's turn (maximizing player)
            bestValue = float('-inf')
            for action in legalActions:
                successorState = gameState.generateSuccessor(agentIndex, action)
                value = self.value(successorState, nextDepth, nextAgentIndex)
                bestValue = max(bestValue, value)
            return bestValue
        else:
            # Ghost's turn (minimizing player)
            bestValue = float('inf')
            for action in legalActions:
                successorState = gameState.generateSuccessor(agentIndex, action)
                value = self.value(successorState, nextDepth, nextAgentIndex)
                bestValue = min(bestValue, value)
            return bestValue
    # def evaluationFunction(self, currentGameState: GameState):
    #     Pos = currentGameState.getPacmanPosition()
    #     Food = currentGameState.getFood()
    #     GhostStates = currentGameState.getGhostStates()
    #     ScaredTimes = [ghostState.scaredTimer for ghostState in GhostStates]


    #     foodList = Food.asList()
    #     if not foodList:
    #         return float('inf')
    #     closestFoodDistance = min(manhattanDistance(Pos, food) for food in foodList)
    #     score = currentGameState.getScore() + 1.0 / closestFoodDistance
    #     for ghostState in GhostStates:
    #         ghostPos = ghostState.getPosition()
    #         ghostDistance = manhattanDistance(Pos, ghostPos)
    #         if ghostDistance > 0:
    #             if ghostState.scaredTimer > 0:
    #                 score += 1.0 / ghostDistance
    #             else:
    #                 score -= 1.0 / ghostDistance
        



class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        ## exprience: 
        # 1.how to transfer the value from root to the next level
        # 2.how to use alpha and beta to prune the search tree
        legalActions = gameState.getLegalActions(0)
        if not legalActions:
            return Directions.STOP
        bestAction = None
        bestValue = float('-inf')
        for action in legalActions:
            successorState = gameState.generateSuccessor(0, action)
            value = self.value(successorState, self.depth, 1,bestValue, float('inf'))
            if value > bestValue:
                bestValue = value
                bestAction = action
        return bestAction
        util.raiseNotDefined()
    def value(self, gameState: GameState, depth: int, agentIndex: int, alpha: float, beta: float):
        if gameState.isWin() or gameState.isLose() or depth == 0:
            return self.evaluationFunction(gameState)
        numAgents = gameState.getNumAgents()
        nextAgentIndex = (agentIndex + 1) % numAgents
        nextDepth = depth - 1 if nextAgentIndex == 0 else depth
        legalActions = gameState.getLegalActions(agentIndex)
        if agentIndex == 0:
            bestvalue = float('-inf')
            for action in legalActions:
                successorState = gameState.generateSuccessor(agentIndex, action)
                value = self.value(successorState, nextDepth, nextAgentIndex, alpha, beta)
                bestvalue = max(bestvalue, value)
                if bestvalue > beta:
                    return bestvalue
                alpha = max(alpha, bestvalue)
        else:
            bestvalue = float('inf')
            for action in legalActions:
                successorState = gameState.generateSuccessor(agentIndex, action)
                value = self.value(successorState, nextDepth, nextAgentIndex, alpha, beta)
                bestvalue = min(bestvalue, value)
                if bestvalue < alpha:
                    return bestvalue
                beta = min(beta, bestvalue)
        return bestvalue

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
        legalActions = gameState.getLegalActions(0)
        if not legalActions:
            return Directions.STOP
        bestAction = None
        bestValue = float('-inf')
        for action in legalActions:
            successorState = gameState.generateSuccessor(0, action)
            value = self.value(successorState, self.depth, 1)
            if value > bestValue:
                bestValue = value
                bestAction = action
        return bestAction
        util.rxaiseNotDefined()
    def value(self, gameState: GameState, depth: int, agentIndex: int):
        """
        Returns the value of the game state for the given agent index at the specified depth.
        """
        if gameState.isWin() or gameState.isLose() or depth == 0:
            score = self.evaluationFunction(gameState)
            if gameState.isWin():
                return score + 1000  # Reward for winning
            elif gameState.isLose():
                return score - 1000
            return score
                

        numAgents = gameState.getNumAgents()
        nextAgentIndex = (agentIndex + 1) % numAgents
        nextDepth = depth - 1 if nextAgentIndex == 0 else depth

        legalActions = gameState.getLegalActions(agentIndex)
        if agentIndex == 0:
            # Pacman's turn (maximizing player)
            bestValue = float('-inf')
            for action in legalActions:
                successorState = gameState.generateSuccessor(agentIndex, action)
                value = self.value(successorState, nextDepth, nextAgentIndex)
                bestValue = max(bestValue, value)
            return bestValue
        else:
            sum = 0
            for action in legalActions:
                successorState = gameState.generateSuccessor(agentIndex, action)
                value = self.value(successorState, nextDepth, nextAgentIndex)
                sum += value
            return sum / len(legalActions) if legalActions else 0   
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: Considers food distance, ghost positions, scared times, and capsules
    """
    "*** YOUR CODE HERE ***"
    Pos = currentGameState.getPacmanPosition()
    Food = currentGameState.getFood()
    GhostStates = currentGameState.getGhostStates()
    ScaredTimes = [ghostState.scaredTimer for ghostState in GhostStates]
    
    # 基础分数
    score = currentGameState.getScore()
    
    # 食物相关评估
    foodList = Food.asList()
    if foodList:
        closestFoodDistance = min(manhattanDistance(Pos, food) for food in foodList)
        # 避免除零错误
        if closestFoodDistance > 0:
            score += 10.0 / closestFoodDistance
        # else:
        #     score += 100  # 如果在食物上，给予高分
        
        # 鼓励吃更多食物
        #score += 100 * (currentGameState.getNumFood() - len(foodList))
    
    # Capsules 评估
    capsules = currentGameState.getCapsules()
    if capsules:
        closestCapsuleDistance = min(manhattanDistance(Pos, capsule) for capsule in capsules)
        if closestCapsuleDistance > 0:
            score += 50 / closestCapsuleDistance #50

    
    # 鬼魂相关评估
    for ghostState in GhostStates:
        ghostPos = ghostState.getPosition()
        ghostDistance = manhattanDistance(Pos, ghostPos)
        
        if ghostDistance > 0:
            if ghostState.scaredTimer > 0:
                # 鬼魂被吓到时，追击它们
                score += 200.0 / ghostDistance
            else:
                # 正常鬼魂，保持距离
                if ghostDistance < 2:
                    score -= 100  # 太近时严重扣分 500
                else:
                    score -= 10.0 / ghostDistance
        else:
            # 与鬼魂在同一位置
            if ghostState.scaredTimer > 0:
                score += 1000  # 吃掉被吓到的鬼魂
            else:
                score -= 1000  # 被正常鬼魂吃掉
    
    return score
# Abbreviation
better = betterEvaluationFunction
