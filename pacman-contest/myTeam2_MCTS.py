# baselineTeam.py
# ---------------
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


# baselineTeam.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from captureAgents import CaptureAgent
import distanceCalculator
import random, time, util, sys
from game import Directions
import game
from util import nearestPoint

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'OffensiveReflexAgent', second = 'DefensiveAgent'):
  """
  This function should return a list of two agents that will form the
  team, initialized using firstIndex and secondIndex as their agent
  index numbers.  isRed is True if the red team is being created, and
  will be False if the blue team is being created.

  As a potentially helpful development aid, this function can take
  additional string-valued keyword arguments ("first" and "second" are
  such arguments in the case of this function), which will come from
  the --redOpts and --blueOpts command-line arguments to capture.py.
  For the nightly contest, however, your team will be created without
  any extra arguments, so you should make sure that the default
  behavior is what you want for the nightly contest.
  """
  return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
  """
  A base class for reflex agents that chooses score-maximizing actions
  """
 
  def registerInitialState(self, gameState):
    self.start = gameState.getAgentPosition(self.index)
    CaptureAgent.registerInitialState(self, gameState)
    if(self.red): self.midGround=int((gameState.data.layout.width-2)/2)
    else: self.midGround=int((gameState.data.layout.width-2)/2+1)
    print(self.midGround)
    self.bounds = []
    for i in range(1, gameState.data.layout.height - 1):
        if not gameState.hasWall(self.midGround, i):
            self.bounds.append((self.midGround, i))

  def chooseAction(self, gameState):
    """
    Picks among the actions with the highest Q(s,a).
    """
    actions = gameState.getLegalActions(self.index)
    actions.remove(Directions.STOP)
    # You can profile your evaluation time by uncommenting these lines
    # start = time.time()
    values=[]
    for a in actions:
        # print(a)
        value = self.monteCarlo(gameState, a, 0.05, 1, 5)
        values.append(value)
    # values = [self.evaluate(gameState, a) for a in actions]
    # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)
    # print(values)
    maxValue = max(values)
    bestActions = [a for a, v in zip(actions, values) if v == maxValue]
    print(bestActions)
    foodLeft = len(self.getFood(gameState).asList())

    eatenFoodatMoment = gameState.getAgentState(self.index).numCarrying

    if foodLeft <= 2 :
      bestDist = 9999
      for action in actions:
        successor = self.getSuccessor(gameState, action)
        pos2 = successor.getAgentPosition(self.index)
        dist = self.getMazeDistance(self.start,pos2)
        if dist < bestDist:
          bestAction = action
          bestDist = dist
      return bestAction

    return random.choice(bestActions)

  def getSuccessor(self, gameState, action):
    """
    Finds the next successor which is a grid position (location tuple).
    """
    successor = gameState.generateSuccessor(self.index, action)
    pos = successor.getAgentState(self.index).getPosition()
    if pos != nearestPoint(pos):
      # Only half a grid position was covered
      return successor.generateSuccessor(self.index, action)
    else:
      return successor

  def evaluate(self, gameState, action):
    """
    Computes a linear combination of features and feature weights
    """
    features = self.getFeatures(gameState, action)
    weights = self.getWeights(gameState, action)
    return features * weights

  def getFeatures(self, gameState, action):
    """
    Returns a counter of features for the state
    """
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    features['successorScore'] = self.getScore(successor)
    return features

  def getWeights(self, gameState, action):
    """
    Normally, weights do not depend on the gamestate.  They can be either
    a counter or a dictionary.
    """
    return {'successorScore': 1.0}

  def monteCarlo(self, gameState, action, df, dfPow, depth):
    succ = gameState.generateSuccessor(self.index, action)
    actions = succ.getLegalActions(self.index)
    actions.remove(Directions.STOP)
    if depth > 0:
        a = random.choice(actions)
        value = self.evaluate(gameState, action)
        # print(depth)
        value = value + df ** dfPow * self.monteCarlo(succ, a, df, dfPow + 1, depth - 1)
        # print('value is: ',value)
        return value
    else:
        return self.evaluate(gameState, action)


class OffensiveReflexAgent(ReflexCaptureAgent):
  """
  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.
  """
  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    foodList = self.getFood(successor).asList()    
    features['successorScore'] = -len(foodList)#self.getScore(successor)

    # Compute distance to the nearest food

    if len(foodList) > 0: # This should always be True,  but better safe than sorry
      myPos = successor.getAgentState(self.index).getPosition()
      minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
      features['distanceToFood'] = minDistance

    distance = []
    opponents = self.getOpponents(gameState)
    for o in opponents:
        opponent = gameState.getAgentState(o)
        if not opponent.isPacman:
            opponentPos = opponent.getPosition()
            if opponentPos is not None:
                print('ghost distance is:',self.getMazeDistance(myPos, opponentPos))
                if((self.getMazeDistance(myPos, opponentPos))==0):
                    distance.append(1)
                else:
                    distance.append(1/self.getMazeDistance(myPos, opponentPos))
    print('distance is:',distance)
    if len(distance) > 0:
        features['ghostDistance'] = min(distance)    
    else: 
        features['ghostDistance'] = 0

    capsuleList = set(gameState.data.capsules) - set(self.getCapsulesYouAreDefending(gameState))
    if len(capsuleList) > 0:
        minCapsuleDistance = 99999
        for c in capsuleList:
            distance = self.getMazeDistance(myPos, c)  
            if distance < minCapsuleDistance:     
                minCapsuleDistance = distance         
        features['capsDistance'] = minCapsuleDistance
    else:
        features['capsDistance'] = 0
    
    if action == Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]:
        features['reverse'] = -1
    else:
        features['reverse'] = 0

    dist = self.getMazeDistance(self.start,myPos)
    features['reachBound'] = dist

    return features

  def getWeights(self, gameState, action):
    
    opponents = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
    visible = list(filter(lambda x: not x.isPacman and x.getPosition() != None, opponents))
    
    eatenFoodatMoment = gameState.getAgentState(self.index).numCarrying

    isScared = False
    for agent in visible:
        if agent.scaredTimer > 10:
            isScared = True

    if((eatenFoodatMoment>=6) and not isScared):
        return {'reachBound':-80,'ghostDistance':-900}
    if(isScared):
       {'successorScore': 100, 'distanceToFood': -2,'ghostDistance':0,'capsDistance':0,'reverse':0,'reachBound':-0.8}

    return {'successorScore': 100, 'distanceToFood': -1,'ghostDistance':-100,'capsDistance':-2,'reverse':0,'reachBound':-0.0}
class AlphaBetaAgent(CaptureAgent):

    # Register initial state for AlphaBetaAgent include start position of the
    # agent, closest food position to the center and timer for defensive agent,
    # and mid point for center position.
    def registerInitialState(self, gameState):
        CaptureAgent.registerInitialState(self, gameState)
        self.start = gameState.getAgentPosition(self.index)
        self.closestFoodToCenter = None
        self.timer = 0

        # Setting up center position and check if the center position has a wall.
        midHeight = gameState.data.layout.height//2
        midWidth = gameState.data.layout.width//2
        while(gameState.hasWall(midWidth, midHeight)):
            midHeight -= 1
            if midHeight == 0:
                midHeight = gameState.data.layout.height
        self.midPoint = (midWidth, midHeight)

        # Register team's agent.
        agentsTeam = []
        agentsLen = gameState.getNumAgents()
        i = self.index
        while len(agentsTeam) < (agentsLen//2):
            agentsTeam.append(i)
            i += 2
            if i >= agentsLen:
                i = 0
        agentsTeam.sort()
        self.registerTeam(agentsTeam)

    # Alphabeta algorithm with the usual algorithm added with visible agents.
    # If the agent are not in visible agents, it will continue the loop to the
    # next agent.
    def alphabeta(self, gameState, action, agent, mdepth, alpha, beta, visibleAgents):

        # restart the agent number if it passed the agents length
        if agent >= gameState.getNumAgents():
            agent = 0

        # add the depth if the alpha beta done a single loop
        if agent == self.index:
            mdepth += 1

        # pass the agent if it is not on the visible agents.
        if agent not in visibleAgents:
            return self.alphabeta(gameState, action, agent + 1, mdepth, alpha, beta, visibleAgents)

        # evaluate the gameState if the depth is 1 or the game is over and its the current agent.
        if mdepth == 1 or gameState.isOver():
            if agent == self.index:
                return self.evaluate(gameState, action)
            else:
                self.alphabeta(gameState, action, agent + 1, mdepth, alpha, beta, visibleAgents)

        legalActions = gameState.getLegalActions(agent)
        if agent in self.agentsOnTeam:
            v = float("-inf")
            for action in legalActions:
                v = max(v, self.alphabeta(gameState.generateSuccessor(agent, action), action, agent + 1,mdepth, alpha, beta, visibleAgents))
                if v > beta:
                    return v
                alpha = max(alpha, v)
            return v
        else:
            v = float("inf")
            for action in legalActions:
                v = min(v, self.alphabeta(gameState.generateSuccessor(agent, action), action, agent + 1,mdepth, alpha, beta, visibleAgents))
                if v < alpha:
                    return v
                beta = min(beta, v)
            return v
    def getFeatures(self, gameState, action):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        features['successorScore'] = self.getScore(gameState)
        return features


    def getWeights(self, gameState, actton):
        """
        Normally, weights do not depend on the gamestate.  They can be either
        a counter or a dictionary.
        """
        return {'successorScore': 1.0}


    def evaluate(self, gameState,action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.getFeatures(gameState, action)
        weights = self.getWeights(gameState, action)
        return features * weights
class DefensiveAgent(AlphaBetaAgent):
    # The basic of defensive agent is to go around the nearest food to the center to detect an enemy
    # and chase one if found.
    def chooseAction(self, gameState):
        start = time.time()

        # Get all visible agents
        allAgents = range(0, gameState.getNumAgents())
        visibleAgents = [a for a in allAgents if gameState.getAgentState(a).getPosition() is not None]

        # Start alpha beta algorithm
        v = (float("-inf"), 'None')
        alpha = float('-inf')
        beta = float('inf')
        legalActions = gameState.getLegalActions(self.index)
        for action in legalActions:
            if action == 'Stop':
                continue
            v = max(v, (self.alphabeta(gameState.generateSuccessor(self.index, action), action, self.index+1, 0, alpha, beta, visibleAgents), action))
            if v[0] > beta:
                #print "Agent {0} chose {1} with value {2}".format(self.index, v[1], v[0])
                #print 'Execution time for agent %d: %.4f' % (self.index, time.time() - start)
                return v[1]
            alpha = max(alpha, v[0])
            #print "Agent {0} chose {1} with value {2}".format(self.index, v[1], v[0])
            #print 'Execution time for agent %d: %.4f' % (self.index, time.time() - start)

        # Minus the timer for the patrol function.
        self.timer -= 1

        return v[1]

    def getFeatures(self, gameState, action):
        features = util.Counter()
        myState = gameState.getAgentState(self.index)
        myPos = myState.getPosition()
        foodList = self.getFoodYouAreDefending(gameState).asList()

        # Computes distance to invaders we can see
        enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
        features['numInvaders'] = len(invaders)

        # Check if any opponent is found.
        if len(invaders) > 0:
            dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
            features['invaderDistance'] = max(dists)
        else:
            # If no opponent is found, patrol around the 3 nearest food to the center.
            # if the nearest food to center is set, calculate the distance.
            if self.closestFoodToCenter:
                dist = self.getMazeDistance(myPos, self.closestFoodToCenter)
            else:
                dist = None

            # Recalculate the 3 nearest food when it's already 20 actions or the food
            # is reached.
            if self.timer == 0 or dist == 0:
                self.timer = 20
                foods = []
                for food in foodList:
                    foods.append((self.getMazeDistance(self.midPoint, food), food))
                foods.sort()
                chosenFood = random.choice(foods[:3])
                self.closestFoodToCenter = chosenFood[1]
            dist = self.getMazeDistance(myPos, self.closestFoodToCenter)
            features['invaderDistance'] = dist
        return features

    def getWeights(self, gameState, action):
        return {'numInvaders': -1000, 'invaderDistance': -10, 'stop': -100, 'reverse': -2}
