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
from game import Directions,Actions
import game
from util import nearestPoint

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'AttackAgent', second = 'AttackAgent'):
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
    # ##print('This is what i want', self.initial_food)
    
    self.start = gameState.getAgentPosition(self.index)
    CaptureAgent.registerInitialState(self, gameState)
    if(self.red): self.midGround=int((gameState.data.layout.width-2)/2)
    else: self.midGround=int((gameState.data.layout.width-2)/2+1)
    ##print(self.midGround)




  # def chooseAction(self, gameState):
  #   """
  #   Picks among the actions with the highest Q(s,a).
  #   """
  #   actions = gameState.getLegalActions(self.index)
  #   actions.remove(Directions.STOP)
  #   # You can profile your evaluation time by uncommenting these lines
  #   # start = time.time()
  #   values=[]
  #   for a in actions:
  #       ##print(a)
  #       value=self.evaluate(gameState,a)
  #       # value = self.monteCarlo(gameState, a, 0.3, 1, 5)
  #       values.append(value)
  #   # values = [self.evaluate(gameState, a) for a in actions]
  #   # ##print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)
    
  #   maxValue = max(values)
  #   bestActions = [a for a, v in zip(actions, values) if v == maxValue]
  #   # ##print(bestActions)
  #   foodLeft = len(self.getFood(gameState).asList())

  #   eatenFoodatMoment = gameState.getAgentState(self.index).numCarrying

  #   if foodLeft <= 2 :
  #     bestDist = 9999
  #     for action in actions:
  #       successor = self.getSuccessor(gameState, action)
  #       pos2 = successor.getAgentPosition(self.index)
  #       dist = self.getMazeDistance(self.start,pos2)
  #       if dist < bestDist:
  #         bestAction = action
  #         bestDist = dist
  #     return bestAction

  #   return random.choice(bestActions)

  # def getSuccessor(self, gameState, action):
  #   """
  #   Finds the next successor which is a grid position (location tuple).
  #   """
  #   successor = gameState.generateSuccessor(self.index, action)
  #   pos = successor.getAgentState(self.index).getPosition()
  #   if pos != nearestPoint(pos):
  #     # Only half a grid position was covered
  #     return successor.generateSuccessor(self.index, action)
  #   else:
  #     return successor

  # def evaluate(self, gameState, action):
  #   """
  #   Computes a linear combination of features and feature weights
  #   """
  #   features = self.getFeatures(gameState, action)
  #   weights = self.getWeights(gameState, action)
    
  #   return features * weights

  # def getFeatures(self, gameState, action):
  #   """
  #   Returns a counter of features for the state
  #   """
  #   features = util.Counter()
  #   successor = self.getSuccessor(gameState, action)
  #   features['successorScore'] = self.getScore(successor)
  #   return features

  # def getWeights(self, gameState, action):
  #   """
  #   Normally, weights do not depend on the gamestate.  They can be either
  #   a counter or a dictionary.
  #   """
  #   return {'successorScore': 1.0}

  # def monteCarlo(self, gameState, action, df, dfPow, depth):
  #   succ = gameState.generateSuccessor(self.index, action)
  #   actions = succ.getLegalActions(self.index)
  #   actions.remove(Directions.STOP)
  #   if depth > 0:
  #       a = random.choice(actions)
  #       value = self.evaluate(gameState, action)
  #       # ##print(depth)
  #       value = value + df ** dfPow * self.monteCarlo(succ, a, df, dfPow + 1, depth - 1)
  #       # ##print('value is: ',value)
  #       return value
  #   else:
  #       return self.evaluate(gameState, action)

##### class for registring the offensive agent
# class OffensiveReflexAgent(ReflexCaptureAgent):
#   """
#   A reflex agent that seeks food. This is an agent
#   we give you to get an idea of what an offensive agent might look like,
#   but it is by no means the best or only way to build an offensive agent.
#   """

  


#   def getFeatures(self, gameState, action):
#     features = util.Counter()
#     position = gameState.getAgentPosition(self.index)
#     nextGameState = gameState.generateSuccessor(self.index, action)
#     nextPosition = nextGameState.getAgentPosition(self.index)
#     food = self.getFood(gameState)
#     capsules = self.getCapsules(gameState)

#     walls = gameState.getWalls()
#     wallsList = walls.asList()
#     mazeSize = walls.width * walls.height

#     enemyIndices = self.getOpponents(gameState)

#     attackablePacmen = [gameState.getAgentPosition(i) for i in enemyIndices if self.isPacman(gameState, i) and self.isGhost(gameState, self.index) and not self.isScared(gameState, self.index)]
#     scaredGhostLocations = [gameState.getAgentPosition(i) for i in self.getOpponents(gameState) if self.isScared(gameState, i) and self.isGhost(gameState, i)]
#     goalPositions = set(food.asList() + attackablePacmen + capsules + scaredGhostLocations)

#     enemyGhostLocations = [gameState.getAgentPosition(i) for i in enemyIndices if (self.isGhost(gameState, i) and not self.isScared(gameState, i) and (self.getMazeDistance(position, gameState.getAgentPosition(i)) < 6))]
#     # enemyGhostLocations.extend(self.validSurroundingPositionsTo(gameState, enemyGhostLocations, wallsList))
#     # enemyGhostLocations.extend(self.validSurroundingPositionsTo(gameState, enemyGhostLocations, wallsList))

#     features['successorScore']=( food.count())

#     if action == Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]:
#       features['reverse'] = -1
#     else:
#       features['reverse'] = 0
#     # features.append( self.getFoodYouAreDefending(gameState).count() / self.initial_defending_food)
#     closestGhost = len(self.aStarSearch(nextPosition, nextGameState, list(enemyGhostLocations))) if enemyGhostLocations else 0
#     if(closestGhost>1 and closestGhost<=5):
      
#       features['ghostDistance']=(1/closestGhost)
#     if(closestGhost>6):
#       features['ghostDistance']=1
#     else:
#       features['ghostDistance']=0

#     avoidPositions = set(enemyGhostLocations)

#     aStar_food_path = self.aStarSearch(nextPosition, nextGameState, list(goalPositions), avoidPositions=avoidPositions)

#     features['distanceToFood']=((len(aStar_food_path) if aStar_food_path is not None else mazeSize))

#     bounds = []
#     for i in range(1, gameState.data.layout.height - 1):
#         if not gameState.hasWall(self.midGround, i):
#             bounds.append((self.midGround, i))

#     goHomePath=self.aStarSearch(nextPosition, nextGameState, list(bounds), avoidPositions=avoidPositions)

#     features['reachBound'] =len (goHomePath)

#     # if action == Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]:
#     #     features['reverse'] = -1
#     # else:
#     #     features['reverse'] = 0

#     ##print(features)

#     return features

#   def isGhost(self, gameState, index):
#     position = gameState.getAgentPosition(index)
#     if position is None:
#         return False
#     return not (gameState.isOnRedTeam(index) ^ (position[0] < gameState.getWalls().width / 2))

#   def isScared(self, gameState, index):
#     isScared = bool(gameState.data.agentStates[index].scaredTimer)
#     return isScared


#   def isPacman(self, gameState, index):
#     position = gameState.getAgentPosition(index)
#     if position is None:
#         return False
#     return not (gameState.isOnRedTeam(index) ^ (position[0] >= gameState.getWalls().width / 2))

#   def aStarSearch(self, startPosition, gameState, goalPositions, avoidPositions=[], returnPosition=False):
#     walls = gameState.getWalls()
#     width = walls.width
#     height = walls.height
#     walls = walls.asList()

#     actions = [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]
#     actionVectors = [Actions.directionToVector(action) for action in actions]
#     actionVectors = [tuple(int(number) for number in vector) for vector in actionVectors]

#     currentPosition, currentPath, currentTotal = startPosition, [], 0
#     queue = util.PriorityQueueWithFunction(lambda entry: entry[2]+width * height if entry[0] in avoidPositions else 0+min(util.manhattanDistance(entry[0], endPosition) for endPosition in goalPositions))

#     # Keeps track of visited positions
#     visited = set([currentPosition])

#     while currentPosition not in goalPositions:

#         possiblePositions = [((currentPosition[0] + vector[0], currentPosition[1] + vector[1]), action) for vector, action in zip(actionVectors, actions)]
#         legalPositions = [(position, action) for position, action in possiblePositions if position not in walls]

#         for position, action in legalPositions:
#             if position not in visited:
#                 visited.add(position)
#                 queue.push((position, currentPath + [action], currentTotal + 1))
#         if len(queue.heap) == 0:
#             return None
#         else:
#             currentPosition, currentPath, currentTotal = queue.pop()

#     if returnPosition:
#         return currentPath, currentPosition
#     else:
#         return currentPath

#   def getWeights(self, gameState, action):
#     opponents = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
#     visible = list(filter(lambda x: not x.isPacman and x.getPosition() != None, opponents))
    
#     eatenFoodatMoment = gameState.getAgentState(self.index).numCarrying
#     isScared = False
#     someOneVisible=False
#     for agent in visible:
#       print(gameState.getAgentPosition(self.index),agent.getPosition())
#       if(self.getMazeDistance(gameState.getAgentPosition(self.index),agent.getPosition())<4):
#         someOneVisible=True
#         if agent.scaredTimer > 10:
#             isScared = True

#     if((eatenFoodatMoment>=5) and not someOneVisible and not isScared):
#       print('food zyaada khaaya, koi nahi pass and koi nahi dara ')
#       return{'distanceToFood':-3,'ghostDistance':-100,'reachBound':-1,'reverse':300}

#     if((eatenFoodatMoment>=5) and not isScared):
#       print('food zyaada khaaya, koi nahi dara par koi hai ')
#       return {'reachBound':-10,'ghostDistance':-100}

#     if(isScared):
#       print('parvaah nahi')
#       return{'distanceToFood':-3,'ghostDistance':0,'reachBound':-0.5,'reverse':-0.3}


#     if(someOneVisible):
#       if(not gameState.getAgentState(self.index).isPacman):
#         print('bhoot se bhaago par ghar ho')
#         return {'ghostDistance':-100,'reachBound':-10,'reverse':5000}
#       else:
#         print('bhoot se bhaago')
#         return {'ghostDistance':-100,'reachBound':-10,}
#     print('run casually')
#     return{'distanceToFood':-3,'ghostDistance':-100,'reachBound':-0.5,'reverse':-0.3}

# class DefensiveReflexAgent(ReflexCaptureAgent):
#   def getFeatures(self, gameState, action):
#     features = util.Counter()
#     successor = self.getSuccessor(gameState, action)

#     myState = successor.getAgentState(self.index)
#     myPos = myState.getPosition()
#     features['onDefense'] = 1
#     if myState.isPacman: features['onDefense'] = 0
#     enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
#     invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
#     features['numInvaders'] = len(invaders)
#     if len(invaders) > 0:
#       dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
#       features['invaderDistance'] = min(dists)

#     if action == Directions.STOP: features['stop'] = 1
#     rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
#     if action == rev: features['reverse'] = 1

#     return features

#   def getWeights(self, gameState, action):
#     return {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -10, 'stop': -100, 'reverse': -2}

class ActionNew():
    def getSuccessor(self, gameState, action):
        successor = gameState.generateSuccessor(self.index, action)
        pos = successor.getAgentState(self.index).getPosition()
        if pos != nearestPoint(pos):
            # Only half a grid position was covered
            return successor.generateSuccessor(self.index, action)
        else:
            return successor

    def evaluate(self, gameState, action):
        features = self.getFeatures(gameState, action)
        weights = self.getWeights(gameState, action)
        return features * weights

    def getFeatures(self, gameState, action):
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        features['successorScore'] = self.agent.getScore(successor)
        return features

    def getWeights(self, gameState, action):
        return {'successorScore': 1.0}


####CLass for registering the defeder
class Defender(ReflexCaptureAgent):
    def __init__(self, index):
        CaptureAgent.__init__(self, index)

    def registerInitialState(self, gameState):
        CaptureAgent.registerInitialState(self, gameState)
        self.DefenceStatus = getDefensiveActions(self, self.index, gameState)
        self.OffenceStatus = getAttackActions(self, self.index, gameState)
        #self.OffenceStatus = OffensiveReflexAgent(self) 

    def chooseAction(self, gameState):
        
        self.enemies = self.getOpponents(gameState)
        invaders = [a for a in self.enemies if gameState.getAgentState(a).isPacman]
        numInvaders = len(invaders)
        scaredTimes = [gameState.getAgentState(enemy).scaredTimer for enemy in self.enemies]
        timeLeft = gameState.data.timeleft
        #if (len(self.getFood(gameState).asList()) <= 2):
         
         # return self.DefenceStatus.chooseAction(gameState)
        
        if self.getScore(gameState) > 15:
          
          return self.DefenceStatus.chooseAction(gameState)
        else:
          if timeLeft >= 700:
            print('greater than 1000')
            return self.OffenceStatus.chooseAction(gameState)
          else:
            print('else of the defense 700')
            return self.DefenceStatus.chooseAction(gameState)
class getDefensiveActions(ActionNew):
    def __init__(self, agent, index, gameState):
        self.index = index
        self.agent = agent
        self.DenfendList = {}
        self.middle = 0

        if self.agent.red:
            self.middle = (gameState.data.layout.width - 2) // 2
        else:
            self.middle = ((gameState.data.layout.width - 2) // 2) + 1
        self.boundary = []
        for i in range(1, gameState.data.layout.height - 1):
            if not gameState.hasWall(self.middle, i):
                self.boundary.append((self.middle, i))

        self.target = None
        self.lastObservedFood = None
        self.DefenceProbability(gameState)


    def aStarSearch(self, startPosition, gameState, goalPositions, avoidPositions=[], returnPosition=False):
      walls = gameState.getWalls()
      width = walls.width
      height = walls.height
      walls = walls.asList()

      actions = [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]
      actionVectors = [Actions.directionToVector(action) for action in actions]
      actionVectors = [tuple(int(number) for number in vector) for vector in actionVectors]

      currentPosition, currentPath, currentTotal = startPosition, [], 0
      queue = util.PriorityQueueWithFunction(lambda entry: entry[2]+width * height if entry[0] in avoidPositions else 0+min(util.manhattanDistance(entry[0], endPosition) for endPosition in goalPositions))

      # Keeps track of visited positions
      visited = set([currentPosition])

      while currentPosition not in goalPositions:

          possiblePositions = [((currentPosition[0] + vector[0], currentPosition[1] + vector[1]), action) for vector, action in zip(actionVectors, actions)]
          legalPositions = [(position, action) for position, action in possiblePositions if position not in walls]

          for position, action in legalPositions:
              if position not in visited:
                  visited.add(position)
                  queue.push((position, currentPath + [action], currentTotal + 1))
          if len(queue.heap) == 0:
              return None
          else:
              currentPosition, currentPath, currentTotal = queue.pop()

      if returnPosition:
          return currentPath, currentPosition
      else:
          return currentPath





      # def allSimulation(self, depth, gameState, decay):
      #     new_state = gameState.deepCopy()
      #     if depth == 0:
      #         result_list = []
      #         actions = new_state.getLegalActions(self.index)
      #         actions.remove(Directions.STOP)

      #         reversed_direction = Directions.REVERSE[new_state.getAgentState(self.index).configuration.direction]
      #         if reversed_direction in actions and len(actions) > 1:
      #             actions.remove(reversed_direction)
      #         a = random.choice(actions)
      #         next_state = new_state.generateSuccessor(self.index, a)
      #         result_list.append(self.evaluate(next_state, Directions.STOP))
      #         return max(result_list)

      #     result_list = []
      #     actions = new_state.getLegalActions(self.index)
      #     current_direction = new_state.getAgentState(self.index).configuration.direction

      #     reversed_direction = Directions.REVERSE[current_direction]
      #     if reversed_direction in actions and len(actions) > 1:
      #         actions.remove(reversed_direction)
      #     for a in actions:
      #         next_state = new_state.generateSuccessor(self.index, a)
      #         result_list.append(
      #             self.evaluate(next_state, Directions.STOP) + decay * self.allSimulation(depth - 1, next_state, decay))
      #     return max(result_list)
      # def MTCS(self, depth, gameState, decay):
      #     new_state = gameState.deepCopy()
      #     value = self.evaluate(new_state, Directions.STOP)
      #     decay_index = 1
      #     while depth > 0:

      #         actions = new_state.getLegalActions(self.index)
      #         current_direction = new_state.getAgentState(self.agent.index).configuration.direction
      #         reversed_direction = Directions.REVERSE[new_state.getAgentState(self.agent.index).configuration.direction]
      #         if reversed_direction in actions and len(actions) > 1:
      #             actions.remove(reversed_direction)
      #         # Randomly chooses a valid action
      #         a = random.choice(actions)
      #         # Compute new state and update depth
      #         new_state = new_state.generateSuccessor(self.agent.index, a)
      #         value = value + decay ** decay_index * self.evaluate(new_state, Directions.STOP)
      #         depth -= 1
      #         decay_index += 1
      #     return value
      # def chooseAction(self, gameState):
          start = time.time()
          actions = gameState.getLegalActions(self.agent.index)
          actions.remove(Directions.STOP)
          feasible = []
          for a in actions:
              value = 0
              value = self.allSimulation(2, gameState.generateSuccessor(self.agent.index, a), 0.7)
              feasible .append(value)
          bestAction = max(feasible)
          possibleChoice = list(filter(lambda x: x[0] == bestAction, zip(feasible, actions)))
          return random.choice(possibleChoice)[1]

    def isGhost(self, gameState, index):
      position = gameState.getAgentPosition(index)
      if position is None:
          return False
      return not (gameState.isOnRedTeam(index) ^ (position[0] < gameState.getWalls().width / 2))

    def isScared(self, gameState, index):
      isScared = bool(gameState.data.agentStates[index].scaredTimer)
      return isScared


    def isPacman(self, gameState, index):
      position = gameState.getAgentPosition(index)
      if position is None:
          return False
      return not (gameState.isOnRedTeam(index) ^ (position[0] >= gameState.getWalls().width / 2))


    def DefenceProbability(self, gameState):
        total = 0

        for position in self.boundary:
            food = self.agent.getFoodYouAreDefending(gameState).asList()
            closestFoodDistance = 0
            if len(food) > 0:
              closestFoodDistance=min([self.agent.getMazeDistance(position,f) for f in food])
            else:
              closestFoodDistance = 1

            if closestFoodDistance == 0:
              closestFoodDistance = 1
            self.DenfendList[position] = 1.0 / float(closestFoodDistance)
            total += self.DenfendList[position]
        if total == 0:
            total = 1
        for x in self.DenfendList.keys():
            self.DenfendList[x] = float(self.DenfendList[x]) // float(total)

    def selectPatrolTarget(self):

        maxProb=max([self.DenfendList[x] for x in self.DenfendList.keys()])
        bestTarget = list(filter(lambda x: self.DenfendList[x] == maxProb, self.DenfendList.keys()))
        return random.choice(bestTarget)

    def chooseAction(self, gameState):
      if (len(self.agent.getFood(gameState).asList()) <= 2):
        
        # print('defence is HERE INSIDE 2 FOOD')
        bounds = []
        for i in range(1, gameState.data.layout.height - 1):
            if not gameState.hasWall(self.middle, i):
                bounds.append((self.middle, i))

        # boundaryMin = 99999

        # foodLeft = len(self.agent.getFood(gameState).asList())

        # if foodLeft <= 2 :
        actions = gameState.getLegalActions(self.index)
        actions.remove(Directions.STOP)
        mingoHomePath = 9999

        for action in actions:
          position = gameState.getAgentPosition(self.index)
          nextGameState = gameState.generateSuccessor(self.index, action)
          nextPosition = nextGameState.getAgentPosition(self.index)
          food = self.agent.getFood(gameState)
          capsules = self.agent.getCapsules(gameState)

          walls = gameState.getWalls()
          wallsList = walls.asList()
          mazeSize = walls.width * walls.height

          enemyIndices = self.agent.getOpponents(gameState)

          attackablePacmen = [gameState.getAgentPosition(i) for i in enemyIndices if self.isPacman(gameState, i) and self.isGhost(gameState, self.index) and not self.isScared(gameState, self.index)]
          scaredGhostLocations = [gameState.getAgentPosition(i) for i in self.agent.getOpponents(gameState) if self.isScared(gameState, i) and self.isGhost(gameState, i)]
          goalPositions = set(food.asList() + attackablePacmen + capsules + scaredGhostLocations)

          enemyGhostLocations = [gameState.getAgentPosition(i) for i in enemyIndices if (self.isGhost(gameState, i) and not self.isScared(gameState, i) and (self.agent.getMazeDistance(position, gameState.getAgentPosition(i)) < 6))]
          avoidPositions = set(enemyGhostLocations)
          goHomePath = len(self.aStarSearch(nextPosition, nextGameState, list(bounds), avoidPositions=avoidPositions))
          if goHomePath < mingoHomePath:
            mingoHomePath = goHomePath
            bestAction = action
          # successor = self.getSuccessor(gameState, action)
          # pos2 = successor.getAgentPosition(self.index)
          # for i in range(len(bounds)):
          #   disBoundary = self.agent.getMazeDistance(pos2, self.start)
          #   if disBoundary < boundaryMin:
          #     boundaryMin = disBoundary
          # #bestAction = action 
          # #dist = self.agent.getMazeDistance(self.start,pos2)
        print('returning my best def action',bestAction, 'for - ', self.index)    
        return bestAction




      DefendingList = self.agent.getFoodYouAreDefending(gameState).asList()
      if self.lastObservedFood and len(self.lastObservedFood) != len(DefendingList):
          self.DefenceProbability(gameState)
      myPos = gameState.getAgentPosition(self.index)
      if myPos == self.target:
          self.target = None
      enemies=[gameState.getAgentState(i) for i in self.agent.getOpponents(gameState)]
      inRange = list(filter(lambda x: x.isPacman and x.getPosition() != None,enemies))
      if len(inRange)>0:
          minDis = 99999
          minPac = None
          #eneDis,enemyPac = min([(self.agent.getMazeDistance(myPos,x.getPosition()), x) for x in inRange])
          for x in inRange:
            eneDis = self.agent.getMazeDistance(myPos, x.getPosition())
            if eneDis < minDis:
              minDis = eneDis
              minPac = x
          enemyPac = minPac
          self.target=enemyPac.getPosition()
      elif self.lastObservedFood != None:
          eaten = set(self.lastObservedFood) - set(self.agent.getFoodYouAreDefending(gameState).asList())
          if len(eaten)>0:
             closestFood, self.target = min([(self.agent.getMazeDistance(myPos,f),f) for f in eaten])
      self.lastObservedFood = self.agent.getFoodYouAreDefending(gameState).asList()
      if self.target == None and len(self.agent.getFoodYouAreDefending(gameState).asList()) <= 4:
          food = self.agent.getFoodYouAreDefending(gameState).asList() + self.agent.getCapsulesYouAreDefending(gameState)
          self.target = random.choice(food)
      elif self.target == None:
          self.target = self.selectPatrolTarget()
      actions = gameState.getLegalActions(self.index)
      feasible = []
      fvalues = []
      for a in actions:
          new_state = gameState.generateSuccessor(self.index, a)
          if not a == Directions.STOP :
              newPosition = new_state.getAgentPosition(self.index)
              feasible.append(a)
              fvalues.append(self.agent.getMazeDistance(newPosition, self.target))
      print(actions)
      best = min(fvalues)
      ties = list(filter(lambda x: x[0] == best, zip(fvalues, feasible)))
      #print('returning my best def action',bestAction, 'for - ', self.index)
      return random.choice(ties)[1]


class getAttackActions(ActionNew):
    def __init__(self, agent, index, gameState):
        self.agent = agent
        self.index = index
        self.agent.distancer.getMazeDistances()
        self.retreat = False
        self.numEnemyFood = "+inf"


        if self.agent.red:
            boundary = (gameState.data.layout.width - 2) // 2
        else:
            boundary = ((gameState.data.layout.width - 2) // 2) + 1
        self.boundary = []
        for i in range(1, gameState.data.layout.height - 1):
            if not gameState.hasWall(boundary, i):
                self.boundary.append((boundary, i))


        self.patrolSpot = []
        while len(self.patrolSpot) > (gameState.data.layout.height - 2) // 2:
            self.patrolSpot.pop(0)
            self.patrolSpot.pop(len(self.patrolSpot) - 1)

        if(self.agent.red): self.midGround=int((gameState.data.layout.width-2)/2)
        else: self.midGround=int((gameState.data.layout.width-2)/2+1)


    def chooseAction(self, gameState):
      """
      Picks among the actions with the highest Q(s,a).
      """
      middle = 0
      food = self.agent.getFood(gameState).asList()
      if(self.agent.red): middle=int((gameState.data.layout.width-2)/2)
      else: middle=int((gameState.data.layout.width-2)/2+1)
      print('printing food length for the index', self.index,'= ',len(self.agent.getFood(gameState).asList()))
      if ((len(self.agent.getFood(gameState).asList())) <= 2 or (gameState.data.timeleft <= 500 and len(food)<6)  )    :
        bounds = []
        for i in range(1, gameState.data.layout.height - 1):
            if not gameState.hasWall(middle, i):
                bounds.append((middle, i))

        actions = gameState.getLegalActions(self.index)
        actions.remove(Directions.STOP)
        mingoHomePath = 9999
        for action in actions:
          position = gameState.getAgentPosition(self.index)
          nextGameState = gameState.generateSuccessor(self.index, action)
          nextPosition = nextGameState.getAgentPosition(self.index)
          food = self.agent.getFood(gameState)
          capsules = self.agent.getCapsules(gameState)

          walls = gameState.getWalls()
          wallsList = walls.asList()
          mazeSize = walls.width * walls.height

          enemyIndices = self.agent.getOpponents(gameState)

          attackablePacmen = [gameState.getAgentPosition(i) for i in enemyIndices if self.isPacman(gameState, i) and self.isGhost(gameState, self.index) and not self.isScared(gameState, self.index)]
          scaredGhostLocations = [gameState.getAgentPosition(i) for i in self.agent.getOpponents(gameState) if self.isScared(gameState, i) and self.isGhost(gameState, i)]
          goalPositions = set(food.asList() + list(bounds) )

          enemyGhostLocations = [gameState.getAgentPosition(i) for i in enemyIndices if (self.isGhost(gameState, i) and not self.isScared(gameState, i) and (self.agent.getMazeDistance(position, gameState.getAgentPosition(i)) < 6))]
          avoidPositions = set(enemyGhostLocations)

          if(gameState.data.timeleft <= 500 ):
            goalPositions = set(food.asList()+list(bounds))
            goHomePath = len(self.aStarSearch(nextPosition, nextGameState, goalPositions, avoidPositions=avoidPositions))
          if(gameState.data.timeleft <= 500 and len(food.asList())>10):
            goalPositions = set(food.asList() )
            goHomePath = len(self.aStarSearch(nextPosition, nextGameState, goalPositions, avoidPositions=avoidPositions))
          if(gameState.data.timeleft <= 200):
            goHomePath = len(self.aStarSearch(nextPosition, nextGameState, list(bounds), avoidPositions=avoidPositions))
          else:
              goHomePath = len(self.aStarSearch(nextPosition, nextGameState, list(bounds), avoidPositions=avoidPositions))
          
          
          if goHomePath < mingoHomePath:
            mingoHomePath = goHomePath
            bestAction = action
          # successor = self.getSuccessor(gameState, action)
          # pos2 = successor.getAgentPosition(self.index)
          # for i in range(len(bounds)):
          #   disBoundary = self.agent.getMazeDistance(pos2, self.start)
          #   if disBoundary < boundaryMin:
          #     boundaryMin = disBoundary
          # #bestAction = action 
          # #dist = self.agent.getMazeDistance(self.start,pos2)
        print('returning my best def action',bestAction, 'for -', self.index) 
        # bestDist = 9999
        # for action in actions:
        #   successor = self.getSuccessor(gameState, action)
        #   pos2 = successor.getAgentPosition(self.index)
        #   for i in range(len(bounds)):
        #     disBoundary = self.agent.getMazeDistance(pos2, bounds[i])
        #     if disBoundary < boundaryMin:
        #       boundaryMin = disBoundary
        #   #bestAction = action 
        #   #dist = self.agent.getMazeDistance(self.start,pos2)
        #   if boundaryMin < bestDist:
        #     bestAction = action
        #     #bestDist = dist
        return bestAction




      actions = gameState.getLegalActions(self.index)
      actions.remove(Directions.STOP)
      # You can profile your evaluation time by uncommenting these lines
      # start = time.time()
      values=[]
      for a in actions:
          ##print(a)
          value=self.evaluate(gameState,a)
          # value = self.monteCarlo(gameState, a, 0.3, 1, 5)
          values.append(value)
      # values = [self.evaluate(gameState, a) for a in actions]
      # ##print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)
      
      maxValue = max(values)
      bestActions = [a for a, v in zip(actions, values) if v == maxValue]
      # ##print(bestActions)
      foodLeft = len(self.agent.getFood(gameState).asList())

      eatenFoodatMoment = gameState.getAgentState(self.index).numCarrying

      bounds = []
      for i in range(1, gameState.data.layout.height - 1):
          if not gameState.hasWall(self.midGround, i):
              bounds.append((self.midGround, i))

      boundaryMin = 99999


      
      print('returning my best def action',bestActions, 'for -', self.index)
      return random.choice(bestActions)

    def getFeatures(self, gameState, action):
        features = util.Counter()
        position = gameState.getAgentPosition(self.index)
        nextGameState = gameState.generateSuccessor(self.index, action)
        nextPosition = nextGameState.getAgentPosition(self.index)
        food = self.agent.getFood(gameState)
        capsules = self.agent.getCapsules(gameState)

        middle = 0
        if(self.agent.red): middle=int((gameState.data.layout.width-2)/2)
        else: middle=int((gameState.data.layout.width-2)/2+1)
    
        bounds = []
        for i in range(1, gameState.data.layout.height - 1):
            if not gameState.hasWall(middle, i):
                bounds.append((middle, i))


        walls = gameState.getWalls()
        wallsList = walls.asList()
        mazeSize = walls.width * walls.height

        enemyIndices = self.agent.getOpponents(gameState)

        attackablePacmen = [gameState.getAgentPosition(i) for i in enemyIndices if self.isPacman(gameState, i) and self.isGhost(gameState, self.index) and not self.isScared(gameState, self.index)]
        scaredGhostLocations = [gameState.getAgentPosition(i) for i in self.agent.getOpponents(gameState) if self.isScared(gameState, i) and self.isGhost(gameState, i)]
        if(self.agent.red and self.index==0 and gameState.data.timeleft>=920):
            print('bound add ho chuka')
            print(list([bounds[0]]))
            foodtoeat=food.asList()
            fte=foodtoeat[1::2]
            goalPositions=set(list([bounds[0]])+fte + attackablePacmen + capsules + scaredGhostLocations)
        elif(self.agent.red and self.index==2 and gameState.data.timeleft>=920):
            print('bound add ho chuka dobaara')
            goalPositions=set(list([bounds[-1]])+food.asList() + attackablePacmen + capsules + scaredGhostLocations)
        elif( (not self.agent.red) and self.index==1 and gameState.data.timeleft>=920):
            print('bound add ho chuka dobaara')
            foodtoeat=food.asList()
            fte=foodtoeat[1::2]
            goalPositions=set(list([bounds[0]])+fte + attackablePacmen + capsules + scaredGhostLocations)
        elif((not self.agent.red) and self.index==3 and gameState.data.timeleft>=920):
            print('bound add ho chuka dobaara')
            goalPositions=set(list([bounds[-1]])+food.asList() + attackablePacmen + capsules + scaredGhostLocations)
        else:
            print('ye to chala hi',self.index)
            goalPositions = set(food.asList() + attackablePacmen + capsules + scaredGhostLocations)

        enemyGhostLocations = [gameState.getAgentPosition(i) for i in enemyIndices if (self.isGhost(gameState, i) and not self.isScared(gameState, i) and (self.agent.getMazeDistance(position, gameState.getAgentPosition(i)) < 6))]
        # enemyGhostLocations.extend(self.validSurroundingPositionsTo(gameState, enemyGhostLocations, wallsList))
        # enemyGhostLocations.extend(self.validSurroundingPositionsTo(gameState, enemyGhostLocations, wallsList))

        features['successorScore']=( food.count())

        if action == Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]:
          features['reverse'] = -1
        else:
          features['reverse'] = 0
        # features.append( self.getFoodYouAreDefending(gameState).count() / self.initial_defending_food)
        closestGhost = len(self.aStarSearch(nextPosition, nextGameState, list(enemyGhostLocations))) if enemyGhostLocations else 0
        if(closestGhost>1 and closestGhost<=5):
          
          features['ghostDistance']=(1/closestGhost)
        if(closestGhost>6):
          features['ghostDistance']=1
        else:
          features['ghostDistance']=0

        avoidPositions = set(enemyGhostLocations)

        aStar_food_path = self.aStarSearch(nextPosition, nextGameState, list(goalPositions), avoidPositions=avoidPositions)

        features['distanceToFood']=((len(aStar_food_path) if aStar_food_path is not None else mazeSize))

        bounds = []
        for i in range(1, gameState.data.layout.height - 1):
            if not gameState.hasWall(self.midGround, i):
                bounds.append((self.midGround, i))

        goHomePath=self.aStarSearch(nextPosition, nextGameState, list(bounds), avoidPositions=avoidPositions)

        features['reachBound'] =len (goHomePath)

        # if action == Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]:
        #     features['reverse'] = -1
        # else:
        #     features['reverse'] = 0

        ##print(features)

        return features

    def getWeights(self, gameState, action):
        opponents = [gameState.getAgentState(i) for i in self.agent.getOpponents(gameState)]
        visible = list(filter(lambda x: not x.isPacman and x.getPosition() != None, opponents))
        
        eatenFoodatMoment = gameState.getAgentState(self.index).numCarrying
        isScared = False
        someOneVisible=False
        for agent in visible:
          print(gameState.getAgentPosition(self.index),agent.getPosition())
          if(self.agent.getMazeDistance(gameState.getAgentPosition(self.index),agent.getPosition())<4):
            someOneVisible=True
            if agent.scaredTimer > 10:
                isScared = True

        if((eatenFoodatMoment>=5) and not someOneVisible and not isScared):
          print('food zyaada khaaya, koi nahi pass and koi nahi dara ')
          return{'distanceToFood':-3,'ghostDistance':-100,'reachBound':-1,'reverse':0}

        if((eatenFoodatMoment>=5) and not isScared):
          print('food zyaada khaaya, koi nahi dara par koi hai ')
          return {'reachBound':-10,'ghostDistance':-100}

        if(isScared):
          print('parvaah nahi')
          return{'distanceToFood':-3,'ghostDistance':0,'reachBound':-0.5,'reverse':-0.3}


        if(someOneVisible):
          if(not gameState.getAgentState(self.index).isPacman):
            print('bhoot se bhaago par ghar ho')
            return {'ghostDistance':-100,'reachBound':-10,'reverse':5000}
          else:
            print('bhoot se bhaago')
            return {'ghostDistance':-100,'reachBound':-10,}
        print('run casually')
        return{'distanceToFood':-3,'ghostDistance':-100,'reachBound':-0.5,'reverse':0}
    

    def isGhost(self, gameState, index):
      position = gameState.getAgentPosition(index)
      if position is None:
          return False
      return not (gameState.isOnRedTeam(index) ^ (position[0] < gameState.getWalls().width / 2))

    def isScared(self, gameState, index):
      isScared = bool(gameState.data.agentStates[index].scaredTimer)
      return isScared


    def isPacman(self, gameState, index):
      position = gameState.getAgentPosition(index)
      if position is None:
          return False
      return not (gameState.isOnRedTeam(index) ^ (position[0] >= gameState.getWalls().width / 2))

    def aStarSearch(self, startPosition, gameState, goalPositions, avoidPositions=[], returnPosition=False):
      walls = gameState.getWalls()
      width = walls.width
      height = walls.height
      walls = walls.asList()

      actions = [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]
      actionVectors = [Actions.directionToVector(action) for action in actions]
      actionVectors = [tuple(int(number) for number in vector) for vector in actionVectors]

      currentPosition, currentPath, currentTotal = startPosition, [], 0
      queue = util.PriorityQueueWithFunction(lambda entry: entry[2]+width * height if entry[0] in avoidPositions else 0+min(util.manhattanDistance(entry[0], endPosition) for endPosition in goalPositions))

      # Keeps track of visited positions
      visited = set([currentPosition])

      while currentPosition not in goalPositions:

          possiblePositions = [((currentPosition[0] + vector[0], currentPosition[1] + vector[1]), action) for vector, action in zip(actionVectors, actions)]
          legalPositions = [(position, action) for position, action in possiblePositions if position not in walls]

          for position, action in legalPositions:
              if position not in visited:
                  visited.add(position)
                  queue.push((position, currentPath + [action], currentTotal + 1))
          if len(queue.heap) == 0:
              return None
          else:
              currentPosition, currentPath, currentTotal = queue.pop()

      if returnPosition:
          return currentPath, currentPosition
      else:
          return currentPath





      def allSimulation(self, depth, gameState, decay):
          new_state = gameState.deepCopy()
          if depth == 0:
              result_list = []
              actions = new_state.getLegalActions(self.index)
              actions.remove(Directions.STOP)

              reversed_direction = Directions.REVERSE[new_state.getAgentState(self.index).configuration.direction]
              if reversed_direction in actions and len(actions) > 1:
                  actions.remove(reversed_direction)
              a = random.choice(actions)
              next_state = new_state.generateSuccessor(self.index, a)
              result_list.append(self.evaluate(next_state, Directions.STOP))
              return max(result_list)

          result_list = []
          actions = new_state.getLegalActions(self.index)
          current_direction = new_state.getAgentState(self.index).configuration.direction

          reversed_direction = Directions.REVERSE[current_direction]
          if reversed_direction in actions and len(actions) > 1:
              actions.remove(reversed_direction)
          for a in actions:
              next_state = new_state.generateSuccessor(self.index, a)
              result_list.append(
                  self.evaluate(next_state, Directions.STOP) + decay * self.allSimulation(depth - 1, next_state, decay))
          return max(result_list)
      # def MTCS(self, depth, gameState, decay):
      #     new_state = gameState.deepCopy()
      #     value = self.evaluate(new_state, Directions.STOP)
      #     decay_index = 1
      #     while depth > 0:

      #         actions = new_state.getLegalActions(self.index)
      #         current_direction = new_state.getAgentState(self.agent.index).configuration.direction
      #         reversed_direction = Directions.REVERSE[new_state.getAgentState(self.agent.index).configuration.direction]
      #         if reversed_direction in actions and len(actions) > 1:
      #             actions.remove(reversed_direction)
      #         # Randomly chooses a valid action
      #         a = random.choice(actions)
      #         # Compute new state and update depth
      #         new_state = new_state.generateSuccessor(self.agent.index, a)
      #         value = value + decay ** decay_index * self.evaluate(new_state, Directions.STOP)
      #         depth -= 1
      #         decay_index += 1
      #     return value
      # def chooseAction(self, gameState):
          start = time.time()
          actions = gameState.getLegalActions(self.agent.index)
          actions.remove(Directions.STOP)
          feasible = []
          for a in actions:
              value = 0
              value = self.allSimulation(2, gameState.generateSuccessor(self.agent.index, a), 0.7)
              feasible .append(value)
          bestAction = max(feasible)
          possibleChoice = list(filter(lambda x: x[0] == bestAction, zip(feasible, actions)))
          return random.choice(possibleChoice)[1]
class AttackAgent(CaptureAgent):
    def __init__(self, index):
        CaptureAgent.__init__(self, index)

    def registerInitialState(self, gameState):
        CaptureAgent.registerInitialState(self, gameState)
        self.DefenceStatus = getDefensiveActions(self, self.index, gameState)
        self.OffenceStatus = getAttackActions(self, self.index, gameState)

    def chooseAction(self, gameState):
        self.enemies = self.getOpponents(gameState)
        invaders = [a for a in self.enemies if gameState.getAgentState(a).isPacman]
        #if gameState.data.timeleft > 600:
        timeLeft = gameState.data.timeleft
        print(timeLeft, self.index)
        
        if timeLeft >= 500:
          return self.OffenceStatus.chooseAction(gameState)

        else:
          if timeLeft <= 400 and not gameState.getAgentState(self.index).isPacman:
            return self.DefenceStatus.chooseAction(gameState)
          if self.getScore(gameState) > 15:
            return self.DefenceStatus.chooseAction(gameState)
          else:
            return self.OffenceStatus.chooseAction(gameState)
