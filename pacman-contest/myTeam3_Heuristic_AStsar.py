##################################################
#   Team Name - Area-51                          #
#   Name - Siddharth Agarwal                     #
#   Student ID - 1077275                         #
##################################################

from captureAgents import CaptureAgent
import distanceCalculator
import random, time, util, sys
from game import Directions
import game
from util import nearestPoint
from game import Actions
from layout import Layout as ll
import copy

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'OffensiveReflexAgent', second = 'DefensiveReflexAgent'):
  
  return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
  
 
  def registerInitialState(self, gameState):
    self.start = gameState.getAgentPosition(self.index)
    CaptureAgent.registerInitialState(self, gameState)
    self.walls = gameState.data.layout.walls

  def getDeepWalls(self):# get the deep copy of the walls for checking ghost or food
    return copy.deepcopy(self.walls)

  def chooseAction(self, gameState):
    """
    Picks among the actions with the highest Q(s,a).
    """
    actions = gameState.getLegalActions(self.index)
    actions = actions[:-1] #removing the stop action
    values = [self.evaluate(gameState, a) for a in actions] 
    maxValue = max(values) #taking maximum action value 
    bestActions = [a for a, v in zip(actions, values) if v == maxValue]
    foodLeft = len(self.getFood(gameState).asList())

    if foodLeft <= 2: # if food is less than f2 go back to end game
      bestDist = 9999
      for action in actions:
        successor = self.getSuccessor(gameState, action)
        pos2 = successor.getAgentPosition(self.index)
        dist = self.aStarSearch(gameState,self.start,pos2,1)
        if dist < bestDist:
          bestAction = action
          bestDist = dist
      return bestAction

    return random.choice(bestActions)

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
    features['successorScore'] = self.getScore(successor)
    return features

  def getWeights(self, gameState, action):
    return {'successorScore': 1.0}

class OffensiveReflexAgent(ReflexCaptureAgent):
  
  def evaluate(self, gameState, action):   
    features = self.getFeatures(gameState, action)
    weights = self.getWeights(gameState, action)
    return features*weights


  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    opponentList = CaptureAgent.getOpponents(self,gameState)# get opponents indices
    foodList = self.getFood(successor).asList() 

    if(gameState.getAgentState(self.index).numCarrying < 4):   # eat 4 food and submit to score
      features['successorScore'] = -len(foodList)
    else: # keep eating food 
      myPos = successor.getAgentState(self.index).getPosition()
      defendFood = self.getFoodYouAreDefending(gameState).asList()
      features['successorScore'] = -min([self.getMazeDistance(myPos, self.start)])

    #calculating distance from the visible opponents
    gDist = 0
    minGhost = 9999
    for i in opponentList:          
      checkNone = gameState.getAgentPosition(i)
      if checkNone is not None: 
        myPos = successor.getAgentState(self.index).getPosition()
        gDist = self.aStarSearch(gameState, myPos, checkNone,0)
        if gDist < 3:
          if gDist < minGhost:
            minGhost = gDist

    if (minGhost == 9999): minGhost = 0
    
    features['distanceToGhost'] = minGhost

    # get the capsule a-star distance
    capsule = self.getCapsules(gameState)
    if len(capsule) != 0:
      myPos = successor.getAgentState(self.index).getPosition()
      features['distanceToCapsule'] = min([self.aStarSearch(gameState, myPos, caps, 1) for caps in capsule])
      
    else:
      features['distanceToCapsule'] = 0

    # get food distance 
    myPos = successor.getAgentState(self.index).getPosition()
    minDistance = min([self.aStarSearch(gameState,myPos,food,1) for food in foodList])
    features['distanceToFood'] = minDistance

    return features

  def getWeights(self, gameState, action):
    return {'successorScore': 100, 'distanceToFood': -5, 'distanceToCapsule':-10, 'distanceToGhost':25}


  def depthFirstSearch(self, gameState, start, ghost):
    from util import Stack
    _walls1 = self.walls
    _walls = _walls1
    visited = [] 
    parent = Stack()
    path = Stack()
    children = {}
    parent.push(start)
    visited.append(start)
    while not parent.isEmpty():
        x = parent.pop()
        parent.push(x)
        if x == ghost:
          break
        if x in children:
            successors = children[x]
        else:
            successors = Actions.getLegalNeighbors(x,_walls)
            children[x] = successors            
        l = 0
        change = 0
        while (change < 1) and (l < len(successors)):         
            if successors[l] not in visited:
                parent.push(successors[l])
                visited.append(successors[l])
                change += 1                           
            l += 1
        if change == 0:
            parent.pop()
            
    k = []
    while not parent.isEmpty():
        k.append(parent.pop())
  
    return len(k)


  def aStarSearch(self,gameState,start,searchPos, foodOrGhost):
    import util
    # Initilization
    opponentList = CaptureAgent.getOpponents(self,gameState)
    getWalls = self.getDeepWalls()# get deep copy of teh walls and edit for walls 
    
    if foodOrGhost == 1:#1 = Food or capsule else ghost
      for i in opponentList:          
        checkNone = gameState.getAgentPosition(i)
        if checkNone is not None:
          x,y = checkNone
          getWalls[x][y] = True
    
    adjacentNodes = util.PriorityQueue()
    visitedNodes = [] #List holding nodes already visited
    #Start by pushing the start node into the priority queue
    adjacentNodes.push((start,[start],0),(0+self.getMazeDistance(start, searchPos)))
    #Pop the initial point from the priority queue
    (state,goalDirection,cost) = adjacentNodes.pop()
    
    #adding the point to the visited list
    visitedNodes.append((state, cost+self.getMazeDistance(start, searchPos)))  

    while state != searchPos: #while we do not find the goal point
      neighbours = Actions.getLegalNeighbors(state,getWalls) #get the point's succesors
      for node in neighbours:
          visitedExist = False
          totalCost = cost + 1
          for (currentState,costToCurrentState) in visitedNodes:
              # Check the closed list to find if there are any nodes at the same level with cost less than the total cost
              if (node == currentState) and (costToCurrentState <= totalCost): 
                  visitedExist = True
                  break

          if not visitedExist:        
              # push the point with priority num of its total cost
              adjacentNodes.push((node,goalDirection + [node],cost + 1),(cost + 1 + self.getMazeDistance(node, searchPos))) 
              visitedNodes.append((node,cost + 1)) # add this point to visited list
      
      if (adjacentNodes.isEmpty()):
        return 0
      (state,goalDirection,cost) = adjacentNodes.pop()
    return len(goalDirection)

class DefensiveReflexAgent(ReflexCaptureAgent):


  def aStarSearch(self,gameState,start,searchPos, foodOrGhost):
    import util
    # Initilization
    opponentList = CaptureAgent.getOpponents(self,gameState)
    getWalls = self.getDeepWalls()

    if foodOrGhost == 1:#1 = Food or capsule else ghost
      for i in opponentList:          
        checkNone = gameState.getAgentPosition(i)
        if checkNone is not None:
          x,y = checkNone
          getWalls[x][y] = True
     
    adjacentNodes = util.PriorityQueue()
    visitedNodes = [] #List holding nodes already visited
    #Start by pushing the start node into the priority queue
    adjacentNodes.push((start,[start],0),(0+self.getMazeDistance(start, searchPos)))
    #Pop the initial point from the priority queue
    (state,goalDirection,cost) = adjacentNodes.pop()
    #adding the point to the visited list
    visitedNodes.append((state, cost+self.getMazeDistance(start, searchPos)))  

    while state != searchPos: #while we do not find the goal point
        neighbours = Actions.getLegalNeighbors(state,getWalls) #get the point's succesors
        for node in neighbours:
            visitedExist = False
            totalCost = cost + 1
            for (currentState,costToCurrentState) in visitedNodes:
                # Check the closed list to find if there are any nodes at the same level with cost less than the total cost
                if (node == currentState) and (costToCurrentState <= totalCost): 
                    visitedExist = True
                    break

            if not visitedExist:        
                # push the point with priority num of its total cost
                adjacentNodes.push((node,goalDirection + [node],cost + 1),(cost + 1 + self.getMazeDistance(node, searchPos))) 
                visitedNodes.append((node,cost + 1)) # add this point to visited list
        if(adjacentNodes.isEmpty()):
          return 0
        (state,goalDirection,cost) = adjacentNodes.pop()
    return len(goalDirection)

  
  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    foodList = self.getFoodYouAreDefending(successor).asList()
    
    #get the capsules to defend
    if gameState.isOnRedTeam(self.index):
      capsule = gameState.getRedCapsules()   
    else:
      capsule = gameState.getBlueCapsules()

    # get the center of the grid
    myState = successor.getAgentState(self.index)
    myPos = myState.getPosition()
    midHeight = int(gameState.data.layout.height//(2))
    midWidth = int(gameState.data.layout.width//(2))
    while(gameState.hasWall(midWidth, midHeight)):
      midHeight -= 1
      if midHeight == 0:
        midHeight = gameState.data.layout.height
    center = (midWidth, midHeight)
    
    features['protectDistance'] = min([self.aStarSearch(gameState,myPos, center,1)])
    
    # Computes whether we're on defense (1) or offense (0)
    features['onDefense'] = 1
    if myState.isPacman: features['onDefense'] = 0

    # Computes a-star distance to invaders we can see
    enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
    features['numInvaders'] = len(invaders)
    if len(invaders) > 0:
      dists = [self.aStarSearch(gameState,myPos, a.getPosition(), 0) for a in invaders]
      features['invaderDistance'] = min(dists)
      
    if action == Directions.STOP: features['stop'] = 1
    rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
    if action == rev: features['reverse'] = 2

    return features

  def getWeights(self, gameState, action):
    return {'foodlist' :-100,'capsule':-150 ,'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -1000, 'stop': -100, 'reverse': -2,'protectDistance':-100}
