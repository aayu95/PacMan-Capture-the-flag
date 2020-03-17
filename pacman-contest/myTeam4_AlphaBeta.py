# myTeam.py
# ---------
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


from captureAgents import CaptureAgent
import random, time, util
from game import Directions
import game
import copy
from game import Actions
#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
			   first = 'OffensiveAgent', second = 'DefensiveAgent'):
	return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########

class AlphaBetaPrunningAgent(CaptureAgent):

	# Register initial state for AlphaBetaAgent include start position of the
	# agent, closest food position to the center and timer for defensive agent,
	# and mid point for center position.
	def getDeepWalls(self):
		return copy.deepcopy(self.walls)
	def registerInitialState(self, gameState):
		CaptureAgent.registerInitialState(self, gameState)
		self.start = gameState.getAgentPosition(self.index)
		self.closestFoodToCenter = None
		self.timer = 0
		self.walls = gameState.data.layout.walls

		# Setting up center position and check if the center position has a wall.
		mHeight = gameState.data.layout.height//2
		mWidth = gameState.data.layout.width//2
		while(gameState.hasWall(mWidth, mHeight)):
			mHeight -= 1
			if mHeight == 0:
				mHeight = gameState.data.layout.height
		#Updating midpoint as the tuple of above calculated mWidth and mHeight
		self.midPoint = (mWidth, mHeight)
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
	def aStarSearch(self,gameState,start,searchPos, foodOrGhost):
	    import util
	    opponentList = CaptureAgent.getOpponents(self,gameState)
	    #getWalls1 = self.walls
	    getWalls = self.getDeepWalls()

	    if foodOrGhost == 1:#1 = Food or capsule else ghost
	      for i in opponentList:          
	        checkNone = gameState.getAgentPosition(i)
	        if checkNone is not None:
	          x,y = checkNone
	          getWalls[x][y] = True
	    adjacentNodes = util.PriorityQueue()
	    visitedNodes = []
	    adjacentNodes.push((start,[start],0),(0+self.getMazeDistance(start, searchPos)))
	    #Pop the initial point from the priority queue
	    (state,goalDirection,cost) = adjacentNodes.pop()
	    #adding the point to the visited list
	    visitedNodes.append((state, cost+self.getMazeDistance(start, searchPos)))
	    while state != searchPos: #while we do not find the goal point
	        neighbours = Actions.getLegalNeighbors(state,self.walls) #get the point's succesors
	        for node in neighbours:
	            visitedExist = False
	            totalCost = cost + 1
	            for (currentState,costToCurrentState) in visitedNodes:
	                if (node == currentState) and (costToCurrentState <= totalCost): 
	                    visitedExist = True
	                    break
	            if not visitedExist:        
	                adjacentNodes.push((node,goalDirection + [node],cost + 1),(cost + 1 + self.getMazeDistance(node, searchPos))) 
	                visitedNodes.append((node,cost + 1))
	        (state,goalDirection,cost) = adjacentNodes.pop()
	    return len(goalDirection)
	def alphabeta(self, gameState, action, agent, mdepth, alpha, beta, visibleAgents):
		# Restart if agent crosses the range
		if agent >= gameState.getNumAgents():
			agent = 0
		# With each iteration increase the depth for further exploration 
		if agent == self.index:
			mdepth += 1
		# Ignore if the agent is not visible
		if agent not in visibleAgents:
			return self.alphabeta(gameState, action, agent + 1, mdepth, alpha, beta, visibleAgents)
		# Evaluate the current state if the game is over or depth is 1
		if mdepth == 1 or gameState.isOver():
			if agent == self.index:
				return self.evaluate(gameState, action)
			else:
				self.alphabeta(gameState, action, agent + 1, mdepth, alpha, beta, visibleAgents)
		#generate legal actions for the current gameState
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

class OffensiveAgent(AlphaBetaPrunningAgent):
	# The basic of offensive agent is to grab all the food while avoiding ghost.
	def chooseAction(self, gameState):
		start = time.time()

		# Get all visible agents
		allAgents = range(0, gameState.getNumAgents())
		visibleAgents = [a for a in allAgents if gameState.getAgentState(a).getPosition() is not None]

		# Start alpha beta pruning algorithm
		v = (float("-inf"), 'None')
		alpha = float('-inf')
		beta = float('inf')
		legalActions = gameState.getLegalActions(self.index)
		for action in legalActions:
			if action == 'Stop':
				continue
			v = max(v, (self.alphabeta(gameState.generateSuccessor(self.index, action), action, self.index+1, 0, alpha, beta, visibleAgents), action))
			if v[0] > beta:
                #print ("Agent {0} chose {1} with value {2}".format(self.index, v[1], v[0])
                #print 'Execution time for agent %d: %.4f' % (self.index, time.time() - start)
				return v[1]
			alpha = max(alpha, v[0])
        #print "Agent {0} chose {1} with value {2}".format(self.index, v[1], v[0])
        #print 'Execution time for agent %d: %.4f' % (self.index, time.time() - start)
		return v[1]

	def getFeatures(self, gameState, action):
		features = util.Counter()
		foodList = self.getFood(gameState).asList()
		# Check the team the current agent is in and on that basis all the features will be calculated
		# Depending on the team(Red or Blue) update the value of successorScore
		if self.red:
			features['successorScore'] = gameState.getScore()
		else:
			features['successorScore'] = -1* gameState.getScore()
		features['successorScore'] -= len(foodList)
		features['distanceToGhost'] = 0

		# Extract all visible agents
		totalAgents = range(0, gameState.getNumAgents())
		visibleAgents = [a for a in totalAgents if gameState.getAgentState(a).getPosition() != None]

		cPos = gameState.getAgentState(self.index).getPosition()

		# Check if opponent is visible.
		if not set(visibleAgents).isdisjoint(self.getOpponents(gameState)):
			# Agent will need to distance themself from ghost if agent is a pacman or agent is scared.
			if gameState.getAgentState(self.index).isPacman and gameState.getAgentState(self.index).scaredTimer > 0:
				ghosts = list(set(visibleAgents).intersection(self.getOpponents(gameState)))
				for ghost in ghosts:
					ghostPos = gameState.getAgentState(ghost).getPosition()
					dist = self.aStarSearch(gameState,cPos, ghostPos,0)
					# Agent will never move to less than 2 distance.
					if dist <= 3:
						features['distanceToGhost'] = -9999
					else:
						features['distanceToGhost'] += 0.5 * dist
			# Ignore or hide from the ghost
			else:
				ghosts = list(set(visibleAgents).intersection(self.getOpponents(gameState)))
				for ghost in ghosts:
					ghostPos = gameState.getAgentState(ghost).getPosition()
					dist = self.aStarSearch(gameState,cPos, ghostPos,0)
					features['distanceToGhost'] += 0.5 * dist
		else:
			ghosts = list(set(totalAgents).difference(self.agentsOnTeam))
			for ghost in ghosts:
				ghostDists = gameState.getAgentDistances()
				#Updating distanceToGhost feature
				features['distanceToGhost'] += ghostDists[ghost]
		# Agent will grab the nearest if it isn't carrying three foods already.
		if gameState.getAgentState(self.index).numCarrying < 4:
			myPos = gameState.getAgentState(self.index).getPosition()
			if len(foodList) > 0:
				minDis = min([self.getMazeDistance(myPos, food) for food in foodList])
				features['distanceToFood'] = minDis
			else:
				myPos = gameState.getAgentState(self.index).getPosition()
				features['distanceToFood'] = self.getMazeDistance(myPos, self.start)
		else:
			myPos = gameState.getAgentState(self.index).getPosition()
			features['distanceToFood'] = self.getMazeDistance(myPos, self.start)
		return features
		# Reverse action should be prevented
		if action == Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]:
			features['reverse'] = 10
		capsuleList = set(gameState.data.capsules) - set(self.getCapsulesYouAreDefending(gameState))
		if len(capsuleList) > 0:
			minCapsuleDistance = 99999
			for c in capsuleList:
				distance = self.aStarSearch(gameState,pos, c,1)
				if distance < minCapsuleDistance:
					minCapsuleDistance = distance
			features['distanceToCapsule'] = minCapsuleDistance
		else:
			features['distanceToCapsule'] = 0

	def getWeights(self, gameState, action):
		return {'successorScore': 100, 'distanceToFood': -1, 'distanceToGhost': 1, 'reverse': -0.1, 'distanceToCapsule': -1}

class DefensiveAgent(AlphaBetaPrunningAgent):
	# The basic of defensive agent is to go around the nearest food to the center to detect an enemy
	# and chase one if found.
	def chooseAction(self, gameState):
		start = time.time()

		# Get all visible agents
		totalAgents = range(0, gameState.getNumAgents())
		visibleAgents = [a for a in totalAgents if gameState.getAgentState(a).getPosition() is not None]

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
				return v[1]
			alpha = max(alpha, v[0])

		# Minus the timer for the patrol function.
		self.timer -= 1
		return v[1]

	def getFeatures(self, gameState, action):
		features = util.Counter()
		myState = gameState.getAgentState(self.index)
		myPos = myState.getPosition()
		foodList = self.getFoodYouAreDefending(gameState).asList()

		# Computes distance to invaders we can see
		adv = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
		oppon = [a for a in adv if a.isPacman and a.getPosition() != None]
		features['numInvaders'] = len(oppon)

		# Check if any opponent is found.
		if len(invaders) > 0:
			distI = [self.aStarSearch(gameState,myPos, a.getPosition(),0) for a in oppon]
			features['invaderDistance'] = max(distI)
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
