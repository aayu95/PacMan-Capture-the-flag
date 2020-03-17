# myTeam.py
# ----------

from captureAgents import CaptureAgent
import random, time, util
from game import Directions, Actions
import game
from util import nearestPoint


#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed, first='OffenseAgent', second='DefenseAgent'):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.
    """
    return [eval(first)(firstIndex), eval(second)(secondIndex)]


##########
# Agents #
##########

class OffenseAgent(CaptureAgent):
    """
    The offensive agent uses q-learning to learn an optimal offensive policy
    over hundreds of games/training sessions. The policy changes this agent's
    focus to offensive features such as collecting pellets/capsules, avoiding ghosts,
    maximizing scores via eating pellets etc.
    """

    def registerInitialState(self, gameState):
        CaptureAgent.registerInitialState(self, gameState)
        self.start = gameState.getAgentPosition(self.index)
        self.epsilon = 0.0  # exploration prob
        self.alpha = 0.2  # learning rate
        self.discountRate = 0.8
        self.weights = {'closest-food': -2.2558226236802597,
                        'successorScore': -0.027287497346388308,
                        'eats-food': 9.970429654829946,
                        '#-of-ghosts-1-step-away': -0.18419418670562,
                        'bias': 1.0856704846852672,
                        'go-home': 12.970429654829946}

        """
        try:
            with open('weights.txt', "r") as file:
                self.weights = eval(file.read())
        except IOError:
                return
        """

    """
    Iterate through all features (closest food, bias, ghost dist), multiply each of the features' value to the feature's weight,
    and return the sum of all these values to get the q-value.
    """

    def getQValue(self, gameState, action):
        features = self.getFeatures(gameState, action)
        return features * self.weights

    # Iterates through all q-values of all actions, and returns the maximum q-value

    def getValue(self, gameState):
        qVals = []
        legalActions = gameState.getLegalActions(self.index)
        if len(legalActions) == 0:
            return 0.0
        else:
            for action in legalActions:
                qVals.append(self.getQValue(gameState, action))
            return max(qVals)

    # Iterates through all q-values of all possible actions, and returns the action with the maximum q-value

    def getPolicy(self, gameState):
        values = []
        legalActions = gameState.getLegalActions(self.index)
        legalActions.remove(Directions.STOP)
        if len(legalActions) == 0:
            return None
        else:
            for action in legalActions:
                # self.updateWeights(gameState, action)
                values.append((self.getQValue(gameState, action), action))
        return max(values)[1]

    """
    If probability is < 0.1, then choose a random action from a list of legal actions.
    Otherwise use the policy defined above to get an action.
    """

    def chooseAction(self, gameState):
        # Pick Action
        legalActions = gameState.getLegalActions(self.index)
        action = random.choice(legalActions)

        foodLeft = len(self.getFood(gameState).asList())

        if len(legalActions) != 0 and foodLeft > 2:
            prob = util.flipCoin(self.epsilon)
            if not prob:
                action = self.getPolicy(gameState)

        return action

    def getFeatures(self, gameState, action):
        food = self.getFood(gameState)
        walls = gameState.getWalls()
        ghosts = []
        opponents = CaptureAgent.getOpponents(self, gameState)
        # Gets visible ghost locations and states
        if opponents:
            for opponent in opponents:
                opp_pos = gameState.getAgentPosition(opponent)
                op_is_pacman = gameState.getAgentState(opponent).isPacman
                if opp_pos and not op_is_pacman:
                    ghosts.append(opp_pos)

        # Initializes features
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)

        features['successorScore'] = self.getScore(successor)

        features["bias"] = 1.0

        # Computes the location of pacman after it takes the action
        x, y = gameState.getAgentPosition(self.index)
        dx, dy = Actions.directionToVector(action)
        next_x, next_y = int(x + dx), int(y + dy)

        # Number of ghosts 1-step away
        features["#-of-ghosts-1-step-away"] = sum(
            (next_x, next_y) in Actions.getLegalNeighbors(ghost, walls) for ghost in ghosts)

        myPos = gameState.getAgentState(self.index).getPosition()

        # Check for ghosts & add the food feature
        if not features["#-of-ghosts-1-step-away"] and food[next_x][next_y]:
            features["eats-food"] = 1.0

        # Finds closest food
        dist = self.closestFood((next_x, next_y), food, walls)
        if dist is not None:
            features["closest-food"] = float(dist) / (walls.width * walls.height)

        if gameState.data.agentStates[self.index].numCarrying > 5:
            print("Enough food! Going home...")
            distHome = self.getMazeDistance(myPos, self.start)
            features["eats-food"] = 0.0
            features["go-home"] = distHome

        # Normalizes all the features and returns
        features.divideAll(10.0)
        print(features)
        return features

    """
    Iterate through all features and for each feature, update its weight values using the Q-learning formula:
    w(i) = w(i) + alpha((reward + discount*value(nextState)) - Q(s,a)) * f(i)(s,a)
    """

    def updateWeights(self, gameState, action):
        features = self.getFeatures(gameState, action)
        nextState = self.getSuccessor(gameState, action)

        reward = nextState.getScore() - gameState.getScore()

        for feature in features:
            correction = (reward + self.discountRate * self.getValue(nextState)) - self.getQValue(gameState, action)
            self.weights[feature] = self.weights[feature] + self.alpha * correction * features[feature]

    def getSuccessor(self, gameState, action):
        successor = gameState.generateSuccessor(self.index, action)
        pos = successor.getAgentState(self.index).getPosition()
        if pos != nearestPoint(pos):
            return successor.generateSuccessor(self.index, action)
        else:
            return successor

    def closestFood(self, pos, food, walls):
        fringe = [(pos[0], pos[1], 0)]
        expanded = set()
        while fringe:
            pos_x, pos_y, dist = fringe.pop(0)
            if (pos_x, pos_y) in expanded:
                continue
            expanded.add((pos_x, pos_y))

            # Checks for food
            if food[pos_x][pos_y]:
                return dist

            nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
            for nbr_x, nbr_y in nbrs:
                fringe.append((nbr_x, nbr_y, dist + 1))

        # Finds no food
        return None


class DefenseAgent(CaptureAgent):
    """
    A simple reflex agent that takes score-maximizing actions. It's given
    features and weights that allow it to prioritize defensive actions over any other.
    """

    def registerInitialState(self, gameState):
        """
        This method handles the initial setup of the
        agent to populate useful fields (such as what team
        we're on).
        """

        CaptureAgent.registerInitialState(self, gameState)
        self.myAgents = CaptureAgent.getTeam(self, gameState)
        self.opAgents = CaptureAgent.getOpponents(self, gameState)
        self.myFoods = CaptureAgent.getFood(self, gameState).asList()
        self.opFoods = CaptureAgent.getFoodYouAreDefending(self, gameState).asList()

    # Finds the next grid position (location tuple).
    def getSuccessor(self, gameState, action):
        successor = gameState.generateSuccessor(self.index, action)
        pos = successor.getAgentState(self.index).getPosition()
        if pos != nearestPoint(pos):
            return successor.generateSuccessor(self.index, action)
        else:
            return successor

    # Returns a counter of features for the state
    def getFeatures(self, gameState, action):
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)

        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()

        # Checks whether we're on defense (1) or offense (0)
        features['onDefense'] = 1
        if myState.isPacman:
            features['onDefense'] = 0

        # Computes distance to visible invaders
        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        invaders = [a for a in enemies if a.isPacman and a.getPosition() is not None]
        features['numInvaders'] = len(invaders)
        if len(invaders) > 0:
            dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
            features['invaderDistance'] = min(dists)

        if action == Directions.STOP:
            features['stop'] = 1
        rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
        if action == rev:
            features['reverse'] = 1

        return features

    # Returns a dictionary of features for the state
    def getWeights(self, gameState, action):
        return {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -10, 'stop': -100, 'reverse': -2}

    # Computes a linear combination of features and feature weights
    def evaluate(self, gameState, action):
        features = self.getFeatures(gameState, action)
        weights = self.getWeights(gameState, action)
        return features * weights

    # Chooses the best action
    def chooseAction(self, gameState):
        agentPos = gameState.getAgentPosition(self.index)
        actions = gameState.getLegalActions(self.index)

        # Distances between agent & foods
        distToFood = []
        for food in self.myFoods:
            distToFood.append(self.distancer.getDistance(agentPos, food))

        # Distances between agent & opponents
        distToOps = []
        for opponent in self.opAgents:
            opPos = gameState.getAgentPosition(opponent)
            if opPos is not None:
                distToOps.append(self.distancer.getDistance(agentPos, opPos))

        # Get the best action
        values = [self.evaluate(gameState, a) for a in actions]
        maxValue = max(values)
        bestActions = [a for a, v in zip(actions, values) if v == maxValue]
        return random.choice(bestActions)

# Update weights file after each game
# def final(self, gameState):
    # print self.weights
    # file = open('weights.txt', 'w')
    # file.write(str(self.weights))
