"""
The steps involved in the implementation of REINFORCE would be as follows:

Initialize a Random Policy (a NN that takes the state as input and returns the probability of actions)
Use the policy to play N steps of the game — record action probabilities-from policy, reward-from environment, action — sampled by agent
Calculate the discounted reward for each step by backpropagation
Calculate expected reward G
Adjust weights of Policy (back-propagate error in NN) to increase G
Repeat from 2
"""

from learningAgents import ReinforcementAgent
import util
import random
import numpy as np
from featureExtractors import *
import pickle
random.seed(1)


class Reinforce(ReinforcementAgent):
    def __init__(self, epsilon=0.05, gamma=0.8, alpha=0.2, numTraining=0, extractor='ExtendedExtractor', **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p REINFORCE -a gamma=0.9

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman

        # Calling base class
        ReinforcementAgent.__init__(self, **args)

        # logging the hyperparameters
        print("initialized Policy gradient agent")
        print(
            f"Initialized with --- \nepsilon = {self.epsilon} ---\ndiscount = {self.discount} ---\nLearning Rate = {self.alpha}")
        print(f"Number of training iterations: {self.numTraining}---")

        # Loading the learnt weights for the policy agent
        # theta = policy agent parameter
        # w = baseline parameter
        # try:
        #     with open('weights.pk', 'rb') as f:
        #         print("Weights loaded")
        #         self.theta = pickle.load(f)
        # except:
        print("Couldn't find weights saved, Reset weights")
        self.theta = util.Counter()
        self.w = util.Counter()
        self.x = util.Counter()

        # features defined as x(s,a)
        self.featExtractor = util.lookup(extractor, globals())()

        # Initializing variables for storing episodes
        # E = [(s0, a0, r1), (s1, a1, r2)...]
        self.probs = [] # storing probabilities by the state probs = [(action_prob[a] = prob(a|s0), (action_prob[b] = prob(b|s0), ...]
        self.actions = []
        self.states = []
        self.rewards = []
        self.bestProbs = []

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With multinomial
          probability or based on categorical distribution, we should take a random action and
          take the random policy categorywise.  Note that if there are
          no legal actions, which is the case at the terminal state, it will 
          choose None as the action.

          PS: This function calls getPolicy to get the probabilities and the action
        """
        # Pick Action
        legalActions = self.getLegalActions(state)
        action = None
        "*** YOUR CODE HERE ***"
        if not legalActions:
            return None
        action = self.getPolicy(state)
        # TODO: Seperate this?
        self.doAction(state, action)
        return action

    # Overriding the policy -- using function approximator for this
    def getPolicy(self, state):
        """Here we first approximate the policy using parameters theta
        the policy is the probability of choosing an action a_t at 
        state s_t with params theta. 
        Notice that we are directly approximating the 
        policy (instead of the Q values)"""
        # Get all legal actions in the state
        # Get action and probability of taking action
        # space = state x action = given by identity extractor
        # then we improve the distribution
        # Firstly capure weight of each feature
        # each feature = feature[state x action]
        # In q value, Q(s,a) = sum all s,a for feature(s,a) * weight[feature]
        # we need prob distribution -- so we create val[feature] = weight[feature] * feature(s,a)
        # take softmax of this => prob[feature]
        # using Q values as the value(s,a) -> using this for prob.
        legalActions = self.getLegalActions(state)
        forwardPassValue = []
        for action in legalActions:
            self.stateFeatures = self.featExtractor.getFeatures(state, action)
            total = 0
            for feature in self.stateFeatures:
                total += self.theta[feature] * self.stateFeatures[feature]
            forwardPassValue.append(total)
        # converting forwardPassValue to np array
        forwardPassValue_np = np.array(forwardPassValue)
        # take softmax to get probabilities (numerically stable softmax)
        # probs = (np.exp(forwardPassValue_np) /
        #          np.exp(forwardPassValue_np - np.max(forwardPassValue_np)).sum())
        probs = self.softmax(forwardPassValue_np)
        # sampling from distribution of probs
        # instead of choosing legalAction, we send list of action index, to take log prob later
        # sampledAction = util.sample(
        #     probs.tolist(), [i for i in range(len(legalActions))])
        try:
            # sampledActionIndex = np.random.choice(actionIndex, p=probs)
            action_prob = {}
            actionIndex = []
            for idx, action in enumerate(legalActions):
                actionIndex.append(idx)
                action_prob[action] = probs[idx]

            sampledActionIndex = util.sample(probs.tolist(), actionIndex)
            self.bestProbs.append(probs[sampledActionIndex])
            self.probs.append(action_prob)
        except: 
            print("Error")
            print(f"Legal Actions: {legalActions}")
            print(f"Probs: {probs}")
            print(f"Dot product: {forwardPassValue}")
        # print(legalActions[sampledAction])
        # need to return probs / log of probs for updating policy later
        # print(self.probs)
        # util.pause()
        return legalActions[sampledActionIndex]

    # def update(self, state, action, nextState, reward):
    #     """
    #        Should update your weights based on transition
    #     """
    #     "*** YOUR CODE HERE ***"
    #     print("Updating here")
    #     sample = reward + self.discount * 1.0
    #     loss = sample - self.getQValue(state, action)
    #     for feature, val in self.stateFeatures.items():
    #       self.theta[feature] = self.theta[feature] + self.alpha * loss * val

    def observeTransition(self, state, action, nextState, deltaReward):
        self.episodeRewards += deltaReward
        # store the trajectory information
        self.actions.append(action)
        self.states.append(state)
        self.rewards.append(deltaReward)
        self.x[state] = 1
        # don't update now
        # update at end of episode

    def update(self):
        ep_len = len(self.states)
        for t in range(ep_len):
            # Calculate discounted return from ith state to end of trajectory
            G_t = 0
            for k in range(0, ep_len):  # since rewards are collected till the last state
                G_t += np.power(self.discount, k-t-1) * self.rewards[k]
            # G_t -= (sum(self.rewards) / len(self.rewards))
            # getting score function
            # scoreVal = self.scoreFnGradDiv(t)
            scoreVal = self.scoreFnGradDiv(t)
            # scoreVal = self.scoreFn(t)
            # Updaing weights
            features = self.featExtractor.getFeatures(self.states[t], self.actions[t])
            # print(f"G = {G_t}, scoreFnVal = {scoreVal}")
            for feature in features:
              self.theta[feature] += self.alpha * np.power(self.discount, t) * G_t * scoreVal

    def stopEpisode(self):
        ReinforcementAgent.stopEpisode(self)
        # print("Episode ended")
        # get all s, a, r and update
        self.update()
        # print("Episode ended")
        # print(self.theta)
        if self.episodesSoFar % 100 == 0:
            # print("Saving weights")
            self.saveWeights()
            print(f"rewards = {self.rewards}, \nactions = {self.actions}, \nprobs = {self.probs}")
            # print(f"weights = {self.theta}")
            # util.pause()

        # reset
        self.states = []
        self.actions = []
        self.probs = []
        self.rewards = []
        self.bestProbs = []

    def scoreFn(self, t):
        f_s_a = self.featExtractor.getFeatures(self.states[t], self.actions[t])
        sumVal = 0
        print(f_s_a)
        # util.pause()
        for action in self.getLegalActions(self.states[t]):
        # required: since sum(prob * action) = 1 = f(s,a), hence score will constantly give 0 if not done
        # if action != self.actions[t]:
            f_s_b = self.featExtractor.getFeatures(self.states[t], action)
            print(f_s_b)
            # util.pause()
            pi_b_s = self.probs[t][action]
            print(pi_b_s)
            util.pause()
            # print(f"action = {action}, prob = {pi_b_s}")
            sumVal += (f_s_b * pi_b_s)
            # sumVal += np.dot(pi_b_s, f_s_b)
            # print(sumVal)
            # util.pause()
        # print(sumVal)
        # util.pause()
        score = f_s_a - sumVal
    #   print(score)
        return score

    # NOTE: Fixed log_prob by using negative log (gradient ascent instead of descent)
    def scoreFnUsingGradients(self, t):
        # taking exactly the gradient of log
        # probs = [self.probs[t][action] for action in self.probs[t] if action == action[t]]
        log_probs = np.negative(np.log(self.bestProbs))
        # print(log_probs)
        # print(np.negative(log_probs))
        # util.pause()
        grad = np.gradient(log_probs)
        # print(grad)
        return grad[t]
    
    # Alternate formula as implemented in sutton --> grad(ln(policy) = grad(policy)/policy
    def scoreFnGradDiv(self, t):
        # probs = []
        # for t in range(len(self.probs)):
        #     for action in self.probs[t]:
        #         if action == self.actions[t]:
        #             probs.append(self.probs[t][action])
        # # probs = [self.probs[t][action] for action in self.probs[t]  for t in self.probs if action == self.actions[t]]
        # # print(probs)
        # # util.pause()
        grad = np.gradient(self.bestProbs)
        vect = grad/self.bestProbs
        return vect[t]

    # def getAction(self, state):
    #     print(self.getLegalActions(state))
    # We then (randomly) sample the action from the probabilities
    # This is where differs from approximate Q learning
    # in approximate q learning we still use argmax
    # But to avoid greedily fetching and to introduce exploration
    # Use random sampling.
            # self.stateFeatures = self.featExtractor.getFeatures(state, action)
    def softmax(self, vector):
        # Take away the max to avoid overflow
        maxVal = np.max(vector)
        vectorCopy = vector.copy()
        newVector = vectorCopy - maxVal
        return np.exp(newVector)/sum(np.exp(newVector))
    
    def saveWeights(self):
        saveWeights = self.theta.copy()
        with open('weights.pk', 'wb') as f:
            pickle.dump(saveWeights, f)
        print("Learnt parameters saved")


class ReinforceBaseline(Reinforce):
    def __init__(self, alpha_w = 1, extractor="ExtendedExtractor", **args):
        # Initializing REINFORCE
        Reinforce.__init__(self, **args)
        self.alpha_w = alpha_w

        # Initializing value function weights
        self.w = util.Counter()

        # state features 
        self.featStateExtractor = util.lookup(extractor, globals())()

    # only override update
    def update(self):
        ep_len = len(self.states)
        avgReward = sum(self.rewards) / len(self.rewards)
        for t in range(ep_len):
            # Calculate discounted return from ith state to end of trajectory
            G_t = 0
            for k in range(0, ep_len):  # since rewards are collected till the last state
                G_t += np.power(self.discount, k-t-1) * self.rewards[k]
                # print(f"returns = {G_t}")
                # util.pause()
            
            # stateFeatures = self.featStateExtractor.getFeatures(self.states[t]) 
            # v_t = self.w * stateFeatures
            A_t = G_t - avgReward

            # getting score function for theta
            # scoreVal = self.scoreFnUsingGradients(t)
            scoreVal = self.scoreFnGradDiv(t)
            # scoreVal = self.scoreFn(t)

            # Updaing theta and w
            featuresStateAction = self.featExtractor.getFeatures(self.states[t], self.actions[t])
            for feature in featuresStateAction:
                self.theta[feature] += self.alpha * np.power(self.discount, t) * A_t * scoreVal
            # print(self.theta)
            # util.pause()

            if self.episodesSoFar % 50 == 0:
                # print("Saving weights")
                print(self.theta)
                self.saveWeights()
                #print(f"value = {value}, \ndelta = {delta}, \nW(s(t)) = {self.w[self.states[t]]}")
                print(f"returns = {G_t}, rewards = {self.rewards}, \nactions = {self.actions}, \nprobs = {self.probs}")

            # TODO: Check this:
            # gradient of w*x = grad(w1x1 + w2x2 + w3x3 + ..) = w1 + w2 + w3 + ..
            # for feature in stateFeatures:
            #     # for scalar
            #     grad = self.w[feature]
            #     self.w[feature] += self.alpha_w * A_t * grad

            # print(f"Feature weights gradient = {sum(self.w[feature])}")\

class ReinforceWithValueBaseline(ReinforceBaseline):
    def stopEpisode(self):
        ReinforcementAgent.stopEpisode(self)
        # print("Episode ended")
        # get all s, a, r and update
        # print(
            # f"length of state = {len(self.states)}, length of reward = {len(self.rewards)}")
        #ep_len = len(self.states)
        ep_len = len(self.states)
        #baseline = sum(self.rewards) / len(self.rewards)
        for t in range(ep_len):
            # Calculate discounted return from ith state to end of trajectory
            G_t = 0
            for k in range(t+1, ep_len):  # since rewards are collected till the last state
                G_t += np.power(self.discount, k-t-1) * self.rewards[k]

            # getting score function
            # scoreVal = self.scoreFnUsingGradients(t)
            scoreVal = self.scoreFnGradDiv(t)
            #scoreVal = self.scoreFn(t)
            # Updaing theta and w
            featuresStateAction = self.featExtractor.getFeatures(self.states[t], self.actions[t])

            value = self.x * self.w
            delta = (G_t - value) * 1e-5
            self.w[self.states[t]] += self.alpha_w * delta * self.x[self.states[t]]
            # util.pause()

            for feature in featuresStateAction:
                self.theta[feature] += self.alpha * np.power(self.discount, t) * delta * scoreVal
        if self.episodesSoFar % 100 == 0:
            # print("Saving weights")
            print(self.theta)
            self.saveWeights()
            #print(f"value = {value}, \ndelta = {delta}, \nW(s(t)) = {self.w[self.states[t]]}")
            print(f"rewards = {self.rewards}, \nactions = {self.actions}, \nprobs = {self.probs}")
        
        # reset
        self.states = []
        self.actions = []
        self.probs = []
        self.rewards = []