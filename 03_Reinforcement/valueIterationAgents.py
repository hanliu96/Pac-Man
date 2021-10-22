# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        while self.iterations != 0:
            self.iterations -= 1
            next_Vs = util.Counter()
            is_updated = util.Counter()
            states = self.mdp.getStates()
            for state in states:
                optimal_action = self.computeActionFromValues(state)
                if optimal_action:
                    next_V = self.computeQValueFromValues(state, optimal_action)
                    next_Vs[state] = next_V
                    is_updated[state] = 1

            for state in states: # update using V from k-1 iteration
                if is_updated[state]:
                    self.values[state] = next_Vs[state]



    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        Q_val = 0.0
        nextState_prob = self.mdp.getTransitionStatesAndProbs(state, action)
        for nextState, prob in nextState_prob:
            Q_val += prob * ((self.mdp.getReward(state, action, nextState))
                             + self.discount * self.getValue(nextState))
        return Q_val

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        actions = self.mdp.getPossibleActions(state)
        if not actions:
            return None
        optimal_action, optimal_reward = '', -1e8
        for action in actions:
            Q_val = 0.0
            nextState_prob = self.mdp.getTransitionStatesAndProbs(state, action)

            for nextState, prob in nextState_prob:
                Q_val += prob * ((self.mdp.getReward(state, action, nextState))
                                 + self.discount * self.getValue(nextState))
            if Q_val > optimal_reward:
                optimal_reward = Q_val
                optimal_action = action
        return optimal_action

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        states = self.mdp.getStates()
        states_size = len(states)
        count = 0
        while self.iterations != 0:
            self.iterations -= 1
            state = states[count % states_size]
            count += 1
            optimal_action = self.computeActionFromValues(state)
            if optimal_action:
                next_V = self.computeQValueFromValues(state, optimal_action)
                self.values[state] = next_V

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        # Compute predecessors of all states and store them in a set.
        all_states = self.mdp.getStates()
        all_pred = []
        count = 0
        idxState = util.Counter()

        for state_i in all_states:
            predSet = set()
            for state_j in all_states:
                actions = self.mdp.getPossibleActions(state_j)
                for action in actions:
                    nextState_prob = self.mdp.getTransitionStatesAndProbs(state_j, action)
                    for nextState, prob in nextState_prob:
                        if prob > 0 and nextState == state_i:
                            predSet.add(state_j)

            all_pred.append(predSet)
            idxState[state_i] = count
            count += 1

        " Initialize an empty priority queue."
        priorQueue = util.PriorityQueue()
        next_Vs = util.Counter()

        " Find the absolute value of the difference between the current values of s in" \
            " self.values and the highest Q_value"
        for state in all_states:
            if self.mdp.isTerminal(state):
                continue
            cur_V = self.getValue(state)
            optimal_action = self.computeActionFromValues(state)
            if optimal_action:
                next_V = self.computeQValueFromValues(state, optimal_action)
                next_Vs[state] = next_V
                diff = abs(next_V - cur_V)
                priorQueue.push(state, -diff)
            else:
                next_Vs[state] = cur_V

        " Iterations and update the value of s. "
        while self.iterations != 0:
            self.iterations -= 1
            if priorQueue.isEmpty():
                break

            # Update the value of s (if it is not a terminal state) in self.values
            poped = priorQueue.pop()
            if not self.mdp.isTerminal(poped):
                self.values[poped] = next_Vs[poped]

            # process each predecessor p of s
            for pred in all_pred[idxState[poped]]:
                cur_pred_V = self.getValue(pred)
                optimal_action_pred = self.computeActionFromValues(pred)

                if optimal_action_pred:
                    next_pred_V = self.computeQValueFromValues(pred, optimal_action_pred)
                    diff_pred = abs(cur_pred_V - next_pred_V)
                    next_Vs[pred] = next_pred_V

                    if diff_pred > self.theta:
                        priorQueue.update(pred, -diff_pred)
