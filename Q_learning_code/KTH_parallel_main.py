import math
import time
import random
import numpy as np
import pandas as pd
from SRModel import SRModel
from KTHModel import KTHModel
from generic_routines import convertNumberBase, MaxLocBreakTies
import warnings
import pickle 
from multiprocessing import Pool
import os 
warnings.filterwarnings("ignore")
    
def generate_uIniPrice(numSessions, memory, numPlayers):
    uIniPrice = np.zeros((memory, numPlayers, numSessions))
    for iSession in range(numSessions):
        for iDepth in range(memory):
            for iPlayer in range(numPlayers):
                uIniPrice[iDepth, iPlayer, iSession] = random.uniform(0, 1)
    uIniPrice = np.array(uIniPrice)
    return uIniPrice


def generate_uExploreation(numPlayers):
    # Generating U(0,1) draws for price initialization
    uExploration = np.zeros((2, numPlayers))
    for iDecision in range(2):
        for iPlayer in range(numPlayers):
            uExploration[iDecision, iPlayer] = random.uniform(0, 1)
    return uExploration


def initQMatrices(game, delta, Profits, indexActions):
    Q = np.zeros((game.numStates, game.numiActions, game.numPlayers))
    # Randomize over the opponents decision
    for iPlayer in range(game.numPlayers):
        for iReport in range(game. numiActions):
            den = np.count_nonzero(indexActions[:, iPlayer] == iReport) * (1 - delta)
            Q[:, iReport, iPlayer] = np.ma.array(Profits[:, iPlayer],
                                                mask=(indexActions[:, iPlayer] != iReport)).sum() / den

    # Find initial optimal strategy
    strategyPrime = np.zeros((game.numStates, game.numPlayers))
    maxVal = np.zeros((game.numStates, game.numPlayers))
    for iPlayer in range(game.numPlayers):
        for iState in range(game.numStates):
            
            maxVal[iState, iPlayer], strategyPrime[iState, iPlayer] = MaxLocBreakTies(game.numiActions, Q[iState, :, iPlayer].copy())
    return Q, strategyPrime, maxVal


def initState(game, u, numPrices):
    p = np.floor(numPrices * u)
    stateNumber = 0
    actionNumber = computeActionNumber(game, p[0, :])
    return p, stateNumber, actionNumber


def computePPrime(game, uExploration, strategyPrime, state, iters):
    pPrime = np.zeros(game.numPlayers)
    #  Greedy with probability 1-epsilon, with exponentially decreasing epsilon
    for iPlayer in range(game.numPlayers):
        u = uExploration[:, iPlayer]
        if u[0] <= np.exp(-game.beta[iPlayer]*iters):
            pPrime[iPlayer] = math.floor(game.numiActions*u[1])
        else:
            pPrime[iPlayer] = strategyPrime[state, iPlayer].copy()
    return pPrime


def computeStateNumber(game, p):
    statevector = np.reshape(p.copy(), (1, game.lengthStates))
    return np.sum(game.cActions * statevector)


def computeActionNumber(game, p):
    return np.sum(game.cActions * p)


def computeStrategyNumber(game, maxLocQ):
    # Given the maxLocQ vectors, computes the lengthStrategies-digit strategy number
    iu = 0
    strategyNumber = np.zeros(game.lengthStrategies)
    for i in range(game.numPlayers):
        il = iu
        iu = iu + game.numStates
        strategyNumber[il:iu] = maxLocQ[:, i]
    return strategyNumber


def q_learning(game, convergedtime):
    # Initializing various quantities
    converged = np.zeros(game.numSessions)
    indexStrategies = np.zeros((game.lengthStrategies, game.numSessions))
    indexConverge = np.zeros((game.numSessions,game.lengthStrategies))
    uIniPrice = generate_uIniPrice(game.numSessions, 1, 2)

    # Loop over numSessions
    for iSession in range(game.numSessions):
        #print('Session = ', iSession, ' started')
        start_time = time.time()
        # Learning Phase
        # Initializing Q matrices
        Q, strategyPrime, maxVal = initQMatrices(game, game.delta, game.Profits, game.indexActions)
        strategy = strategyPrime.copy()
        # Randomly initializing prices and state
        p, statePrime, actionPrime = initState(game, uIniPrice[:, :, iSession], game.numiActions)
        state = int(statePrime)
        # Loop
        itersInStrategy = 0
        convergedSession = -1

        strategyFix = np.zeros((game.numStates, game.numPlayers))
        for iters in range(game.maxIters):
            # Iterations counter

            # Generating exploration random numbers
            uExploration = generate_uExploreation(2)
            # Compute pPrime by balancing exploration vs. exploitation
            pPrime = computePPrime(game, uExploration, strategyPrime, state, iters)
            p[0, :] = pPrime.copy()
            statePrime = 0
            actionPrime = int(computeActionNumber(game, pPrime))
            for iPlayer in range(game.numPlayers):
                # Q matrices and strategies update
                oldq = Q[state, int(pPrime[iPlayer]), iPlayer]
                newq = oldq + game.alpha[iPlayer] * (game.Profits[actionPrime, iPlayer] + game.delta * maxVal[statePrime, iPlayer] - oldq)
                Q[state, int(pPrime[iPlayer]), iPlayer] = newq
                """             
                if newq > maxVal[state, iPlayer]:
                    maxVal[state, iPlayer] = newq
                    if strategyPrime[state, iPlayer] != pPrime[iPlayer]:
                        strategyPrime[state, iPlayer] = pPrime[iPlayer]
                """
			#need to update new strategyPrime and maxVal
            strategyPrime = np.zeros((game.numStates, game.numPlayers))
            maxVal = np.zeros((game.numStates, game.numPlayers))
            for iPlayer in range(game.numPlayers):
                for iState in range(game.numStates):
                    maxVal[iState, iPlayer], strategyPrime[iState, iPlayer] = MaxLocBreakTies(game.numiActions, Q[iState, :, iPlayer].copy())
 
			# Assessing convergence
            if np.array_equiv(strategyPrime[state, :], strategy[state, :]):
                itersInStrategy = itersInStrategy + 1
            else:
                # print(strategyPrime[state, :])
                # print(strategy)
                itersInStrategy = 1

            # Check for convergence in strategy
            if convergedSession == -1:
                # Maximum number of iterations exceeded
                if iters >= game.maxIters - 1:
                    convergedSession = 0
#                    strategyFix = strategy
                    strategyFix = strategy.copy()                #testing

                # Convergence in strategy reached
 #               if itersInStrategy == 100000:
                #if itersInStrategy == 10000:
                if itersInStrategy == convergedtime:
                    convergedSession = 1
#                    strategyFix = strategy
                    strategyFix = strategy.copy()               #testing
                    #print('iters: ', iters, " strategy: ", strategy)
                    #print(iters)
                    #print("maxVal: ", maxVal)
                    #print("strategyPrime: ", strategyPrime)
                    #print("pPrime: ", pPrime)
                    #print("Q: ", Q)

            # Check for loop exit criteria
            if convergedSession != -1:
                break
            # if no converge yet, update and iterate
            strategy[state, :] = strategyPrime[state, :].copy()
            state = statePrime

        converged[iSession] = convergedSession
        indexStrategies[:, iSession] = computeStrategyNumber(game, strategyFix)
        indexConverge[iSession] = (convergedSession, iters)
        # print(convergedSession)
        # if convergedSession == 1:
        #     print("Session =", iSession, "converged")
        # else:
        #     print("Session =", iSession, "did not converge")

        end_time = time.time()
        # print("session time:", end_time-start_time)
        # print('\n')
    return converged, indexStrategies, indexConverge


def single_process(alpha):
    c_list = [500,1000,2000,5000,10000]
    #c_list = [10000]
    for cl in c_list:
        df = pd.DataFrame(columns=['alpha','beta','total_sessions','converged_times','i ndexStrategie','indexConverge'])
        beta_list = np.linspace(0.0005,0.00001,20)
        path = 'KTH_parallel_result'
        if not os.path.exists(path):
            os.mkdir(path)
        
        for beta in beta_list:
        
            kth = KTHModel(alpha = alpha, beta = beta)
            converged, indexStrategies,indexConverge = q_learning(kth,convergedtime= cl)
            df = df.append({'alpha':alpha,'beta':beta,'total_sessions':kth.numSessions,'converged_times':cl,'indexStrategie':indexStrategies,'indexConverge':indexConverge },ignore_index=True)
            print('alpha:',alpha,' beta:',beta,' converged_times:',cl,' finish!')
        
        with open(path + '/' + 'a_'+ str(alpha) + 'b_'+ str(beta) + '_cl_' + str(cl) + '.pkl', 'wb') as f:
            pickle.dump(df, f)
 
def run_complex_operations(operation, input, pool):
    pool.map(operation, input)
      
processes_count = 10
     
if __name__ == '__main__':
    alpha_list = np.linspace(0.025,0.25,20)
    processes_pool = Pool(processes_count)
    run_complex_operations(single_process,alpha_list , processes_pool)