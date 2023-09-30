import numpy as np
import pandas as pd
import random
from itertools import product
from generic_routines import convertNumberBase, MaxLocBreakTies

class SPIModel(object):
    def __init__(self, **kwargs):
        self.numPlayers = 2
        # number of individual actions
        self.numiActions = 18
        self.eps = kwargs.get('eps', 0.05)
        self.memory = kwargs.get('memory', 0)
        self.numStates = self.numiActions ** (self.numPlayers * self.memory)
        self.numPeriods = self.numStates + 1
        self.numActions = self.numiActions ** self.numPlayers
        self.buyerInvestment = 75
        self.sellerInvestment = 25
        self.trueValue = self.init_TrueValue(self.sellerInvestment)
        self.trueCost = self.init_TrueCost(self.buyerInvestment)
        self.value = np.array([200, 250, 320])
        self.cost = np.array([130, 80, 10])
        self.buyerActions = np.array([[0, 0, 0],[0, 0, 1],[0, 1, 0],[0, 1, 1],[0, 2, 0],[0, 2, 1],
                                      [1, 0, 0],[1, 0, 1],[1, 1, 0],[1, 1, 1],[1, 2, 0],[1, 2, 1],
                                       [2, 0, 0],[2, 0, 1],[2, 1, 0],[2, 1, 1],[2, 2, 0],[2, 2, 1]])
        self.sellerActions = np.array([[0, 0, 0],[0, 0, 1],[0, 1, 0],[0, 1, 1],[0, 2, 0],[0, 2, 1],
                                      [1, 0, 0],[1, 0, 1],[1, 1, 0],[1, 1, 1],[1, 2, 0],[1, 2, 1],
                                       [2, 0, 0],[2, 0, 1],[2, 1, 0],[2, 1, 1],[2, 2, 0],[2, 2, 1]])
    
        self.indexActions = self.init_indexActions()
        self.Prices = self.init_Prices()
        self.Profits = self.init_Profits()

        # QL
        self.numSessions = 100
        self.maxIters = 1000000
        self.delta = kwargs.get('delta', 0.95)
        self.alpha = kwargs.get('alpha', 0.15) * np.ones(self.numPlayers)
        self.beta = kwargs.get('beta', 0.00001) * np.ones(self.numPlayers)
        self.lengthStates = self.numPlayers * self.memory
        self.lengthStrategies = self.numPlayers * self.numStates
        self.Q = self.init_Q()
        self.cStates = self.init_cStates()
        self.cActions = self.init_cActions()

    def init_TrueValue(self, i):
        if i == 0:
            return 200
        elif i == 25:
            return 250
        else:
            return 320

    def init_TrueCost(self, i):
        if i == 0:
            return 130
        elif i == 25:
            return 80
        else:
            return 10

    def init_Prices(self):
        pricesArray = np.zeros((3, 3))
        for cb in range(0, 3):
            for vs in range(0, 3):
                # (vs-200) - (130-cb) + 165
                pricesArray[cb][vs] = (self.value[vs] - 200) - (130 - self.cost[cb]) + 165
        return pricesArray

    def init_indexActions(self):
        indexActions = []
        for i in range(self.numActions):
            indexActions.append(convertNumberBase(i, self.numiActions, self.numPlayers))
        indexActions = np.array(indexActions)
        return indexActions

    def init_Profits(self):
        Profits = np.zeros((2, self.numActions, self.numPlayers))
        Actions = np.zeros((self.numActions, 6))
        for i in range(self.numActions):
            b = int(self.indexActions[i][0])
            s = int(self.indexActions[i][1])
            vb, cb, vb_a = self.buyerActions[b]
            vs, cs, cs_a = self.sellerActions[s]
            Actions[i] = vb, cb, vb_a, vs, cs, cs_a

            
            # if no arbitration
            # profit for buyer: true value - price - investment_buyer
            pb_na = self.trueValue - self.Prices[cb][vs] - self.buyerInvestment
            # profit for seller: price - true cost - investment_seller
            ps_na = self.Prices[cb][vs] - self.trueCost - self.sellerInvestment
            Profits[0][i] = [pb_na, ps_na]

            # if into arbitration
            if vb_a == 0:
                # no_trade
                pb1 = -300 - self.buyerInvestment
                ps1 = -300 - self.sellerInvestment
            elif vb_a == 1:
                # accept offer price
                pb1 = self.trueValue - (self.value[vb] + 5) - 300 - self.buyerInvestment
                ps1 = (self.value[vb] + 5) + 300 - self.trueCost - self.sellerInvestment

            if cs_a == 0:
                # no_trade
                pb2 = -300 - self.buyerInvestment
                ps2 = -300 - self.sellerInvestment
            elif cs_a == 1:
                pb2 = self.trueValue - (self.value[cs] - 5) + 300 - self.buyerInvestment
                ps2 = (self.value[cs] - 5) - 300 - self.trueCost - self.sellerInvestment


            pb_a = 0.5 * (pb1 + pb2)
            ps_a = 0.5 * (ps1 + ps2)

            Profits[1][i] = [pb_a, ps_a]

        return Profits

    def init_Q(self):
        Q = np.zeros((self.numStates, self.numiActions, self.numPlayers))
        # Randomize over the opponents decision
        for iAgent in range(self.numPlayers):
            for iReport in range(self.numiActions):
                den = np.count_nonzero(self.indexActions[:, iAgent] == iReport) * (1 - self.delta)
                Q[:, iReport, iAgent] = np.ma.array((1 - self.eps) * self.Profits[0][:, iAgent] + self.eps * self.Profits[1][:, iAgent],
                                                    mask=(self.indexActions[:, iAgent] != iReport)).sum() / den
        return Q

    def init_cStates(self):
        """Initialize cStates"""
        x = np.arange(self.lengthStates - 1, -1, -1)
        cStates = self.numiActions ** x
        return cStates

    def init_cActions(self):
        """Initialize cActions"""
        x = np.arange(self.numPlayers - 1, -1, -1)
        cActions = self.numiActions ** x
        return cActions

if __name__ == '__main__':
    spi = SPIModel()
    print("Prices", spi.Prices)
    print("Profits", spi.Profits)
    print("Q", spi.Q[0])
    print("cStates", spi.cStates)
    print("cActions", spi.cActions)
    # profit1_df = pd.DataFrame(spi.Profits[0])
    # profit1_df.to_excel("p1.xlsx")
    # profit2_df = pd.DataFrame(spi.Profits[1])
    # profit2_df.to_excel("p2.xlsx")
