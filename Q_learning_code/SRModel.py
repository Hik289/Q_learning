import numpy as np
import pandas as pd
from itertools import product
from generic_routines import convertNumberBase, MaxLocBreakTies

class SRModel(object):
    def __init__(self, **kwargs):
        self.numPlayers = 2
        # number of individual actions
        self.numiActions = 27
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
        self.buyerActions = np.array(list(product([0, 1, 2], repeat=3)))
        self.sellerActions = np.array(list(product([0, 1, 2], repeat=3)))
        self.indexActions = self.init_indexActions()
        self.Prices = self.init_Prices()
        self.Profits = self.init_Profits()

        # QL
        self.numSessions = 100
        self.maxIters = 1000000
        # self.delta = kwargs.get('delta', 0.95)
        # self.alpha = kwargs.get('alpha', 0.15) * np.ones(self.numPlayers)
#        self.beta = kwargs.get('beta', 0.0001) * np.ones(self.numPlayers)
        self.delta = kwargs.get('delta', 0.95)
        self.alpha = kwargs.get('alpha', 0.15) * np.ones(self.numPlayers)
        self.beta = kwargs.get('beta', 0.0001) * np.ones(self.numPlayers)
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
        Profits = np.zeros((self.numActions, self.numPlayers))
        Actions = np.zeros((self.numActions, 6))
        for i in range(self.numActions):
            b = int(self.indexActions[i][0])
            s = int(self.indexActions[i][1])
            vb, cb, vb_a = self.buyerActions[b]
            vs, cs, cs_a = self.sellerActions[s]
            Actions[i] = vb, cb, vb_a, vs, cs, cs_a
            if (vb == vs) and (cb == cs):
                # if all report coincide
                # profit for buyer: true value - price - investment_buyer
                pb = self.trueValue - self.Prices[cb][vs] - self.buyerInvestment
                # profit for seller: price - true cost - investment_seller
                ps = self.Prices[cb][vs] - self.trueCost - self.sellerInvestment
            elif (vb != vs) and (cb == cs):
                # only the value reports differ
                # buyer enters into arbitration stage
                if vb_a == 0:
                    # no_trade
                    pb = 0 - self.buyerInvestment
                    ps = 0 - self.sellerInvestment
                elif vb_a == 1:
                    pb = 0.5 * 0 + 0.5 * (self.trueValue - 205) - self.buyerInvestment
                    ps = 0.5 * 0 + 0.5 * (205 - self.trueCost) - self.sellerInvestment
                else:
                    pb = 0.5 * (self.trueValue - 205) + 0.5 * (self.trueValue - 255) - self.buyerInvestment
                    ps = 0.5 * (255 - self.trueCost) + 0.5 * (205 - self.trueCost) - self.sellerInvestment

                # buyers is fined 300
                pb = pb - 300
                # the seller is rewarded a bonus of 300 if the second report of the buyer matches the first stage report
                # of seller. In other cases, the seller is also fined 300.
                if vb_a == vs:
                    ps = ps + 300
                else:
                    ps = ps - 300
            elif (vb == vs) and (cb != cs):
                # only the cost reports differ
                # seller enters into arbitration stage
                if cs_a == 0:
                    # no_trade
                    pb = 0 - self.buyerInvestment
                    ps = 0 - self.sellerInvestment
                elif cs_a == 1:
                    pb = 0.5 * 0 + 0.5 * (self.trueValue - 125) - self.buyerInvestment
                    ps = 0.5 * 0 + 0.5 * (125 - self.trueCost) - self.sellerInvestment
                else:
                    pb = 0.5 * (self.trueValue - 125) + 0.5 * (self.trueValue - 75) - self.buyerInvestment
                    ps = 0.5 * (125 - self.trueCost) + 0.5 * (75 - self.trueCost) - self.sellerInvestment

                # seller is fined 300
                ps = ps - 300
                # the buyer is rewarded a bonus of 300 if the second report of the seller matches the first stage report
                # of buyer. In other cases, the buyer is also fined 300.
                if cs_a == cb:
                    pb = pb + 300
                else:
                    pb = pb - 300
            else:
                if vb_a == 0:
                    # no_trade
                    pb1 = 0 - self.buyerInvestment
                    ps1 = 0 - self.sellerInvestment
                elif vb_a == 1:
                    pb1 = 0.5 * 0 + 0.5 * (self.trueValue - 205) - self.buyerInvestment
                    ps1 = 0.5 * 0 + 0.5 * (205 - self.trueCost) - self.sellerInvestment
                else:
                    pb1 = 0.5 * (self.trueValue - 205) + 0.5 * (self.trueValue - 255) - self.buyerInvestment
                    ps1 = 0.5 * (255 - self.trueCost) + 0.5 * (205 - self.trueCost) - self.sellerInvestment
                if vb_a == vs:
                    ps1 = ps1 + 300

                if cs_a == 0:
                    # no_trade
                    pb2 = 0 - self.buyerInvestment
                    ps2 = 0 - self.sellerInvestment
                elif cs_a == 1:
                    pb2 = 0.5 * 0 + 0.5 * (self.trueValue - 125) - self.buyerInvestment
                    ps2 = 0.5 * 0 + 0.5 * (125 - self.trueCost) - self.sellerInvestment
                else:
                    pb2 = 0.5 * (self.trueValue - 125) + 0.5 * (self.trueValue - 75) - self.buyerInvestment
                    ps2 = 0.5 * (125 - self.trueCost) + 0.5 * (75 - self.trueCost) - self.sellerInvestment
                if cs_a == cb:
                    pb2 = pb2 + 300

                pb = 0.5 * (pb1 + pb2) - 300
                ps = 0.5 * (ps1 + ps2) - 300
            Profits[i] = [pb, ps]
        action_df = pd.DataFrame(Actions)
        action_df.to_excel("action.xlsx")
        profit_df = pd.DataFrame(Profits)
        profit_df.to_excel("profit.xlsx")
        return Profits

    def init_Q(self):
        Q = np.zeros((self.numStates, self.numiActions, self.numPlayers))
        # Randomize over the opponents decision

        for iAgent in range(self.numPlayers):
            for iReport in range(self.numiActions):
                den = np.count_nonzero(self.indexActions[:, iAgent] == iReport) * (1 - self.delta)
 #               Q[:, iReport, iAgent] = np.ma.array(self.Profits[:, iAgent],
 #                                                   mask=(self.indexActions[:, iAgent] != iReport)).sum() / den
                Q[:, iReport, iAgent] = np.ma.array(self.Profits[:, iAgent],
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
    sr = SRModel()
    sr.init_Q()
    print("Prices: ", sr.Prices)
    print("Profits: ", sr.Profits)
    print("Q: ", sr.Q[0])
    print("cStates: ", sr.cStates)
    print("cActions: ", sr.cActions)
    print("sr.trueValue: ", sr.trueValue)
    print("sr.trueCost: ", sr.trueCost)
    print("sr.numStates: ", sr.numStates)
    Q_df = pd.DataFrame(sr.Q[0])
    Q_df.to_excel("Q.xlsx")