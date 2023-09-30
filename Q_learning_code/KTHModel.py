import numpy as np
from generic_routines import convertNumberBase, MaxLocBreakTies

class KTHModel(object):
    def __init__(self, **kwargs):
        self.numPlayers = 2
        self.value = np.array([200, 250, 320])
        self.cost = np.array([130, 80, 10])
        # number of individual actions
        self.numiActions = 9
        self.buyerActions = np.array([[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2], [2, 0], [2, 1], [2, 2]])
        self.sellerActions = np.array([[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2], [2, 0], [2, 1], [2, 2]])
        self.memory = kwargs.get('memory', 0)
        self.numActions = self.numiActions ** self.numPlayers
        self.numStates = self.numiActions ** (self.numPlayers * self.memory)
        self.indexActions = self.init_indexActions()
        self.Prices = self.init_Prices()
        self.Profits = self.init_Profits()

        # QL
        self.numSessions = 100
        self.maxIters = 1000000
        self.delta = kwargs.get('delta', 0.95)
        self.alpha = kwargs.get('alpha', 0.15) * np.ones(self.numPlayers)
        self.beta = kwargs.get('beta', 100) * np.ones(self.numPlayers)
        self.lengthStates = self.numPlayers * self.memory
        self.lengthStrategies = self.numPlayers * self.numStates
        self.Q = self.init_Q()
        self.cStates = self.init_cStates()
        self.cActions = self.init_cActions()

    def init_indexActions(self):
        indexActions = []
        for i in range(self.numActions):
            indexActions.append(convertNumberBase(i, self.numiActions, self.numPlayers))
        indexActions = np.array(indexActions)
        return indexActions

    def init_Prices(self):
        pricesArray = np.zeros((3, 3))
        for cb in range(0, 3):
            for vs in range(0, 3):
                # (vs-200) - (130-cb) + 165
                pricesArray[cb][vs] = (self.value[vs] - 200) - (130 - self.cost[cb]) + 165
        return pricesArray

    def init_Profits(self):
        Profits = np.zeros((self.numActions, self.numPlayers))
        for i in range(self.numActions):
            b = int(self.indexActions[i][0])
            s = int(self.indexActions[i][1])
            vb, cb = self.buyerActions[b]
            vs, cs = self.sellerActions[s]
            # profit for buyer: vs - price - max(0, cs - cb)
            pb = self.value[vs] - self.Prices[cb][vs] - max(0, self.cost[cs] - self.cost[cb])
            # profit for seller: price - cb - max(0, vs - vb)
            ps = self.Prices[cb][vs] - self.cost[cb] - max(0, self.value[vs] - self.value[vb])
            Profits[i] = [pb, ps]
        return Profits

    def init_Q(self):
        Q = np.zeros((self.numActions, self.numiActions, self.numPlayers))
        for iPlayer in range(self.numPlayers):
            for iReport in range(self.numiActions):
                den = np.count_nonzero(self.indexActions[:, iPlayer] == iReport) * (1 - self.delta)
                Q[:, iReport, iPlayer] = np.ma.array(self.Profits[:, iPlayer],
                                                    mask=(self.indexActions[:, iPlayer] != iReport)).sum() / den
        return Q

    def init_cStates(self):
        """Initialize cStates (used for q-learning)"""
        x = np.arange(self.lengthStates - 1, -1, -1)
        cStates = self.numiActions ** x
        return cStates

    def init_cActions(self):
        """Initialize cActions (used for q-learning)"""
        x = np.arange(self.numPlayers - 1, -1, -1)
        cActions = self.numiActions ** x
        return cActions

if __name__ == '__main__':
    kth = KTHModel()
    print("Prices", kth.Prices)
    print("Profits", kth.Profits)
    print("Q", kth.Q[0])
    print(kth.Profits[65])
