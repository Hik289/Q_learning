import numpy as np
import random

def convertNumberBase(n, b, l):
    '''
    Converts an integer n from base 10 to base b,
    generating a vector of integers of length l
    '''
    tmp = n
    ans = np.zeros(l)
    for i in range(1, l+1):
        ans[l-i] = int(tmp % b)
        tmp = np.floor(tmp/b)
    return ans


def MaxLocBreakTies(n, x):
    # MaxLocBreakTies: Given the n*1 array x, finds the maximum m and the position of m in x
    tied = []
    h = 0
    m = max(x)
    for i in range(0, n):
#        m = max(x)
        if x[i] == m:
            h = h + 1
            tied.append(i)
    if h > 1:
        u = random.uniform(0, 1)
        p = tied[int(h*u)]
    else:
        p = tied[0]
    return m, p

