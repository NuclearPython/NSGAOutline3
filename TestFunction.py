import numpy as np
def testfittness(vec):
    length = len(vec)
    calc = 0
    for x in range(0,length):
        calc = calc+vec[x]
    fitness = length-calc
    return fitness