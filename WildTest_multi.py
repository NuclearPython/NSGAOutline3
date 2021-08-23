#this file is for testing a updated version of Gnowee_multi with a single objective function

# Gnowee Modules
import Gnowee_multi
from ObjectiveFunction_Multi import ObjectiveFunction_multi
from Constraints import Constraint
from GnoweeHeuristics_multi import GnoweeHeuristics_multi
import numpy as np
from OptiPlot_multi import plot_vars
import matplotlib.pyplot as plt
from GnoweeUtilities_multi import ProblemParameters_multi, Event, Parent_multi

#ALEX'S FILES:

import NSGAmethods
from NSGAmethods import pop_member, FastNonDomSort_Gnowee, Population_class, get_fronts_crowding_distances

import random
#original NSGA files
from nsga2.utils import NSGA2Utils
from nsga2.problem import Problem

# User Function Module
from TestFunction import testfittness
testarray = np.zeros(6)
print(testfittness(testarray))
sz = 1000
all_ints = ["i" for i in range(sz)]
LB = np.zeros(sz)
UppB = np.ones(sz)

#some _multi specific things:
listOfObjectiveFunctions = [ObjectiveFunction_multi(TestFunction)]
# Select optimization problem type and associated parameters
gh = GnoweeHeuristics_multi(objective=listOfObjectiveFunctions,
                      lowerBounds=LB, upperBounds=UppB,
                      varType=all_ints, optimum=0)
print(gh)

# Run optimization
(timeline) = Gnowee_multi.main_multi(gh)

length = len(timeline)
fitnesses = np.zeros(length)
generations = np.zeros(length)
for i in range(0,length):
    t = timeline[i]
    fitnesses[i] = t.fitness
    generations[i] = t.generation

plt.plot(generations, fitnesses, '-r', lw=2) # plot the data as a line
plt.xlabel('Generation', fontsize=14) # label x axis
plt.ylabel('Fittness', fontsize=14) # label y axis
plt.gca().grid() # add grid lines
plt.show() # display the plot
print('\nThe result:\n', timeline[-1])