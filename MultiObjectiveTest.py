# Gnowee Modules
import Gnowee_multi
from ObjectiveFunction_Multi import ObjectiveFunction_multi
from Constraints import Constraint
from GnoweeHeuristics_multi import GnoweeHeuristics_multi
import numpy as np
from OptiPlot import plot_vars
import matplotlib.pyplot as plt

# User Function Module
from TestFunction import testfittness
#second Function

#objective function array


testarray = np.zeros(6)
print(testfittness(testarray))
sz = 100
all_ints = ["i" for i in range(sz)]
LB = np.zeros(sz)
UppB = np.ones(sz)

# Select optimization problem type and associated parameters
gh = GnoweeHeuristics_multi(objective=ObjectiveFunction_multi(testfittness),
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