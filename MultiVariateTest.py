# Thsi file is meant to be used for testing the use of multiple continous variables
#in a multi objective context

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

#start by defining the 2 test functions
def MultiVariateTestFunction1(vec):
    x = vec[0]
    y = vec[1]
    return x+y

def MultiVariateTestFunction2(vec):
    x = vec[0]
    y = vec[1]
    f = (x**2)+((y-2)**2)
    return f


objectiveList = [MultiVariateTestFunction1, MultiVariateTestFunction2]
testObjectiveObject = ObjectiveFunction_multi(objectiveList) #fails
listOfObjectiveFunctions = [ObjectiveFunction_multi(MultiVariateTestFunction1), ObjectiveFunction_multi(MultiVariateTestFunction2)]
varTypeVariable = ["c", "c"]
LB = [-10, -10]
UB = [10, 10]
print(len(objectiveList))


print(len(listOfObjectiveFunctions))

# Select optimization problem type and associated parameters
test_gh = GnoweeHeuristics_multi(objective = listOfObjectiveFunctions, lowerBounds = LB, upperBounds = UB, varType = varTypeVariable, optimum = 0)
real = 0.5

#set parameters and hyperparameters
testProblemParameters = ProblemParameters_multi(objective = listOfObjectiveFunctions, lowerBounds = LB, upperBounds = UB, varType = varTypeVariable, optimum = 0)
populationSize =26
MaxPopulationIndex = populationSize -1
MaxGenerations = 100
#initiallize the population of 'Parents'
pop_test = [] #to be list of parents
parent_fitness = np.zeros((len(listOfObjectiveFunctions),1))


# Select optimization problem type and associated parameters

# Run optimization
(timeline, Last_pop) = Gnowee_multi.main_multi(test_gh)

length = len(timeline)
fitnesses = np.zeros(length)
generations = np.zeros(length)
for i in range(0,length):
    t = timeline[i]
    fitnesses[i] = t.fitness
    generations[i] = t.generation

plt.figure(1)
plt.scatter(generations, fitnesses)#, '-r', lw=2) # plot the data as a line
plt.xlabel('Generation', fontsize=14) # label x axis
plt.ylabel('Fittness', fontsize=14) # label y axis
plt.gca().grid() # add grid lines
#plt.show() # display the plot

#analyze the last population
#run a NonDomSort

plt.figure(2)
numPop = len(Last_pop)
function1 = np.zeros((numPop,1))
function2 = np.zeros((numPop,1))
for i in range(0, numPop):
    function1[i] = Last_pop[i].fitness[0]
    function2[i] = Last_pop[i].fitness[1]
plt.xlabel('Function 1', fontsize=15)
plt.ylabel('Function 2', fontsize=15)
plt.scatter(function1, function2)
print('\nThe result:\n', timeline[-1])
print('...')
print('Minimum function 1 fitness: ', min(function1))
print('Index of :', np.argmin(function1))#np.where(function1 == np.amin(function1))
print('x = :', Last_pop[np.argmin(function1)].variables)
print('function 2 fitness at this index: ', function2[np.argmin(function1)])
print('...')
print('Minimum function 2 fitness: ', min(function2))
print('Index of :', np.argmin(function2)) #np.where(function2 == np.amin(function2))
print('x = :', Last_pop[np.argmin(function2)].variables)
print('function 1 fitness at this index: ', function1[np.argmin(function2)])
plt.show()
print('Done')
