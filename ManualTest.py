#for manually testing various parts of the the code
import numpy as np
# Gnowee Modules
import Gnowee_multi_testNSGA
from ObjectiveFunction_Multi import ObjectiveFunction_multi
from Constraints import Constraint
from GnoweeHeuristics_multi import GnoweeHeuristics_multi
from OptiPlot_multi import plot_vars
import matplotlib.pyplot as plt

import NSGAmethods
from NSGAmethods import pop_member, FastNonDomSort_Gnowee, Population_class, get_fronts_crowding_distances

from NSGATestFunctions import f1
from NSGATestFunctions import f2
from GnoweeUtilities_multi import ProblemParameters_multi, Event, Parent_multi

import random
#original NSGA files
from nsga2.utils import NSGA2Utils
from nsga2.problem import Problem


#set up some testing functions

def testFunction1(x):
    return -x

def testFunction2(x):
    t1 = (x-5)**2
    t2 = -t1 +25
    return t2
#set up a population

PopulationNumber = 10

objectiveList = [testFunction1, testFunction2]
testObjectiveObject = ObjectiveFunction_multi(objectiveList) #fails
listOfObjectiveFunctions = [ObjectiveFunction_multi(testFunction1), ObjectiveFunction_multi(testFunction2)]
varTypeVariable = "c"
LB = [-55]
UB = [55]
print(len(objectiveList))
#print len(testObjectiveObject)

print(len(listOfObjectiveFunctions))

#test_problem = ProblemParameters_multi(objective = listOfObjectiveFunctions, lowerBounds = LB, upperBounds = UB, varType = varTypeVariable)
test_gh = GnoweeHeuristics_multi(objective = listOfObjectiveFunctions, lowerBounds = LB, upperBounds = UB, varType = varTypeVariable, optimum = 0)

#set parameters and hyperparameters
testProblemParameters = ProblemParameters_multi(objective = listOfObjectiveFunctions, lowerBounds = LB, upperBounds = UB, varType = varTypeVariable, optimum = 0)
populationSize =PopulationNumber
problemParametersObject =testProblemParameters
numVariables = len(problemParametersObject.varType)
objectiveFunctionlist = problemParametersObject.objective
num_cont_int_bin_variables = len(problemParametersObject.lb)
num_Features = num_cont_int_bin_variables
var_range = []
for k in range(0, num_cont_int_bin_variables):
    var_range += (problemParametersObject.lb[k], problemParametersObject.ub[k])


x_array = np.zeros((populationSize, 1))
F1_array = np.zeros((populationSize, 1))
F2_array = np.zeros((populationSize, 1))
Fitness_array = np.zeros((populationSize,2))
for i in range(1, populationSize+1):
    x_array[i-1] = i
    #print "x = ", x_array[i-1]
    F1_array[i-1] = testFunction1(i)
    #print "F1 = ", F1_array[i-1]
    F2_array[i-1] = testFunction2(i)
    #print "F2 = ", F2_array[i-1]
    Fitness_array[i-1, 0] = testFunction1(i)
    Fitness_array[i-1, 1] = testFunction2(i)
#convert to an R_t array of evaluated pop_members

R_t = []
for i in range(0,populationSize):
    entry = pop_member(numVariables, var_range, features=x_array[i], fitness=Fitness_array[i,:], changeCount=0, stallCount=0, Evaluated=1)
    R_t.append(entry)

#test the dominates function
if R_t[0].dominates(R_t[1]):
    print(" x = ", R_t[0].features, "dominates x = ", R_t[1].features)
#test the nonDom Sort
N = populationSize
sorted_population = FastNonDomSort_Gnowee(R_t, N, problemParametersObject)
print("Test Done")