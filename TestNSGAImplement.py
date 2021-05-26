
#this is a dry test run of my NSGA code
#(as interfaced with some Gnowee stuff)
import numpy as np
# Gnowee Modules
import Gnowee_multi_testNSGA
from ObjectiveFunction_Multi import ObjectiveFunction_multi
from Constraints import Constraint
from GnoweeHeuristics_multi import GnoweeHeuristics_multi
from OptiPlot_multi import plot_vars
import matplotlib.pyplot as plt

import NSGAmethods

from NSGATestFunctions import basic1
from NSGATestFunctions import basic2
from GnoweeUtilities_multi import ProblemParameters_multi, Event, Parent_multi

import random
#original NSGA files
from nsga2.utils import NSGA2Utils
from nsga2.problem import Problem

objectiveList = [basic1, basic2]
testObjectiveObject = ObjectiveFunction_multi(objectiveList) #fails
listOfObjectiveFunctions = [ObjectiveFunction_multi(basic1), ObjectiveFunction_multi(basic2)]
varTypeVariable = "c"
LB = [0.5]
UB = [120]
print len(objectiveList)
#print len(testObjectiveObject)

print len(listOfObjectiveFunctions)

#test_problem = ProblemParameters_multi(objective = listOfObjectiveFunctions, lowerBounds = LB, upperBounds = UB, varType = varTypeVariable)
test_gh = GnoweeHeuristics_multi(objective = listOfObjectiveFunctions, lowerBounds = LB, upperBounds = UB, varType = varTypeVariable, optimum = 0)

real = 0.5
# Run optimization
#(timeline) = Gnowee_multi_testNSGA.main(test_gh, real)

#set parameters and hyperparameters
testProblemParameters = ProblemParameters_multi(objective = listOfObjectiveFunctions, lowerBounds = LB, upperBounds = UB, varType = varTypeVariable, optimum = 0)
populationSize =26
MaxPopulationIndex = populationSize -1
MaxGenerations = 100
#initiallize the population of 'Parents'
pop_test = [] #to be list of parents
parent_fitness = np.zeros((len(listOfObjectiveFunctions),populationSize))
for i in range(0, populationSize):
    #choose a random number between the upper and lower bounds
    #x = [random.uniform(LB[0], UB[0])]
    x = np.ones((1,1))*i
    #evaluate their fittness

    for t in range(0,len(listOfObjectiveFunctions)):
        parent_fitness[t,i] = listOfObjectiveFunctions[t].func(x[0])
    pop_test.append(Parent_multi(fitness=parent_fitness[:,i], variables=x))
#initallize the population of 'children'
children_test = []
for i in range(0, populationSize):
    #choose a random number between the upper and lower bounds
    #x = random.uniform(LB[0], UB[0])
    x = i
    children_test.append(x)

#try to run a single generation
newish_population = NSGAmethods.runNSGA_generation(pop_test,children_test, testProblemParameters)
#stuff to set up the NSGA code's child creation proceedure
problem = Problem(num_of_variables=1, objectives=[basic1, basic2], variables_range=[(LB[0], UB[0])])
num_of_individuals = populationSize
num_of_tour_particips=2
tournament_prob=0.9
crossover_param=2
mutation_param=5
used_utils = NSGA2Utils(problem, num_of_individuals, num_of_tour_particips, tournament_prob, crossover_param, mutation_param)

new_children_listOfIndividuals = used_utils.create_children(newish_population)
numChildren = len(new_children_listOfIndividuals)
#convert children a make them into a list of unevaluated arrays
#doing this is painful



#set up a loop and repeat until max generations
for GenCounter in range(0, MaxGenerations):
    #step 1: combine parents and children into one population (which is technically done in runNSGA_generation)
    #STEP 1A: Convert the children from a list of individuals (which store evaluation information)
    #to a list of arrays (the way Gnowee store them) just containing their features
    GnoweeChildList = []
    for childNum in range(0, numChildren):
        VarValue = new_children_listOfIndividuals[childNum].features[0]
        GnoweeChildList.append(VarValue)
    #STEP 1B:Set the Parent Population by converting from the Population class to
    #a list of Parent_multi objects (used in Gnowee)
    GnoweeParentObjectList = []
    numParents = len(newish_population)
    for ParentNum in range(0, numParents):
        ParentObject = Parent_multi(variables = newish_population.population[ParentNum].features, fitness= newish_population.population[ParentNum].fitness)
        GnoweeParentObjectList.append(ParentObject)
    #STEP 2: run an NSGA generation of sorting/evaluation
    newish_population = []
    newish_population = NSGAmethods.runNSGA_generation(GnoweeParentObjectList,GnoweeChildList, testProblemParameters)
    #STEP 3: Create a new set of children (standing in for alll the other Gnowee Heuristics)
    new_children_listOfIndividuals = used_utils.create_children(newish_population)
    numChildren = len(new_children_listOfIndividuals)

    #for debugging puposes:
    if GenCounter == 80:
        print 'Check Fronts'
    print 'Generation = ', GenCounter, '/ ', MaxGenerations

#Plot the results
LastPopulation = newish_population
numPop = len(LastPopulation.population)
function1 = np.zeros((numPop,1))
function2 = np.zeros((numPop,1))
for i in range(0, len(LastPopulation.population)):
    function1[i] = LastPopulation.population[i].fitness[0]
    function2[i] = LastPopulation.population[i].fitness[1]
plt.xlabel('Function 1', fontsize=15)
plt.ylabel('Function 2', fontsize=15)
plt.scatter(function1, function2)
plt.show()
print "run complete"
