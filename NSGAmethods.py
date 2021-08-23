# uncompyle6 version 3.5.0
# Python bytecode 2.7 (62211)
# Decompiled from: Python 2.7.5 (default, Aug  7 2019, 00:51:29) 
# [GCC 4.8.5 20150623 (Red Hat 4.8.5-39)]
# Embedded file name: C:\Users\depila\Desktop\Graduate Research\GeneticAlgorithims\CreateNSGA2\NSGAOutline\NSGAmethods.py
# Compiled at: 2021-04-09 02:21:50
"""This file is meanto contain the methods and classes used for NSGA-2 and its interface with Gnowee"""
import numpy as np

def runNSGA_generation(P_t, Q_t, problemParametersObject):
    #INPUTS: 
    #P_t: list of parent objects (class from Gnowee)
    #Q_t: list of feature variables, each just the bare variables (the way Gnowee store the children)
    #problemParameterObject: contains the needed hyperparameters (class defined in Gnowee)
    #OUTPUTS:
    #newish_Population: a member of the newly defined Population Class which stores the class variable 'population'
    #(a list of members of the newly defined pop_member class) whicich only contains the members of P_t and Q_t 
    #that deserve to survive
    #note: because it is never sent to the NonDomSort, it is not organized into fronts (but they do have ranks)
    numVariables = len(problemParametersObject.varType)
    objectiveFunctionlist = problemParametersObject.objective
    num_cont_int_bin_variables = len(problemParametersObject.lb)
    num_Features = num_cont_int_bin_variables
    var_range = []
    for k in range(0, num_cont_int_bin_variables):
        var_range += (problemParametersObject.lb[k], problemParametersObject.ub[k])

    num_children = len(Q_t)
    child_list = []
    for k in range(0, num_children):
        child = pop_member(numVariables, var_range, features=Q_t[k] * np.ones((1, 1)))
        child.objectives = objectiveFunctionlist
        child_list.append(child)

    num_objectives = len(objectiveFunctionlist)
    for k in range(0, num_children):
        child_list[k].fitness = np.resize(child_list[k].fitness, (num_objectives, 1))
        #child_list[k].features = np.resize(child_list[k].features, (numVariables, 1))

    for k in range(0, num_children):
        #print 'k = ', k
        for i in range(0, num_objectives):
            #print 'i = ', i
            child_list[k].fitness[i] = problemParametersObject.objective[i].func(Q_t[k])

        child_list[k].Evaluated = 1

    num_Parents = len(P_t)
    parent_list = []
    for k in range(0, num_Parents):
        parent = pop_member(numVariables, var_range, features=P_t[k].variables, fitness=P_t[k].fitness, changeCount=P_t[k].changeCount, stallCount=P_t[k].stallCount, Evaluated=1)
        parent.objectives = objectiveFunctionlist
        parent_list.append(parent)
    #make changes to the dimensions of some of the data in order to make things more
    #compatible with Gnowee

    R_t = child_list + parent_list
    N = len(P_t)
    sorted_population = FastNonDomSort_Gnowee(R_t, N, problemParametersObject)
    new_population = Population_class()
    front_index = 0
    while len(new_population) + len(sorted_population.fronts[front_index]) <= N:
        eval_front = get_fronts_crowding_distances(sorted_population.fronts[front_index]) #gets a list of members of the front but with crowding distance evaluated for each
        new_population.extend(eval_front)
        front_index = front_index + 1

    eval_front = get_fronts_crowding_distances(sorted_population.fronts[front_index]) #make eval_ the last front in case you don't have enough just yet
    eval_front.sort(key=lambda individual: individual.crowding_distance, reverse=True) #since every front prior to this has been a less dominated front, only have to use crowding
    #distance as a tie breaker on the very last one
    new_population.extend(eval_front[0:N - len(new_population)]) #include any needed winner of the tie breaking
    return new_population #never asked for the fronts


def get_fronts_crowding_distances(front):
    if len(front) > 0:
        solutions_num = len(front)
        for individual in front:
            individual.crowding_distance = 0

        for m in range(len(front[0].objectives)):
            front.sort(key=lambda individual: individual.fitness[m])
            front[0].crowding_distance = 1000000000
            front[(solutions_num - 1)].crowding_distance = 1000000000
            m_values = [ individual.fitness[m] for individual in front ]
            scale = max(m_values) - min(m_values)
            if scale == 0:
                scale = 1
            for i in range(1, solutions_num - 1):
                front[i].crowding_distance += (front[(i + 1)].fitness[m] - front[(i - 1)].fitness[m]) / scale

    return front


def FastNonDomSort_Gnowee(R_t, N, ProblemParametersObject):
    """Takes in the combined parent and offspring population as arrays of EVALUATED pop_members.

    R_t is an array of pop_members

    Returns a population class with fronts"""
    old_code = 0
    if old_code ==0:
        #attempt a new implementation of Fast Non Dom Sort
        #INPUTS:
        #
        fronts = [[]]
        num_total_initial = len(R_t)
        population_inst = Population_class() #create instance of the population class
        for i in range(0, num_total_initial):
            R_t[i].objectives = ProblemParametersObject.objective
            population_inst.append(R_t[i])

        #print 'Test'
        for k in range(0, num_total_initial): #for p in P
            individual = population_inst.population[k]
            individual.domination_count = 0 #n_p:the number of solutions which DOM THE INDIVIUAL
            individual.dominated_solutions = [] #S_p: the set of solutions which the INDIVIDUAL DOMINATES
            for p in range(0, num_total_initial):
                other_individual = population_inst.population[p]
                if individual.dominates(other_individual):
                    individual.dominated_solutions.append(other_individual)
                elif other_individual.dominates(individual):
                    individual.domination_count += 1

            if individual.domination_count == 0:
                individual.rank = 0
                population_inst.fronts[0].append(individual)

        i = 0
        while len(population_inst.fronts[i]) > 0:
            temp = []
            for individual in population_inst.fronts[i]:
                for other_individual in individual.dominated_solutions:
                    other_individual.domination_count -= 1
                    if other_individual.domination_count == 0:
                        other_individual.rank = i + 1
                        temp.append(other_individual)

            i = i + 1
            population_inst.fronts.append(temp)

        return population_inst
    else:
        popNumber = len(R_t)
        fronts = [[]]
        num_total_initial = len(R_t)
        population_inst = Population_class() #create instance of the population class
        for i in range(0, num_total_initial):
            R_t[i].objectives = ProblemParametersObject.objective
            population_inst.append(R_t[i])

        #print 'Test'
        for individual in population_inst: #for p in P
            individual.domination_count = 0 #n_p:the number of solutions which DOM THE INDIVIUAL
            individual.dominated_solutions = [] #S_p: the set of solutions which the INDIVIDUAL DOMINATES
            for other_individual in population_inst: #for each q in P
                #print len(other_individual.fitness)
                if individual.dominates(other_individual):
                    individual.dominated_solutions.append(other_individual)
                elif other_individual.dominates(individual):
                    individual.domination_count += 1

            if individual.domination_count == 0:
                individual.rank = 0
                population_inst.fronts[0].append(individual)

        i = 0
        while len(population_inst.fronts[i]) > 0:
            temp = []
            for individual in population_inst.fronts[i]:
                for other_individual in individual.dominated_solutions:
                    other_individual.domination_count -= 1
                    if other_individual.domination_count == 0:
                        other_individual.rank = i + 1
                        temp.append(other_individual)

            i = i + 1
            population_inst.fronts.append(temp)

        return population_inst


class pop_member(object):

    def __init__(self, num_variables, variable_bounds, features=[], fitness=1000000000000000.0 * np.ones((1, 1)), changeCount=0, stallCount=0, Evaluated=0):
        self._num_variables = num_variables
        self._variable_bounds = variable_bounds
        self.features = features
        self.rank = None
        self.crowding_distance = None
        self.domination_count = None
        self.dominated_solutions = None
        self.objectives = None
        self.Evaluated = Evaluated
        self.variables = features
        self.fitness = fitness
        self.changeCount = changeCount
        self.stallCount = stallCount
        return

    def __str__(self):
        """Returns a string representation of the particular solution."""
        return self._values

    def __eq__(self, other):
        if isinstance(self, other.__class__):
            return self.features == other.features
        return False

    def dominates(self, other_individual, maxYesNO = True):
        #will need to change this function if I move to minimizing fitness
        #set maxYesNO to Flase if you want to test a version that minimizes fitness
        and_condition = True
        or_condition = False
        num_objectives = len(self.fitness)
        first_fitness_set = np.zeros((num_objectives, 1))
        Second_fitness_set = np.zeros((num_objectives, 1))
        #maxYesNO = False
        for p in range(0, num_objectives):
            first_fitness_set[p] = self.fitness[p]
            #print np.shape(other_individual.fitness)
            Second_fitness_set[p] = other_individual.fitness[p]
        if maxYesNO: #if you want to maximize fitness
            #appears to trigger based on fitness minimization
            for first, second in zip(self.fitness, other_individual.fitness):
                and_condition = and_condition and first <= second
                or_condition = or_condition or first < second
            return and_condition and or_condition
        else: #if you want to minimize fittness
            #definition of dominance:
            #x1 dominates x2 if and only if:
            #a) x1 is no worse than x2 in all objectives
            #and b) x1 is strictly better than x2 in at least one objective
            #x1 is self
            num_worse = 0
            num_better = 0
            for k in range(0, num_objectives):
                #check if for objective k, x1 is no worse than x2
                if first_fitness_set[k] > Second_fitness_set[k]:
                    num_worse = num_worse +1
                    return False #if you find an objective for which x is strictly worse than x1 you can say it dosen't dominate
                #check if for objective k, x1 is strictly better than x2 for objective k
                if first_fitness_set[k] < Second_fitness_set[k]:
                    num_better = num_better +1
            #see if the condition for dominance is met
            if (num_better >= 1) and (num_worse == 0):
                return True
            else:
                return False

class Population_class:

    def __init__(self):
        self.population = []
        self.fronts = [[]]

    def __len__(self):
        return len(self.population)

    def __iter__(self):
        return self.population.__iter__()

    def extend(self, new_individuals):
        self.population.extend(new_individuals)

    def append(self, new_individual):
        self.population.append(new_individual)


class ObjectiveFunction(object):

    def __init__(self, input_vector, func):
        """this constructor is used to define the inputs to an objective function.

        input_vector is a 1 by num_variables (where num_variables includes every element in each solution)
        numpy array containing zeros coresponding to 
        non-inputs and 1 coreponding to inputs. The order of the zeros and ones is determined
        by the form the solutions are defined in.
        
        func is the basic function that takes in used_num_inputs inputs in the order they
        appear in input_vector (and solution object) from left to right"""
        self._input_vector = input_vector
        self._func = func
        self.used_num_inputs = np.sum(input_vector)

    def evaluate(solution_input):
        """the basic use of the Objective Function Class.
        The output is the fittness.

        solutino_input is an instance of a solution object that is to be evaluated using the objective function"""
        num_vars = solution_input._num_variables
        var_vec = np.zeros(self.used_num_inputs)
        counter = 0
        for n in range(num_vars):
            if self._input_vector[n] == 1:
                var_vec[counter] = solution_input._values[counter]
                counter = counter + 1

        value = self._func(*var_vec)
        return value

def getParentsFromPopulation(Population_input):
    #a utility function that converts from the Population_class (INPUT) to
    #the (gnowee) Parent Class (OUTPUT)
    num_pop = len(Population_input.population)
    Parents_out = []
    for i in range(0, num_pop):
        Parents_out.append(Population_input.population[i])
    return Parents_out

