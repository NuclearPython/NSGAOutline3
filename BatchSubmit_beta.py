#this file is for developing the functions that will perform batch
#job submission on the cluster

#the user

import os
import numpy as np
import copy
import time
from subprocess import Popen
import os
import fileinput
import shutil
from functions import program_gen
from translate_feature_vector import translate_feature_vector

evaluationNumber = 0
feature_vector_dictionary = {}
unique_evals = 0

mcnp_script_template= """#!/bin/bash
#PBS -V
#PBS -q fill
#PBS -l nodes=1:ppn=8

hostname
module load MCNP6

RTP="/tmp/runtp--".`date "+%R%N"`
cd $PBS_O_WORKDIR
mcnp6 TASKS 8 name=%%%INPUT%%% runtpe=$RTP
grep -a "final result" %%%INPUT%%%o > %%%INPUT%%%_done.dat
rm $RTP"""

def build_mcnp_running_script(input_file_name):
    #modified by Alex
    if input_file_name.endswith('.inp') == False:
        input_file_name = input_file_name + ".inp"

    write_string = mcnp_script_template.replace("%%%INPUT%%%", input_file_name)
    script_file = open(input_file_name + ".sh", 'w')
    script_file.write(write_string)
    script_file.close()

def wait_on_jobs(jobs_list):
    #modified by Alex
    waiting_bool = True

    actively_waiting_list = copy.deepcopy(jobs_list)
    while waiting_bool:
        job_count = 0
        for file in os.listdir():
            if "_done.dat" in file:
                for job_file in jobs_list:
                    if job_file in file:
                        job_count += 1
                        actively_waiting_list.remove(job_file)

        if job_count == len(jobs_list):
            waiting_bool = False
            print("All jobs have run, continuing")
            return
        print('Jobs complete:', job_count, 'of', len(jobs_list))
        for job_ in actively_waiting_list:
            print(job_)
        print('Waiting 15 seconds')
        #os.wait(15)
        time.sleep(15)

def write_mcnp_input_simple(template_file_str, dictionary_of_replacements, input_file_str):
    #MODIFIED BY ALEX TO REMOVE DEPENDENCE ON VARIABLES CASSETTE FEATURES
    print("Building MCNP model from template")

    ### Opening template file
    template_file = open(template_file_str, 'r')

    ### Opening new file
    output_file = open(input_file_str , 'w')

    #the following 4 lines have been commented out by Alex
    #### If there are 0 plates in the variable cassette then another MCNP input template is used
    #if dictionary_of_replacements["cassette_pattern_2A_fill_value"] == 'fill=0:-1':
    #    ### Opening template file
    #    template_file = open(options["variable_cassette_2A_unique_template_files"][template_file_str], 'r')

    ### Moving over the template file, creating list of new strings from template strings,
    ### updating where necessary according to the replacement dictionary
    lines_list = []
    for line in template_file:
        for key in dictionary_of_replacements:
            key = str(key)
            for split_ in line.split():
                if key == split_:
                    if key == "cassette_pattern_2A_fill_value":
                        print("cassette_pattern_2A_fill_value", key, dictionary_of_replacements[key])
                    line = line.replace(key, str(dictionary_of_replacements[key]))

        # Add to the list of lines to be written
        lines_list.append(line)
    # Write out input file
    output_file.writelines(lines_list)

    template_file.close()
    output_file.close()

def Batch_Submit(list_of_feature_vectors, templateFile, prefix = "MCNP_File_Num"):
    #assumes that all of the feature vectors should be run (easy way)
    global evaluationNumber
    num_vectors = len(list_of_feature_vectors)
    outputFiles = [] #the order of this list should match the order of the feature vectors submitted to the function
    hard_way = False
    if hard_way == True:
        global evaluationNumber
        global feature_vector_dictionary 
        global unique_evals
        num_vectors = len(list_of_feature_vectors)
        #check if each feature vector has already been run
        list_to_be_run = []
        for i in range(0,num_vectors):
            Specific_vector = list_of_feature_vectors[i]
            string_version = ' '.join(map(str, Specific_vector)) #creates a string version of a feature vector
            #so that it can be looked for in the global dictionary
            if string_ID in feature_vector_dictionary:
                newVector = False
                print("Old Vector on Eval number:" +str(evaluationNumber))
            else:
                newVector = True
                print(str(evaluationNumber) + " run's feature vector is: "+ str(Specific_vector))
                if evaluationNumber == 1:
                    unique_evals = 1
                    #feature_vector_library = np.array([evaluationNumber, material_Vector, keff])
                else:
                    unique_evals = unique_evals +1
    #translate the feature vectors into runnalble MCNP files
    execution_scripts = []
    run_list = []
    ID_list = []
    for i in range(0,num_vectors):
        feature_vector = list_of_feature_vectors[i]
        try:
            #create dictionary
            variableNames, variableValues = translate_feature_vector(feature_vector) #this function should be supplied by the user
            dict = {} #erases the dictionary for each feature vector
            num_variables = len(variableNames)
            for r in range(0,num_variables):
                dict[variableNames[r]] = variableValues[r] 
            BadVector = False
        except ValueError:
            BadVector = True

        if BadVector == False:
            mainfolder = os.getcwd()
            #specify files
            MockMCNPfilename = templateFile
            evalNumber = evaluationNumber + i #translate the global variable into the unique ID
            proposedFilename = prefix+str(evalNumber)+'.inp'
            #generate the MCNP file
            write_mcnp_input_simple(templateFile, dict, proposedFilename)
            #create execution scripts for each feature vector
            build_mcnp_running_script(proposedFilename)
            execution_scripts.append(proposedFilename + ".sh")
            ID_list.append(str(evalNumber))
            #help report the names of the output files
            outFileName = proposedFilename +'o'
            outputFiles.append(outFileName)
            
    #rapidly submit jobs to cluster by running the execution scripts
    for i in range(0,num_vectors):
        os.system('qsub ' + execution_scripts[i])
    #wait for files to finish running
    print("Jobs to wait for:")
    for i in range (0,num_vectors):
        print(ID_list[i])

    wait_on_jobs(ID_list)
    #return true, calculating the actual fittness is left to the user supplied objective functions
    return outputFiles
