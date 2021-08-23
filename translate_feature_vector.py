#IMPORTANT!!!
#THIS FILE SHOULD NOT BE OVERRIDDEN BY FILES OF THE SAME NAME
# the user is meant to supply a file called "translate_feature_vector.py"
# which contains a method called translate_feature_vector
#this method (and thus this file) will differ based on the application 
#there are a couple of requirements for this method:
#1) it should take in the feature vector from Gnowee
#2) it should return 2 lists
#the first being a list of strings that appear in the template MCNP file and which are to be replaced
# the second being a list of the values that will be placed where the strings appear 
#the order of the two lists should match

#this particular file is for the upcoming paper
import numpy as np
def translate_feature_vector(feature_vector):
    #the feature vecotr in this case is the base perturbation
    # of the radius
    string_set = []
    for i in range(0,150):
        string_set.append("mat_"+str(i+1))
    names = string_set
    variableValues = feature_vector
    valueString = []
    for i in range(0,150):
        valueString.append(str(int(round(variableValues[i]))))

    #mate sure there is fissionable material otherwise MCNP may have problems
    #INSERT CONTION TO TELL IF THIS MEANS THERE IS NO FISSIONABLE MATERIAL WHERE THERE NEEDS TO BE
    BadVector = False
    if BadVector == True:
        raise ValueError('the feature vector: ' + str(feature_vector) + ' is a bad feature vector')
    return names, valueString
