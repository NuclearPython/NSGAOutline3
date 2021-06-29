# Gnowee Modules
import Gnowee
from ObjectiveFunction import ObjectiveFunction
from GnoweeUtilities import ProblemParameters
from GnoweeHeuristics import GnoweeHeuristics

# Select optimization problem type and associated parameters
gh = GnoweeHeuristics()
gh.set_preset_params('spring')
print(gh)

# Run optimization
(timeline) = Gnowee.main(gh)
print(('\nThe result:\n', timeline[-1]))