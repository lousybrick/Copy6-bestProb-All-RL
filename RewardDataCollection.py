
# IMPORTANT: Override the final() function of PolicyGradientAgents for example to add this to collect reward data:
# inside the if statement ' if self.episodesSoFar <= self.numTraining: '

#     file = open("PolicyGradientData.txt", 'a')
#     file.write('%d,%f,%f,%f,%f,SmallGrid' % (self.episodesSoFar,trainAvg,self.discount,self.alpha,self.epsilon))
#     file.write('\n')
#     file.close()

# and set NUM_EPS_UPDATE = 1 if you want it recored every episode
# I am not sure yet how to access the layout, so it's smallGrid for now


import os
import csv
import numpy as np
import pandas as pd

# file = open("QagentData.txt",'w')
# file.write('episodes,avgTrainingReward,Gamma,Alpha,Epsilon,Layout')
# file.write('\n')
# file.close()
#
# for alpha in np.arange(0.25,0.55,0.05):
#     for epsilon in np.arange(0.005,0.02,0.005):
#         for gamma in np.arange(0.65,1,0.05):
#             print("\n alpha = %f, epsilon = %f, gamma = %f \n" % (alpha,epsilon,gamma))
#             os.system("python pacman.py -p PacmanQAgent -x 1001 -n 1002 -l smallGrid -q -f --fixRandomSeed -a \
#             epsilon=%f,alpha=%f,gamma=%f" % (epsilon,alpha,gamma))

#
# file = open("ApproximateIdentityAgentData.txt", 'w')
# file.write('episodes,avgTrainingReward,Gamma,Alpha,Epsilon,Layout')
# file.write('\n')
# file.close()
#
# for alpha in np.arange(0.05,0.5,0.05):
#     for epsilon in np.arange(0.01,0.07,0.01):
#         for gamma in np.arange(0.7,0.95,0.05):
#             print("\n alpha = %f, epsilon = %f, gamma = %f \n" % (alpha,epsilon,gamma))
#             os.system("python pacman.py -p ApproximateQAgent -x 1001 -n 1002 -l smallGrid -q -f --fixRandomSeed -a \
#             extractor=IdentityExtractor,epsilon=%f,alpha=%f,gamma=%f" % (epsilon,alpha,gamma))
#

# default hyperparams according to project 4: epsilon=0.05,gamma=0.8,alpha=0.2

file = open("PolicyGradientData.txt",'w')
file.write('episodes,avgTrainingReward,Gamma,Alpha,Epsilon,Layout')
file.write('\n')
file.close()


os.system("python pacman.py -p PolicyGradientAgents -x 2001 -n 2002 -l smallGrid -q -f --fixRandomSeed\
 -a epsilon=0.1,alpha=0.01,gamma=0.9")
