# PacmanREINFORCE
#### Pacman with Policy Gradient - REINFORCE Algorithm
To run the algorithm: (from reinforcement.docx -- can use the same command lines)
1. `python pacman.py -p PolicyGradientAgents -x 2000 -n 2010 -l smallGrid`
2. `python pacman.py -p PolicyGradientAgents -a extractor=SimpleExtractor -x 50 -n 60 -l mediumGrid`
3. `python pacman.py -p PolicyGradientAgents -a extractor=SimpleExtractor -x 50 -n 60 -l mediumClassic`

#### Vary alpha and gamma as follows:
`python pacman.py -p PolicyGradientAgents -a gamma=0.1,alpha=0.4`

### Updated code to using score function with numpy gradient
1. Utilizes `scoreFnUsingGradients` function. 
2. Instead of linearly approximating the gradient as per barton sutton -- uses numpy's function for calculating gradient for a list using central differences. 
3. Works best for SimpleExtractor 
`python pacman.py -p PolicyGradientAgents -x 2000 -n 2010 -l smallGrid  -a extractor=SimpleExtractor,alpha=0.8`

### Updated code to use alternative formula for score function
1. Utilizes `scoreFnGradDiv` function
2. Divides the gradient of policies by policies. 

### Feature Extractor Update 
1. FullExtractor feature extractor considers features involving power pellets and inactive ghosts. 
2. To run the extractor, select extractor=FullExtractor in the arguments. An example is given below 

`python pacman.py -p PolicyGradientAgents -x 2000 -n 2010 -l smallGrid -a extractor=FullExtractor,gamma=0.8,alpha=0.5`
