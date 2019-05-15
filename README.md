[//]: # (Image References)

[image1]: 

# Project 2: Continuous Control

### Introduction
 

![Trained Agent][image1]



### Getting Started

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here]()
    - Mac OSX: [click here]()
    - Windows (32-bit): [click here]()
    - Windows (64-bit): [click here]()
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

2. Place the file in the root folder of the repo and decompress it. 

## Installation

1. Download or clone this GitHub repository.

2. Download and install Anaconda Python 3.6 from the following link : https://www.anaconda.com/download/

2. Create a new conda environment named drlnd (or whatever name you prefer) and then activate it:

	- Linux or Mac:
	
		`conda create --name drlnd python=3.6`
	
		`source activate drnld`

	- Windows:
	
		`conda create --name drnld python=3.6`
	
		`activate drnld`

4. Install the required dependencies navigating to where you downloaded and saved this GitHub repository and then into the '.python/' subdirectory. Then run from the command line:
	
		`pip3 install .`
 
## Files

- agent.py: Contains the agent who interacts with the environment and is used to train the model. 
- actor.py: Contains the Neural Network implemented in Pytorch that is used to pick the actions. 
- critic.py: Contains the Neural Network implemented in Pytorch that is used to evaluate the actions chosen by the actor. 
- replay_buffer.py: Helper class to implement the Esperience Replay algorithm.
- OUNoise.py: Helper Class to implement the addition of some noise to the actions chosen by the agent 
- agent_training.py: Process that delivers the trained model. 
- agent_test.py: Execution of some episodes with the agent using the trained model. 

## Training

 - Go to the root folder of the repo and run:
 
 	`python agent_training.py`
	
 - When the score reaches the value +13 it will stop and save the model weights to the file .

## Testing

 - To test the trained agent:
 
 	`python agent_test.py`
	
