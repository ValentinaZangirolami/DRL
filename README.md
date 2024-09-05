# Dealing with uncertainty: balancing exploration and exploitation in deep recurrent reinforcement learning

## **Description**

This repo contains an implementation of Double Dueling Deep Recurrent Q-Network which can be enhanced with several exploration strategies, like deterministic epsilon-greedy, adaptive epsilon-greedy (VDBE and BMC) [1], softmax, max-boltzmann exploration and VDBE-softmax, and an error masking strategy [2], [4]. 

### **Code Structure:**
* <code>./AirsimEnv/</code>: folder where the two environments ( <code>AirsimEnv.py</code> and <code>AirsimEnv_9actions.py</code> ) are stored; the former includes five steering angles and the latter nine steering angles. Further, this folder contains:
  * <code>DRQN_classes.py</code>: implementation of agent, experience replay, exploration strategies, neural network and connection with AirSim NH are defined
  * <code>bayesian.py</code>: a support for BMC epsilon-greedy
  * <code>final_reward_points.csv</code>: a support for reward calculation (required for env scripts)
* <code>DRQN_airsim_training.py</code>: contains training loop in which all files in the previous points are required (main script for training process)
* <code>DRQN_evaluation.py</code>: contains training and test evaluation; each subset is defined with a different set of starting points to evaluate the model performance

## **Prerequisites**
  * Python 3.7.6 
  * Tensorflow 2.5.0
  * Tornado 4.5.3
  * OpenCV 4.5.2.54
  * OpenAI Gym 0.18.3
  * Airsim 1.5.0
  
## **Hardware**
  * 2 GPU Tesla M60 with 8 Gb
 
## **References**
[1] Gimelfarb, M., S. Sanner, and C.-G. Lee, 2020: *ε-BMC: A Bayesian Ensemble Approach to Epsilon-Greedy Exploration in Model-Free Reinforcement Learning*. CoRR 

[2] Juliani A., 2016: *Simple Reinforcement Learning with Tensorflow Part 6: Partial Observability and Deep Recurrent Q-Networks*. URL: https://github.com/awjuliani/DeepRL-Agents

[3] Riboni, A., A. Candelieri, and M. Borrotti, 2021: *Deep Autonomous Agents comparison for Self-Driving Cars*. Proceedings of The 7th International Conference on Machine Learning, Optimization and Big Data - LOD 
  
[4] *Welcome to AirSim*, https://microsoft.github.io/AirSim/
 
## **How to cite**

Zangirolami, V. and M. Borrotti, 2024: *Dealing with uncertainty: balancing exploration and exploitation in deep recurrent reinforcement learning*. In: Knowledge-Based Systems 293. [Paper](https://doi.org/10.1016/j.knosys.2024.111663)

## **Acknowledgements**
I acknowledge Data Science Lab of Department of Economics, Management and Statistics (DEMS) of University of Milan-Bicocca for providing a virtual machine.

## **DEMO**
<video src="https://user-images.githubusercontent.com/78240304/149147549-29936bd7-f629-4b66-a125-ddcd50443bcb.mp4">.


