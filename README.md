# Pontryagin-Differentiable-Programming

The **_Pontryagin-Differentiable-Programming (PDP)_** project establishes a unified 
end-to-end framework to solve a broad class of learning and control tasks. Please find out more details in

* Paper: https://arxiv.org/abs/1912.12970 for theory and technical details (accepted by NeurIPS 2020).
* Demos: https://wanxinjin.github.io/posts/pdp for some video demos.



## 1. Project Overview

The current version of the PDP project consists of three folders:

* __PDP__: an independent package implementing PDP cores. It  contains a core module called _PDP.py_, where four classes 
are defined with each accounting for certain functionalities:
    - ___OCSys___: specify a parameterized optimal control system;  an build-in OC solver to solve the optimal control problem; differentiate  Pontryagin's maximum principle and obtain the auxiliary control system.
    - ___LQR___: specify an time-varying/time-invariant LQR system;  an build-in LQR solver to solve the LQR problem.
    - ___ControlPlanning___: specify an optimal control system with  parameterized  policy;  integrate the  control system to obtain its trajectory;
    differentiate  Pontryagin's maximum principle and  obtain the auxiliary control system. 
    - ___SysID___:  specify a parameterized dynamics equation; integrate the dynamics equation to obtain its trajectory;  differentiate  Pontryagin's maximum principle and and  obtain the auxiliary control system. 
    
  Note: each class can be used independently, for example, you can  use only _**OCSys**_ to solve your own optimal control problem. 
  Each of the above class is easy to approach  and 
  you can immediately tell the utility of different  methods by looking at its name. All important lines are commented in details.
  
* __JinEnv__: an independent package that provides  environments/visualizations of some typical physical systems from simple (e.g. single inverted pendulum) to complex ones (e.g., 6-DoF rocket powered landing). 
  These environments can be used for you to test your performance of your learning/control methods. The dynamics and control cost functions of these physical systems are off-the-shelf, while also allow you to customize  according to your own purpose.
  Each physical system is defined by a independent class:
 
    - __SinglePendulum__: a pendulum environment.
    - __RobotArm__: a robot arm environment.
    - __CartPole__: a cart pole environment.
    - __Quadrotor__: a 6-DoF quadrotor UAV maneuvering environment. 
    - __Rocket__: a 6-Dof Rocket landing environment.
    
  For each system, you can freely customize its dynamics parameters and control cost function. Each environment is independent. Each environment has visualization method for you to showcase your results.

 
* __Examples__: including various examples of using PDP to solve three fundamental problems: inverse reinforcement learning, optimal control, and 
system identification in various environments. The examples are classified based on the problems:
    - Examples/IRL/: examples of using PDP to solve IRL/IOC problems (i.e., IRL/IOC Mode of PDP framework).
    - Examples/OC/: examples of using PDP to solve Control/Planning problems (i.e., Control/Planning Mode of PDP framework).
    - Examples/SysID/: examples of using PDP to solve SysID problems (i.e., SysID Mode of PDP framework).
    

    
 
## 2. Dependency  Packages
Please make sure that the following packages have  already been installed before use of the PDP package or JinEnv Package.
* CasADi: version > 3.5.1. Info: https://web.casadi.org/
* Numpy: version >  1.18.1. Info: https://numpy.org/
* Matplotlib: version > 3.2.1. Info: https://matplotlib.org/

Note: before you hand on the PDP and JinEnv Packages, we strongly recommend you to familiarize yourself with the 
CasADi  programming language, e.g., how to define a symbolic expression/function. Reading through Sections 2, 3, 4 on the page  https://web.casadi.org/docs/ is enough (around 1 hour)! 
Because this really help you to debug your codes 
when you are test your own system based on  PDP codes. We also recommend you to read through the PDP paper: https://arxiv.org/abs/1912.12970 
because all of the notations/steps in the codes are strictly following the paper (Appendix D).

The codes have been tested on Python 3.7.

## 3. How to Use the PDP Package

First of all, you need to be relaxed: we have optimized the interface of the PDP package, which will minimize your effort 
on understanding and use of PDP and JinEnv Packages. All codes be pretty straightforward and easy! In most of cases,  all you need to 
do is to specify the symbolic expressions of your control system: its dynamics, policy, or control cost function, then
PDP will take care of the rest!

The quickest way to get a big picture of the codes is to examine and run each example:

* Read and run Examples/IRL/pendulum/pendulum_PDP.py --- you will  understand how to use PDP to solve IRL/IOC problems.
* Read and run  Examples/OC/pendulum/pendulum_PDP.py --- you will  understand how to use PDP to solve optimal control.
* Read and run  Examples/SysID/pendulum/pendulum_PDP.py --- you will  understand how to use PDP to solve system ID.

###  Solving Inverse Reinforcement Learning (IRL/IOC Mode)

To solve IRL/IOC problems,  you will need the following classes from the module ./PDP/PDP.py:
* _**OCSys**_: solve the optimal control system in the forward pass, and obtain the auxiliary control system. The procedure to instantiate an OCSys object is fairly straightforward (understand each method by looking at its name):
    - Step 1: set state variable ----> setStateVariable
    - Step 2: set control variable ----> setControlVariable
    - Step 3: set dynamics/cost function parameter (if applicable) ----> setAuxvarVariable; otherwise you can ignore this step.
    - Step 4: set dynamics equation----> setDyn
    - Step 5: set path cost function ----> setPathCost
    - Step 6: set final cost function -----> setFinalCost
    - Step 7: solve the optimal control problem -----> ocSolver
    - Step 8: differentiate the Pontryagin's maximum principle (if you have Step 3) -----> diffPMP
    - Step 9: get the auxiliary control system (if have Step 3) ------> getAuxSys

    Note: if you are only using OCSys to solve your optimal control problem, you can ignore Steps 3, 8, and 9, and also 
     ignore the use of LQR to solve your auxiliary control system.

* _**LQR**_ : solve the auxiliary control system in the backward pass, and obtain the gradient of trajectory with respect to the parameter.
The procedure to instantiate an LQR object is fairly straightforward (just understand each method by looking at its name):
    - Step 1: set dynamics equation ----> setDyn, a time-varying dynamics needs you to specify the sequence of dynF/dynG/dynE
    - Step 2: set path cost function ----> setPathCost,  a time-varying path cost needs you to specify the sequence of Hxx/Huu...
    - Step 3: set path final function ----> setFinalCost, 
    - Step 4: solve LQR problem -----> ocSolver

Examples: check and run all examples under  ./Examples/IRL/

### Solving Optimal Control or Planning Problems (Control/Planning Mode)
To solve optimal control or planning problems,  you will need the following classes from the module ./PDP/PDP.py:

* _**ControlPlanning**_.
The procedure to instantiate an ControlPlanning object is fairly straightforward (just understand each method by looking at its name):
    * Step 1: set state variable ----> setStateVariable
    * Step 2: set control variable ----> setControlVariable
    * Step 3: set dynamics equation----> setDyn
    * Step 4: set path cost function ----> setPathCost
    * Step 5: set final cost function -----> setFinalCost
    * Step 6: set policy parameterization ----->  for planning, use setPolyControl (parameterize the policy as Lagrangian polynomial);  for control, use setNeuralPolicy (parameterize the policy as feedback controller)
    * Step 7: integrate the control system in forward pass -----> integrateSys
    * Step 8: get the auxiliary control system ------> getAuxSys
    * Step 9: integrate the auxiliary control system in backward pass ------> integrateAuxSys

* The user can also choose one of the following added features to improve the performance of PDP:
    * Including warping techniques: please use the methods beginning with 'warped_'.   The idea is to map the time axis 
    of the original OC problem into a shorter horizon OC problem;  after solve it, then map it back to original time axis.
    Advantage of this is that it will make PDP more robust and not easy to trap in local optima.
    * Including the recovery matrix techniques: please see all the methods beginning with 'recmat_'. 
    The idea is to parameterize the policy as Lagrange polynomial with the pivot points are all input trajectory points.
    The advantage of using the recovery matrix is that PDP Control/Planning Mode is more faster, because we can use Recovery matrix (https://arxiv.org/abs/1803.07696) to 
    iteratively solve the gradient.

Examples: check and run all examples under  ./Examples/OC/

### Solving System Identification Problems (SysID Mode)
To solve system identification problems,  you will need the following classes from the module ./PDP/PDP.py:

* _**SysID**_: integrate the  controlled system in the forward pass,  obtain the corresponding auxiliary control system,
and integrate the auxiliary control system in the backward pass.
The procedure to instantiate an ControlPlanning object is fairly straightforward (just understand each method by looking at its name):
    - Step 1: set state variable ----> setStateVariable
    - Step 2: set control variable ----> setControlVariable
    - Step 3: set parameter in dynamics equation----> setAuxvarVariable; otherwise you can ignore this step.
    - Step 4: set dynamics equation----> setDyn
    * Step 5: integrate the dynamics equation in forward pass -----> integrateDyn
    * Step 6: get the auxiliary control system ------> getAuxSys
    * Step 7: integrate the auxiliary control system in backward pass ------> integrateAuxSys

Examples: check and run all examples under  ./Examples/SysID/

## 4. How to Use the JinEnv Package
Each environment is defined by a class, which contains the following methods:
* _**initDyn**_: this is used to initialize the dynamics of a pysical system, the input arguments are   parameters of the dynamics. 
You can pass a specific value to each parameter, otherwise, the parameter is None (by default) and will become a learnable variable in your dynamics. Some attributes for the initDyn method are
    * X: the vector of state variables.
    * U: the vector of control variable.
    * f:  the expression of dynamics equation  of the system.
    * dyn_auxvar: the vector of  parameter variables (if all dynamics parameters are assigned, this will be empty)

* _**initCost**_: this is used to initialize the control cost function of a pysical system, the cost function by default is a weighed distance to the goal state plus control penalty terms. The input arguments are the weights. 
You can pass a specific value to each weight, otherwise the weight is   None (by default) and will be a learnable variable in your cost function. Some attributes for the initCost method are
    * final_cost: the expression of the final cost function.
    * path_cost: the expression of the path cost function.
    * cost_auxvar: the vector of weight variables (if all weights are assigned, this will be empty)

* _**play_animation**_: this is used to visualize the motion of the pysical system. The input is the state (control) trajectory.

Examples: check and run all examples under  ./Examples/  
   
 


## 5. Contact Information and Citation
If you have encountered a bug in your implementation of the code, please feel free to report to me via email at the following address:
* name: wanxin jin (he/his)
* email: wanxinjin@gmail.com


If you find this project helpful in your publications, please consider citing our paper.

    @article{jin2019pontryagin,
      title={Pontryagin Differentiable Programming: An End-to-End Learning and Control Framework},
      author={Jin, Wanxin and Wang, Zhaoran and Yang, Zhuoran and Mou, Shaoshuai},
      journal={arXiv preprint arXiv:1912.12970},
      year={2019}
    }


