# Pontryagin-Differentiable-Programming

The **_Pontryagin-Differentiable-Programming (PDP)_** project establishes a unified 
end-to-end framework to solve a broad class of learning and control tasks. Please find out more details in

* Our NeurIPS 2020 Paper: https://proceedings.neurips.cc/paper/2020/hash/5a7b238ba0f6502e5d6be14424b20ded-Abstract.html for theory and technical details.
* Blog section [[https://wanxinjin.github.io/Robots/#PDP](https://wanxinjin.github.io/Pontryagin-Differentiable-Programming/)](https://wanxinjin.github.io/Pontryagin-Differentiable-Programming/) for the simulated robot demos solved by PDP.




Additional Note (updated on June 2022):
  - **_Continuous Pontryagin Differentiable Programming (Continuous PDP)_**  can be found in our arxiv paper: https://arxiv.org/pdf/2008.02159.pdf (Accepted by IEEE T-RO)
  - **_Safe Pontryagin Differentiable Programming (Safe PDP)_**  can be found in our NeurIPS 2021 paper: https://papers.nips.cc/paper/2021/hash/85ea6fd7a2ca3960d0cf5201933ac998-Abstract.html 


## 1. Project Overview

The current version of the PDP project consists of three folders:

* __PDP__: an independent package implementing PDP cores. It  contains a core module called _PDP.py_, where four classes 
are defined and each includes certain functionalities, as described below.
    - ___OCSys___:  an interface for symbolically specifying a parameterized optimal control system;  a build-in OC solver to solve the optimal control problem; differentiating  Pontryagin's maximum/minimum principle; and obtaining the auxiliary control system.
    - ___LQR___: an interface for symbolically specifying an time-varying/time-invariant LQR system;  a build-in LQR solver to solve the LQR problem.
    - ___ControlPlanning___: an interface for symbolically specifying an optimal control system with  parameterized  policy (polynomials or neural networks);  integrating the  controlled system to obtain its trajectory;
    differentiating  Pontryagin's maximum/minimum principle; and  obtaining the auxiliary control system. 
    - ___SysID___: an interface for symbolically specifying a parameterized dynamics equation; integrating the dynamics (difference) equation to obtain its trajectory;  differentiating the  Pontryagin's maximum/minimum principle; and  obtaining the auxiliary control system. 
    
  Note: each class can be used independently, for example, you can  use only _**OCSys**_ to solve your own optimal control problem. 
  Each of the above classes is easy to approach  and 
  you can immediately tell the utility of different  methods within by looking at its name. All important lines are commented in great details.
  
* __JinEnv__: an independent package that provides  environments/visualizations of some typical physical systems for you to run your algorithms on. The JinEnv includes environments from simple (e.g., single inverted pendulum) to complex one (e.g., 6-DoF rocket powered landing). 
  These environments can be used for you to test your performance of your learning/control methods. The dynamics and control objective functions of these physical systems are off-the-shelf by default, but also allow you to customize them using the user-friendly interfaces.
  Each  environment is defined as an independent class:
 
    - __Single Pendulum__: a pendulum environment.
    - __Robot Arm__: a robot arm environment.
    - __Cart Pole__: a cart pole environment.
    - __Quadrotor Maneuvering__: a 6-DoF quadrotor UAV maneuvering environment. 
    - __Rocket Powered Landing__: a 6-Dof Rocket powered landing environment.
    
  For each environment, you can freely customize its dynamics parameters and control cost function. Each environment is independent. Each environment has visualization methods for you to showcase your results.

 
* __Examples__: including various examples of using PDP to solve different learning or control tasks, including inverse reinforcement learning, optimal control or model-based reinforcement learning, and 
system identification. The examples are classified based on the problems:
    - Examples/IRL/: examples of using PDP to solve IRL/IOC problems (i.e., IRL/IOC Mode of PDP framework).
    - Examples/OC/: examples of using PDP to solve Control/Planning problems (i.e., Control/Planning Mode of PDP framework).
    - Examples/SysID/: examples of using PDP to solve SysID problems (i.e., SysID Mode of PDP framework).

Each learning or control task is tested in different environments: inverted pendulum, robot arm,  cart-pole, quadrotor maneuvering, and rocket powered landing.

You can directly run each script.py under  __Examples__ folder.
    
 
## 2. Dependency  Packages
Please make sure that the following packages have  already been installed before use of the PDP package or JinEnv Package.
* CasADi: version > 3.5.1. Info: https://web.casadi.org/
* Numpy: version >  1.18.1. Info: https://numpy.org/
* Matplotlib: version > 3.2.1. Info: https://matplotlib.org/
* Python: version >= 3.7. Info: https://www.python.org/downloads/

Note: before you try the PDP and JinEnv Packages, we strongly recommend you to familiarize yourself with the 
CasADi  programming language, e.g., how to define a symbolic expression/function. Reading through Sections 2, 3, 4 on the page  https://web.casadi.org/docs/ is enough (around 30 mins)! 
Because this really helps you to debug your codes 
when you  test your own system using the PDP package here. We also recommend you to read through the PDP paper: https://arxiv.org/abs/1912.12970 
because all of the notations/steps in the codes are strictly following the paper.

The codes have been tested and run smoothly with Python 3.7. on MacOS (10.15.7) machine.

## 3. How to Use the PDP Package

First of all, you need to be relaxed: we have optimized the interface in the PDP package and JinEnv Package, which hopefully minimizes your effort 
on understanding and using them. All methods and variables within are pretty straightforward and carefully commented! In most of cases,  all you need to 
do is to specify the symbolic expressions of your control system: its dynamics, policy, or control cost function, then
PDP will take care of the rest.

The quickest way to get a big picture of the codes is to examine and run each example:

* Read and run Examples/IRL/pendulum/pendulum_PDP.py --- you will  understand how to use PDP to solve IRL/IOC problems.
* Read and run  Examples/OC/pendulum/pendulum_PDP.py --- you will  understand how to use PDP to solve model-based optimal control problems.
* Read and run  Examples/SysID/pendulum/pendulum_PDP.py --- you will  understand how to use PDP to solve system identification problems.

###  PDP for Solving Inverse Reinforcement Learning Tasks (IRL/IOC Mode)

To solve IRL/IOC problems,  you will mainly need the following two classes from   ./PDP/PDP.py module:
* _**OCSys**_: which is to solve the optimal control system in  forward pass and then construct the auxiliary control system in  backward pass. 
The procedure to instantiate an OCSys object is fairly straightforward, including nine steps:
    - Step 1: set state variable of your system ----> setStateVariable
    - Step 2: set control variable of your system ----> setControlVariable
    - Step 3: set (unknown) parameters in the dynamics and cost function (if applicable) ----> setAuxvarVariable; otherwise you can ignore this step.
    - Step 4: set dynamics (difference) equation of your system----> setDyn
    - Step 5: set path cost function of your system ----> setPathCost
    - Step 6: set final cost function of your system -----> setFinalCost
    - Step 7: solve the optimal trajectory from your  optimal control system -----> ocSolver
    - Step 8: differentiate the Pontryagin's maximum principle (if you have Step 3) -----> diffPMP
    - Step 9: get the auxiliary control system (if have Step 3) ------> getAuxSys

    Note: if you are only using OCSys to solve your optimal control problem (not for IOC/IRL), you can ignore Steps 3, 8, and 9, and also 
     ignore the use of LQR (the next class) to solve your auxiliary control system.

* _**LQR**_ : which is to solve the auxiliary control system in  backward pass and obtain the analytical derivative of the forward-pass trajectory with respect to the parameters within the dynamics and control cost function.
The procedure to instantiate an LQR object is fairly straightforward, including four steps:
    - Step 1: set dynamics equation ----> setDyn, a time-varying dynamics needs you to specify the sequence of dynF/dynG/dynE
    - Step 2: set path cost function ----> setPathCost,  a time-varying path cost needs you to specify the sequence of Hxx/Huu...
    - Step 3: set path final function ----> setFinalCost, 
    - Step 4: solve LQR problem -----> ocSolver

Examples for IRL/IOC tasks: check and run all examples under  ./Examples/IRL/ folder.

### PDP for Solving Optimal Control or Planning Tasks (Control/Planning Mode)
To solve optimal control or planning problems,  you only need _**ControlPlanning**_ class from ./PDP/PDP.py module:

* _**ControlPlanning**_.
The procedure to instantiate a ControlPlanning object is fairly straightforward, including the following nine steps:
    * Step 1: set state variable of your system ----> setStateVariable
    * Step 2: set input variable of your system ----> setControlVariable
    * Step 3: set dynamics (difference) equation of  your system ----> setDyn
    * Step 4: set path cost function of  your system ----> setPathCost
    * Step 5: set final cost function of  your system -----> setFinalCost
    * Step 6: set policy parameterization with (unknown) parameters ----->  for planning, you can use setPolyControl (parameterize the policy as Lagrangian polynomial), or for feedback control, you can use setNeuralPolicy (parameterize the policy as feedback controller)
    * Step 7: integrate the control system in forward pass -----> integrateSys
    * Step 8: get the auxiliary control system ------> getAuxSys
    * Step 9: integrate the auxiliary control system in backward pass ------> integrateAuxSys

* The user can also choose one of the following added features to improve the performance of PDP:
    * Including warping techniques: please use the methods beginning with 'warped_'.   The idea is to map the time axis 
    of the original control/planning problem into a shorter time horizon;  after solve it, then map it back to original time axis.
    Advantage of this is that it will make PDP more robust and not easy to get trapped in local minima.
    * Including the recovery matrix techniques: please see all the methods beginning with 'recmat_'. 
    The idea is to parameterize the policy as Lagrange polynomial with the pivot points being all trajectory points.
    The advantage of using the recovery matrix is that PDP Control/Planning Mode is more faster, because we can use Recovery matrix to 
     solve the gradient in a one-time fashion. Fore more information of the recovery matrix technique, please refer to my previous paper 
     https://arxiv.org/abs/1803.07696 (conditionally accepted by IJRR).

Examples for control or planning tasks: check and run all examples under  ./Examples/OC/ folder.

### PDP for Solving System Identification Tasks (SysID Mode)
To solve system identification problems,  you will need _**SysID**_ class from the module ./PDP/PDP.py:

* _**SysID**_: which is to integrate the  controlled (autonomous) system in  forward pass,  obtain the corresponding auxiliary control system,
and then integrate the auxiliary control system in  backward pass.
The procedure to instantiate a SysID object is fairly straightforward, including seven steps:
    - Step 1: set state variable of your dynamics ----> setStateVariable
    - Step 2: set input variable of your dynamics ----> setControlVariable
    - Step 3: set (unknown) parameters in dynamics----> setAuxvarVariable
    - Step 4: set dynamics (difference) equation----> setDyn
    * Step 5: integrate the dynamics equation in forward pass -----> integrateDyn
    * Step 6: get the auxiliary control system ------> getAuxSys
    * Step 7: integrate the auxiliary control system in backward pass ------> integrateAuxSys

Examples for system identification tasks: check and run all examples under  ./Examples/SysID/ folder.

## 4. How to Use the JinEnv Package
Each environment is defined as a class, which contains the following methods:
* _**initDyn**_: which is used to initialize the dynamics of a pysical system. The input arguments are  parameters (values) of the dynamics. 
You can pass a specific value to each parameter, otherwise, the parameter is None (by default) and will become a learnable variable in your dynamics. Some variables within the initDyn method are
    * X: the vector of state variables in the dynamics.
    * U: the vector of control variable in the dynamics.
    * f:  the symbolic expression of dynamics (differential) equation for the dynamics.
    * dyn_auxvar: the vector of  parameter variables in the dynamics (if all  parameters are assigned values during initialization, this vector will be empty).

* _**initCost**_: which is used to initialize the control cost function of a pysical system. The cost function by default is a weighed distance to the goal state plus a control effort term, and  the input arguments to initCost are the weights. 
You can pass a specific value to each weight, otherwise the weight is   None (by default) and will be a learnable variable in your cost function. Some attributes for the initCost method are
    * final_cost: the symbolic expression of the final cost function.
    * path_cost: the symbolic expression of the path cost function.
    * cost_auxvar: the vector of weight variables (if all weights are assigned values during initialization, vector will be empty).

* _**play_animation**_: which is used to visualize the motion of the pysical system. The input is the state (control) trajectory.

Examples for using each of the environments: check and run all examples under  ./Examples/ folder.
   
 


## 5. Information and Citation
If you have encountered a bug in your implementation of the code, please feel free to let me know.
* name: wanxin jin (he/his)
* email: wanxinjin@gmail.com

If you also want the codes of other methods, e.g., inverse KKT, iterative LQR, or GPS, policy imitations, which are compared in our paper (https://arxiv.org/abs/1912.12970). 
Please also let me know. \
Currently, I am working on developing a general control tool box in Python, which includes all these popular methods (may publish also in near future).



If you find this project helpful in your publications, please consider citing our paper (accepted by NeurIPS, 2020).

    @article{jin2020pontryagin,
      title={Pontryagin differentiable programming: An end-to-end learning and control framework},
      author={Jin, Wanxin and Wang, Zhaoran and Yang, Zhuoran and Mou, Shaoshuai},
      journal={Advances in Neural Information Processing Systems},
      volume={33},
      pages={7979--7992},
      year={2020}
    }


