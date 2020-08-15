"""
# Module name: Pontryagin Differentiable Programming (PDP)
# Technical details can be found in the Arxiv Paper:
# Pontryagin Differentiable Programming: An End-to-End Learning and Control Framework
# https://arxiv.org/abs/1912.12970

# If you want to use this modules or part of it in your academic work, please cite our paper:
# @article{jin2019pontryagin,
#   title={Pontryagin Differentiable Programming: An End-to-End Learning and Control Framework},
#   author={Jin, Wanxin and Wang, Zhaoran and Yang, Zhuoran and Mou, Shaoshuai},
#   journal={arXiv preprint arXiv:1912.12970},
#   year={2019}
# }

# Do NOT distribute without written permission from Wanxin Jin
# Do NOT use it for any commercial purpose

# Contact email: wanxinjin@gmail.com
# Last update: May 02, 2020
# Last update: Aug 11, 2020 (add the neural policy parameterization in OC mode)
"""

from casadi import *
import numpy
from scipy import interpolate

'''
# =============================================================================================================
# The OCSys class has multiple functionaries: 1) define an optimal control system, 2) solve the optimal control
# system, and 3) obtain the auxiliary control system.

# The standard form of the dynamics of an optimal control system is
# x_k+1= f（x_k, u_k, auxvar)
# The standard form of the cost function of an optimal control system is
# J = sum_0^(T-1) path_cost + final_cost,
# where path_cost = c(x, u, auxvar) and final_cost= h(x, auxvar).
# Note that in the above standard optimal control system form, "auxvar" is the parameter (which can be learned)
# If you don't need the parameter, e.g.m you just want to use this class to solve an optimal control problem,
# instead of learning the parameter, you can ignore setting this augment in your code.

# The procedure to use this class is fairly straightforward, just understand each method by looking at its name:
# Step 1: set state variable ----> setStateVariable
# Step 2: set control variable ----> setControlVariable
# Step 3: set parameter (if applicable) ----> setAuxvarVariable; otherwise you can ignore this step.
# Step 4: set dynamics equation----> setDyn
# Step 5: set path cost function ----> setPathCost
# Step 6: set final cost function -----> setFinalCost
# Step 7: solve the optimal control problem -----> ocSolver
# Step 8: differentiate the Pontryagin's maximum principle (if you have Step 3) -----> diffPMP
# Step 9: get the auxiliary control system (if have Step 3) ------> getAuxSys

# Note that if you are not wanting to learn the parameter in an optimal control system, you can ignore Step 3. 8. 9.
# Note that most of the notations used here are consistent with the notations defined in the PDP paper.
'''


class OCSys:

    def __init__(self, project_name="my optimal control system"):
        self.project_name = project_name

    def setAuxvarVariable(self, auxvar=None):
        if auxvar is None or auxvar.numel() == 0:
            self.auxvar = SX.sym('auxvar')
        else:
            self.auxvar = auxvar
        self.n_auxvar = self.auxvar.numel()

    def setStateVariable(self, state, state_lb=[], state_ub=[]):
        self.state = state
        self.n_state = self.state.numel()
        if len(state_lb) == self.n_state:
            self.state_lb = state_lb
        else:
            self.state_lb = self.n_state * [-1e20]

        if len(state_ub) == self.n_state:
            self.state_ub = state_ub
        else:
            self.state_ub = self.n_state * [1e20]

    def setControlVariable(self, control, control_lb=[], control_ub=[]):
        self.control = control
        self.n_control = self.control.numel()

        if len(control_lb) == self.n_control:
            self.control_lb = control_lb
        else:
            self.control_lb = self.n_control * [-1e20]

        if len(control_ub) == self.n_control:
            self.control_ub = control_ub
        else:
            self.control_ub = self.n_control * [1e20]

    def setDyn(self, ode):
        if not hasattr(self, 'auxvar'):
            self.setAuxvarVariable()

        self.dyn = ode
        self.dyn_fn = casadi.Function('dynamics', [self.state, self.control, self.auxvar], [self.dyn])

    def setPathCost(self, path_cost):
        if not hasattr(self, 'auxvar'):
            self.setAuxvarVariable()

        assert path_cost.numel() == 1, "path_cost must be a scalar function"

        self.path_cost = path_cost
        self.path_cost_fn = casadi.Function('path_cost', [self.state, self.control, self.auxvar], [self.path_cost])

    def setFinalCost(self, final_cost):
        if not hasattr(self, 'auxvar'):
            self.setAuxvarVariable()

        assert final_cost.numel() == 1, "final_cost must be a scalar function"

        self.final_cost = final_cost
        self.final_cost_fn = casadi.Function('final_cost', [self.state, self.auxvar], [self.final_cost])

    def ocSolver(self, ini_state, horizon, auxvar_value=1, print_level=0, costate_option=0):
        assert hasattr(self, 'state'), "Define the state variable first!"
        assert hasattr(self, 'control'), "Define the control variable first!"
        assert hasattr(self, 'dyn'), "Define the system dynamics first!"
        assert hasattr(self, 'path_cost'), "Define the running cost function first!"
        assert hasattr(self, 'final_cost'), "Define the final cost function first!"

        if type(ini_state) == numpy.ndarray:
            ini_state = ini_state.flatten().tolist()

        # Start with an empty NLP
        w = []
        w0 = []
        lbw = []
        ubw = []
        J = 0
        g = []
        lbg = []
        ubg = []

        # "Lift" initial conditions
        Xk = MX.sym('X0', self.n_state)
        w += [Xk]
        lbw += ini_state
        ubw += ini_state
        w0 += ini_state

        # Formulate the NLP
        for k in range(horizon):
            # New NLP variable for the control
            Uk = MX.sym('U_' + str(k), self.n_control)
            w += [Uk]
            lbw += self.control_lb
            ubw += self.control_ub
            w0 += [0.5 * (x + y) for x, y in zip(self.control_lb, self.control_ub)]

            # Integrate till the end of the interval
            Xnext = self.dyn_fn(Xk, Uk, auxvar_value)
            Ck = self.path_cost_fn(Xk, Uk, auxvar_value)
            J = J + Ck

            # New NLP variable for state at end of interval
            Xk = MX.sym('X_' + str(k + 1), self.n_state)
            w += [Xk]
            lbw += self.state_lb
            ubw += self.state_ub
            w0 += [0.5 * (x + y) for x, y in zip(self.state_lb, self.state_ub)]

            # Add equality constraint
            g += [Xnext - Xk]
            lbg += self.n_state * [0]
            ubg += self.n_state * [0]

        # Adding the final cost
        J = J + self.final_cost_fn(Xk, auxvar_value)

        # Create an NLP solver and solve it
        opts = {'ipopt.print_level': print_level, 'ipopt.sb': 'yes', 'print_time': print_level}
        prob = {'f': J, 'x': vertcat(*w), 'g': vertcat(*g)}
        solver = nlpsol('solver', 'ipopt', prob, opts)
        # Solve the NLP
        sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
        w_opt = sol['x'].full().flatten()

        # take the optimal control and state
        sol_traj = numpy.concatenate((w_opt, self.n_control * [0]))
        sol_traj = numpy.reshape(sol_traj, (-1, self.n_state + self.n_control))
        state_traj_opt = sol_traj[:, 0:self.n_state]
        control_traj_opt = numpy.delete(sol_traj[:, self.n_state:], -1, 0)
        time = numpy.array([k for k in range(horizon + 1)])

        # Compute the costates using two options
        if costate_option == 0:
            # Default option, which directly obtains the costates from the NLP solver
            costate_traj_opt = numpy.reshape(sol['lam_g'].full().flatten(), (-1, self.n_state))
        else:
            # Another option, which solve the costates by the Pontryagin's Maximum Principle
            # The variable name is consistent with the notations used in the PDP paper
            dfx_fun = casadi.Function('dfx', [self.state, self.control, self.auxvar], [jacobian(self.dyn, self.state)])
            dhx_fun = casadi.Function('dhx', [self.state, self.auxvar], [jacobian(self.final_cost, self.state)])
            dcx_fun = casadi.Function('dcx', [self.state, self.control, self.auxvar],
                                      [jacobian(self.path_cost, self.state)])
            costate_traj_opt = numpy.zeros((horizon, self.n_state))
            costate_traj_opt[-1, :] = dhx_fun(state_traj_opt[-1, :], auxvar_value)
            for k in range(horizon - 1, 0, -1):
                costate_traj_opt[k - 1, :] = dcx_fun(state_traj_opt[k, :], control_traj_opt[k, :],
                                                     auxvar_value).full() + numpy.dot(
                    numpy.transpose(dfx_fun(state_traj_opt[k, :], control_traj_opt[k, :], auxvar_value).full()),
                    costate_traj_opt[k, :])

        # output
        opt_sol = {"state_traj_opt": state_traj_opt,
                   "control_traj_opt": control_traj_opt,
                   "costate_traj_opt": costate_traj_opt,
                   'auxvar_value': auxvar_value,
                   "time": time,
                   "horizon": horizon,
                   "cost": sol['f'].full()}

        return opt_sol

    def diffPMP(self):
        assert hasattr(self, 'state'), "Define the state variable first!"
        assert hasattr(self, 'control'), "Define the control variable first!"
        assert hasattr(self, 'dyn'), "Define the system dynamics first!"
        assert hasattr(self, 'path_cost'), "Define the running cost/reward function first!"
        assert hasattr(self, 'final_cost'), "Define the final cost/reward function first!"

        # Define the Hamiltonian function
        self.costate = casadi.SX.sym('lambda', self.state.numel())
        self.path_Hamil = self.path_cost + dot(self.dyn, self.costate)  # path Hamiltonian
        self.final_Hamil = self.final_cost  # final Hamiltonian

        # Differentiating dynamics; notations here are consistent with the PDP paper
        self.dfx = jacobian(self.dyn, self.state)
        self.dfx_fn = casadi.Function('dfx', [self.state, self.control, self.auxvar], [self.dfx])
        self.dfu = jacobian(self.dyn, self.control)
        self.dfu_fn = casadi.Function('dfu', [self.state, self.control, self.auxvar], [self.dfu])
        self.dfe = jacobian(self.dyn, self.auxvar)
        self.dfe_fn = casadi.Function('dfe', [self.state, self.control, self.auxvar], [self.dfe])

        # First-order derivative of path Hamiltonian
        self.dHx = jacobian(self.path_Hamil, self.state).T
        self.dHx_fn = casadi.Function('dHx', [self.state, self.control, self.costate, self.auxvar], [self.dHx])
        self.dHu = jacobian(self.path_Hamil, self.control).T
        self.dHu_fn = casadi.Function('dHu', [self.state, self.control, self.costate, self.auxvar], [self.dHu])

        # Second-order derivative of path Hamiltonian
        self.ddHxx = jacobian(self.dHx, self.state)
        self.ddHxx_fn = casadi.Function('ddHxx', [self.state, self.control, self.costate, self.auxvar], [self.ddHxx])
        self.ddHxu = jacobian(self.dHx, self.control)
        self.ddHxu_fn = casadi.Function('ddHxu', [self.state, self.control, self.costate, self.auxvar], [self.ddHxu])
        self.ddHxe = jacobian(self.dHx, self.auxvar)
        self.ddHxe_fn = casadi.Function('ddHxe', [self.state, self.control, self.costate, self.auxvar], [self.ddHxe])
        self.ddHux = jacobian(self.dHu, self.state)
        self.ddHux_fn = casadi.Function('ddHux', [self.state, self.control, self.costate, self.auxvar], [self.ddHux])
        self.ddHuu = jacobian(self.dHu, self.control)
        self.ddHuu_fn = casadi.Function('ddHuu', [self.state, self.control, self.costate, self.auxvar], [self.ddHuu])
        self.ddHue = jacobian(self.dHu, self.auxvar)
        self.ddHue_fn = casadi.Function('ddHue', [self.state, self.control, self.costate, self.auxvar], [self.ddHue])

        # First-order derivative of final Hamiltonian
        self.dhx = jacobian(self.final_Hamil, self.state).T
        self.dhx_fn = casadi.Function('dhx', [self.state, self.auxvar], [self.dhx])

        # second order differential of path Hamiltonian
        self.ddhxx = jacobian(self.dhx, self.state)
        self.ddhxx_fn = casadi.Function('ddhxx', [self.state, self.auxvar], [self.ddhxx])
        self.ddhxe = jacobian(self.dhx, self.auxvar)
        self.ddhxe_fn = casadi.Function('ddhxe', [self.state, self.auxvar], [self.ddhxe])

    def getAuxSys(self, state_traj_opt, control_traj_opt, costate_traj_opt, auxvar_value=1):
        statement = [hasattr(self, 'dfx_fn'), hasattr(self, 'dfu_fn'), hasattr(self, 'dfe_fn'),
                     hasattr(self, 'ddHxx_fn'), \
                     hasattr(self, 'ddHxu_fn'), hasattr(self, 'ddHxe_fn'), hasattr(self, 'ddHux_fn'),
                     hasattr(self, 'ddHuu_fn'), \
                     hasattr(self, 'ddHue_fn'), hasattr(self, 'ddhxx_fn'), hasattr(self, 'ddhxe_fn'), ]
        if not all(statement):
            self.diffPMP()

        # Initialize the coefficient matrices of the auxiliary control system: note that all the notations used here are
        # consistent with the notations defined in the PDP paper.
        dynF, dynG, dynE = [], [], []
        matHxx, matHxu, matHxe, matHux, matHuu, matHue, mathxx, mathxe = [], [], [], [], [], [], [], []

        # Solve the above coefficient matrices
        for t in range(numpy.size(control_traj_opt, 0)):
            curr_x = state_traj_opt[t, :]
            curr_u = control_traj_opt[t, :]
            next_lambda = costate_traj_opt[t, :]
            dynF += [self.dfx_fn(curr_x, curr_u, auxvar_value).full()]
            dynG += [self.dfu_fn(curr_x, curr_u, auxvar_value).full()]
            dynE += [self.dfe_fn(curr_x, curr_u, auxvar_value).full()]
            matHxx += [self.ddHxx_fn(curr_x, curr_u, next_lambda, auxvar_value).full()]
            matHxu += [self.ddHxu_fn(curr_x, curr_u, next_lambda, auxvar_value).full()]
            matHxe += [self.ddHxe_fn(curr_x, curr_u, next_lambda, auxvar_value).full()]
            matHux += [self.ddHux_fn(curr_x, curr_u, next_lambda, auxvar_value).full()]
            matHuu += [self.ddHuu_fn(curr_x, curr_u, next_lambda, auxvar_value).full()]
            matHue += [self.ddHue_fn(curr_x, curr_u, next_lambda, auxvar_value).full()]
        mathxx = [self.ddhxx_fn(state_traj_opt[-1, :], auxvar_value).full()]
        mathxe = [self.ddhxe_fn(state_traj_opt[-1, :], auxvar_value).full()]

        auxSys = {"dynF": dynF,
                  "dynG": dynG,
                  "dynE": dynE,
                  "Hxx": matHxx,
                  "Hxu": matHxu,
                  "Hxe": matHxe,
                  "Hux": matHux,
                  "Huu": matHuu,
                  "Hue": matHue,
                  "hxx": mathxx,
                  "hxe": mathxe}
        return auxSys


'''
# =============================================================================================================
# The LQR class is mainly for solving (time-varying or time-invariant) LQR problems.
# The standard form of the dynamics in the LQR system is
# X_k+1=dynF_k*X_k+dynG_k*U_k+dynE_k,
# where matrices dynF_k, dynG_k, and dynE_k are system dynamics matrices you need to specify (maybe time-varying)
# The standard form of cost function for the LQR system is
# J=sum_0^(horizon-1) path_cost + final cost, where
# path_cost  = trace (1/2*X'*Hxx*X +1/2*U'*Huu*U + 1/2*X'*Hxu*U + 1/2*U'*Hux*X + Hue'*U + Hxe'*X)
# final_cost = trace (1/2*X'*hxx*X +hxe'*X)
# Here, Hxx, Huu, Hux, Hxu, Heu, Hex, hxx, hex are cost matrices you need to specify (maybe time-varying).
# Some of the above dynamics and cost matrices, by default, are zero (none) matrices
# Note that the variable X and variable U can be matrix variables.
# The above defined standard form is consistent with the auxiliary control system defined in the PDP paper
'''


class LQR:

    def __init__(self, project_name="LQR system"):
        self.project_name = project_name

    def setDyn(self, dynF, dynG, dynE=None):
        if type(dynF) is numpy.ndarray:
            self.dynF = [dynF]
            self.n_state = numpy.size(dynF, 0)
        elif type(dynF[0]) is numpy.ndarray:
            self.dynF = dynF
            self.n_state = numpy.size(dynF[0], 0)
        else:
            assert False, "Type of dynF matrix should be numpy.ndarray  or list of numpy.ndarray"

        if type(dynG) is numpy.ndarray:
            self.dynG = [dynG]
            self.n_control = numpy.size(dynG, 1)
        elif type(dynG[0]) is numpy.ndarray:
            self.dynG = dynG
            self.n_control = numpy.size(self.dynG[0], 1)
        else:
            assert False, "Type of dynG matrix should be numpy.ndarray  or list of numpy.ndarray"

        if dynE is not None:
            if type(dynE) is numpy.ndarray:
                self.dynE = [dynE]
                self.n_batch = numpy.size(dynE, 1)
            elif type(dynE[0]) is numpy.ndarray:
                self.dynE = dynE
                self.n_batch = numpy.size(dynE[0], 1)
            else:
                assert False, "Type of dynE matrix should be numpy.ndarray, list of numpy.ndarray, or None"
        else:
            self.dynE = None
            self.n_batch = None

    def setPathCost(self, Hxx, Huu, Hxu=None, Hux=None, Hxe=None, Hue=None):

        if type(Hxx) is numpy.ndarray:
            self.Hxx = [Hxx]
        elif type(Hxx[0]) is numpy.ndarray:
            self.Hxx = Hxx
        else:
            assert False, "Type of path cost Hxx matrix should be numpy.ndarray or list of numpy.ndarray, or None"

        if type(Huu) is numpy.ndarray:
            self.Huu = [Huu]
        elif type(Huu[0]) is numpy.ndarray:
            self.Huu = Huu
        else:
            assert False, "Type of path cost Huu matrix should be numpy.ndarray or list of numpy.ndarray, or None"

        if Hxu is not None:
            if type(Hxu) is numpy.ndarray:
                self.Hxu = [Hxu]
            elif type(Hxu[0]) is numpy.ndarray:
                self.Hxu = Hxu
            else:
                assert False, "Type of path cost Hxu matrix should be numpy.ndarray or list of numpy.ndarray, or None"
        else:
            self.Hxu = None

        if Hux is not None:
            if type(Hux) is numpy.ndarray:
                self.Hux = [Hux]
            elif type(Hux[0]) is numpy.ndarray:
                self.Hux = Hux
            else:
                assert False, "Type of path cost Hux matrix should be numpy.ndarray or list of numpy.ndarray, or None"
        else:
            self.Hux = None

        if Hxe is not None:
            if type(Hxe) is numpy.ndarray:
                self.Hxe = [Hxe]
            elif type(Hxe[0]) is numpy.ndarray:
                self.Hxe = Hxe
            else:
                assert False, "Type of path cost Hxe matrix should be numpy.ndarray or list of numpy.ndarray, or None"
        else:
            self.Hxe = None

        if Hue is not None:
            if type(Hue) is numpy.ndarray:
                self.Hue = [Hue]
            elif type(Hue[0]) is numpy.ndarray:
                self.Hue = Hue
            else:
                assert False, "Type of path cost Hue matrix should be numpy.ndarray or list of numpy.ndarray, or None"
        else:
            self.Hue = None

    def setFinalCost(self, hxx, hxe=None):

        if type(hxx) is numpy.ndarray:
            self.hxx = [hxx]
        elif type(hxx[0]) is numpy.ndarray:
            self.hxx = hxx
        else:
            assert False, "Type of final cost hxx matrix should be numpy.ndarray or list of numpy.ndarray"

        if hxe is not None:
            if type(hxe) is numpy.ndarray:
                self.hxe = [hxe]
            elif type(hxe[0]) is numpy.ndarray:
                self.hxe = hxe
            else:
                assert False, "Type of final cost hxe matrix should be numpy.ndarray, list of numpy.ndarray, or None"
        else:
            self.hxe = None

    def lqrSolver(self, ini_state, horizon):

        # Data pre-processing
        n_state = numpy.size(self.dynF[0], 1)
        if type(ini_state) is list:
            self.ini_x = numpy.array(ini_state, numpy.float64)
            if self.ini_x.ndim == 2:
                self.n_batch = numpy.size(self.ini_x, 1)
            else:
                self.n_batch = 1
                self.ini_x = self.ini_x.reshape(n_state, -1)
        elif type(ini_state) is numpy.ndarray:
            self.ini_x = ini_state
            if self.ini_x.ndim == 2:
                self.n_batch = numpy.size(self.ini_x, 1)
            else:
                self.n_batch = 1
                self.ini_x = self.ini_x.reshape(n_state, -1)
        else:
            assert False, "Initial state should be of numpy.ndarray type or list!"

        self.horizon = horizon

        if self.dynE is not None:
            assert self.n_batch == numpy.size(self.dynE[0],
                                              1), "Number of data batch is not consistent with column of dynE"

        # Check the time horizon
        if len(self.dynF) > 1 and len(self.dynF) != self.horizon:
            assert False, "time-varying dynF is not consistent with given horizon"
        elif len(self.dynF) == 1:
            F = self.horizon * self.dynF
        else:
            F = self.dynF

        if len(self.dynG) > 1 and len(self.dynG) != self.horizon:
            assert False, "time-varying dynG is not consistent with given horizon"
        elif len(self.dynG) == 1:
            G = self.horizon * self.dynG
        else:
            G = self.dynG

        if self.dynE is not None:
            if len(self.dynE) > 1 and len(self.dynE) != self.horizon:
                assert False, "time-varying dynE is not consistent with given horizon"
            elif len(self.dynE) == 1:
                E = self.horizon * self.dynE
            else:
                E = self.dynE
        else:
            E = self.horizon * [numpy.zeros(self.ini_x.shape)]

        if len(self.Hxx) > 1 and len(self.Hxx) != self.horizon:
            assert False, "time-varying Hxx is not consistent with given horizon"
        elif len(self.Hxx) == 1:
            Hxx = self.horizon * self.Hxx
        else:
            Hxx = self.Hxx

        if len(self.Huu) > 1 and len(self.Huu) != self.horizon:
            assert False, "time-varying Huu is not consistent with given horizon"
        elif len(self.Huu) == 1:
            Huu = self.horizon * self.Huu
        else:
            Huu = self.Huu

        hxx = self.hxx

        if self.hxe is None:
            hxe = [numpy.zeros(self.ini_x.shape)]

        if self.Hxu is None:
            Hxu = self.horizon * [numpy.zeros((self.n_state, self.n_control))]
        else:
            if len(self.Hxu) > 1 and len(self.Hxu) != self.horizon:
                assert False, "time-varying Hxu is not consistent with given horizon"
            elif len(self.Hxu) == 1:
                Hxu = self.horizon * self.Hxu
            else:
                Hxu = self.Hxu

        if self.Hux is None:  # Hux is the transpose of Hxu
            Hux = self.horizon * [numpy.zeros((self.n_control, self.n_state))]
        else:
            if len(self.Hux) > 1 and len(self.Hux) != self.horizon:
                assert False, "time-varying Hux is not consistent with given horizon"
            elif len(self.Hux) == 1:
                Hux = self.horizon * self.Hux
            else:
                Hux = self.Hux

        if self.Hxe is None:
            Hxe = self.horizon * [numpy.zeros((self.n_state, self.n_batch))]
        else:
            if len(self.Hxe) > 1 and len(self.Hxe) != self.horizon:
                assert False, "time-varying Hxe is not consistent with given horizon"
            elif len(self.Hxe) == 1:
                Hxe = self.horizon * self.Hxe
            else:
                Hxe = self.Hxe

        if self.Hue is None:
            Hue = self.horizon * [numpy.zeros((self.n_control, self.n_batch))]
        else:
            if len(self.Hue) > 1 and len(self.Hue) != self.horizon:
                assert False, "time-varying Hue is not consistent with given horizon"
            elif len(self.Hue) == 1:
                Hue = self.horizon * self.Hue
            else:
                Hue = self.Hue

        # Solve the Riccati equations: the notations used here are consistent with Lemma 4.2 in the PDP paper
        I = numpy.eye(self.n_state)
        PP = self.horizon * [numpy.zeros((self.n_state, self.n_state))]
        WW = self.horizon * [numpy.zeros((self.n_state, self.n_batch))]
        PP[-1] = self.hxx[0]
        WW[-1] = self.hxe[0]
        for t in range(self.horizon - 1, 0, -1):
            P_next = PP[t]
            W_next = WW[t]
            invHuu = numpy.linalg.inv(Huu[t])
            GinvHuu = numpy.matmul(G[t], invHuu)
            HxuinvHuu = numpy.matmul(Hxu[t], invHuu)
            A_t = F[t] - numpy.matmul(GinvHuu, numpy.transpose(Hxu[t]))
            R_t = numpy.matmul(GinvHuu, numpy.transpose(G[t]))
            M_t = E[t] - numpy.matmul(GinvHuu, Hue[t])
            Q_t = Hxx[t] - numpy.matmul(HxuinvHuu, numpy.transpose(Hxu[t]))
            N_t = Hxe[t] - numpy.matmul(HxuinvHuu, Hue[t])

            temp_mat = numpy.matmul(numpy.transpose(A_t), numpy.linalg.inv(I + numpy.matmul(P_next, R_t)))
            P_curr = Q_t + numpy.matmul(temp_mat, numpy.matmul(P_next, A_t))
            W_curr = N_t + numpy.matmul(temp_mat, W_next + numpy.matmul(P_next, M_t))

            PP[t - 1] = P_curr
            WW[t - 1] = W_curr

        # Compute the trajectory using the Raccti matrices obtained from the above: the notations used here are
        # consistent with the PDP paper in Lemma 4.2
        state_traj_opt = (self.horizon + 1) * [numpy.zeros((self.n_state, self.n_batch))]
        control_traj_opt = (self.horizon) * [numpy.zeros((self.n_control, self.n_batch))]
        costate_traj_opt = (self.horizon) * [numpy.zeros((self.n_state, self.n_batch))]
        state_traj_opt[0] = self.ini_x
        for t in range(self.horizon):
            P_next = PP[t]
            W_next = WW[t]
            invHuu = numpy.linalg.inv(Huu[t])
            GinvHuu = numpy.matmul(G[t], invHuu)
            A_t = F[t] - numpy.matmul(GinvHuu, numpy.transpose(Hxu[t]))
            M_t = E[t] - numpy.matmul(GinvHuu, Hue[t])
            R_t = numpy.matmul(GinvHuu, numpy.transpose(G[t]))

            x_t = state_traj_opt[t]
            u_t = -numpy.matmul(invHuu, numpy.matmul(numpy.transpose(Hxu[t]), x_t) + Hue[t]) \
                  - numpy.linalg.multi_dot([invHuu, numpy.transpose(G[t]), numpy.linalg.inv(I + numpy.dot(P_next, R_t)),
                                            (numpy.matmul(numpy.matmul(P_next, A_t), x_t) + numpy.matmul(P_next,
                                                                                                         M_t) + W_next)])

            x_next = numpy.matmul(F[t], x_t) + numpy.matmul(G[t], u_t) + E[t]
            lambda_next = numpy.matmul(P_next, x_next) + W_next

            state_traj_opt[t + 1] = x_next
            control_traj_opt[t] = u_t
            costate_traj_opt[t] = lambda_next
        time = [k for k in range(self.horizon + 1)]

        opt_sol = {'state_traj_opt': state_traj_opt,
                   'control_traj_opt': control_traj_opt,
                   'costate_traj_opt': costate_traj_opt,
                   'time': time}
        return opt_sol


'''
# =============================================================================================================
# This class is used to solve the motion planning or optimal control problems:
# The standard form of the system dynamics is
# x_k+1= f（x_k, u_k)
# The standard form of the cost function is
# J = sum_0^(T-1) path_cost +final cost,
# where
# path_cost = c(x, u)
# final_cost= h(x)
Note that most of the notations used in codes are consistent with the notations defined in the PDP paper

The procedure to use ControlPlanning is fairly straightforward, just understand each method by looking at its name:
* Step 1: set state variable ----> setStateVariable
* Step 2: set control variable ----> setControlVariable
* Step 3: set dynamics equation----> setDyn
* Step 5: set path cost function ----> setPathCost
* Step 6: set final cost function -----> setFinalCost
* Step 7: set control policy -----> setPolyControl (usually for planning) OR setNeuralPolicy (for control)
* Step 8: integrate the control system in forward pass -----> integrateSys
* Step 9: get the auxiliary control system ------> getAuxSys

Note that method init_step  wraps Step 7, and method step wraps Step 8 and Step 9.

The user can also choose the following added features to improve the execution of PDP:
One is the warping techniques: please see the methods beginning with 'warped_'. 
The advantage of using the warping technique is that PDP Control/Planning Mode is more robust!
The other is the recovery matrix techniques (https://arxiv.org/abs/1803.07696): please see all the methods beginning with 'recmat_'. 
The advantage of using the recovery matrix is that PDP Control/Planning Mode is more faster!
'''


class ControlPlanning:

    def __init__(self, project_name="planner"):
        self.project_name = project_name

    def setStateVariable(self, state, state_lb=[], state_ub=[]):
        self.state = state
        self.n_state = self.state.numel()
        if len(state_lb) == self.n_state:
            self.state_lb = state_lb
        else:
            self.state_lb = self.n_state * [-1e20]

        if len(state_ub) == self.n_state:
            self.state_ub = state_ub
        else:
            self.state_ub = self.n_state * [1e20]

    def setControlVariable(self, control, control_lb=[], control_ub=[]):
        self.control = control
        self.n_control = self.control.numel()

        if len(control_lb) == self.n_control:
            self.control_lb = control_lb
        else:
            self.control_lb = self.n_control * [-1e20]

        if len(control_ub) == self.n_control:
            self.control_ub = control_ub
        else:
            self.control_ub = self.n_control * [1e20]

    def setDyn(self, ode):
        self.dyn = ode
        self.dyn_fn = casadi.Function('dynFun', [self.state, self.control], [self.dyn])

        # Differentiate system dynamics
        self.dfx = jacobian(self.dyn, self.state)
        self.dfx_fn = casadi.Function('dfx', [self.state, self.control], [self.dfx])
        self.dfu = jacobian(self.dyn, self.control)
        self.dfu_fn = casadi.Function('dfu', [self.state, self.control], [self.dfu])

    def setPathCost(self, path_cost):
        self.path_cost = path_cost
        self.path_cost_fn = casadi.Function('pathCost', [self.state, self.control], [self.path_cost])

        # Differentiate the objective (cost) functions: the notations used here are the consistent with the notations
        # defined in the PDP paper
        # This is in fact belonging to diffPMP, but we just add it here
        self.dcx_fn = casadi.Function('dcx', [self.state, self.control], [jacobian(self.path_cost, self.state)])
        self.dcu_fn = casadi.Function('dcx', [self.state, self.control], [jacobian(self.path_cost, self.control)])

    def setFinalCost(self, final_cost):
        self.final_cost = final_cost
        self.final_cost_fn = casadi.Function('finalCost', [self.state], [self.final_cost])

        # Differentiate the final cost function
        self.dhx_fn = casadi.Function('dhx', [self.state], [jacobian(self.final_cost, self.state)])

    def setPolyControl(self, pivots):
        # Use the Lagrange polynomial to represent the control (input) function: u_t=u(t,auxvar).
        # Note that here we still use auxvar to denote the unknown parameter

        # time variable
        self.t = SX.sym('t')

        # pivots of time steps for the Lagrange polynomial
        poly_control = 0
        piovts_control = []
        for i in range(len(pivots)):
            Ui = SX.sym('U_' + str(i), self.n_control)
            piovts_control += [Ui]
            bi = 1
            for j in range(len(pivots)):
                if j != i:
                    bi = bi * (self.t - pivots[j]) / (pivots[i] - pivots[j])
            poly_control = poly_control + bi * Ui
        self.auxvar = vcat(piovts_control)
        self.n_auxvar = self.auxvar.numel()
        self.policy_fn = casadi.Function('policy_fn', [self.t, self.state, self.auxvar], [poly_control])

        # Differentiate control policy function
        dpolicy_dx = casadi.jacobian(poly_control, self.state)
        self.dpolicy_dx_fn = casadi.Function('dpolicy_dx', [self.t, self.state, self.auxvar], [dpolicy_dx])
        dpolicy_de = casadi.jacobian(poly_control, self.auxvar)
        self.dpolicy_de_fn = casadi.Function('dpolicy_de', [self.t, self.state, self.auxvar], [dpolicy_de])

    def setNeuralPolicy(self,hidden_layers):
        # Use neural network to represent the policy function: u_t=u(t,x,auxvar).
        # Note that here we use auxvar to denote the parameter of the neural policy
        layers=hidden_layers+[self.n_control]

        # time variable
        self.t = SX.sym('t')

        # construct the neural policy with the argument inputs to specify the hidden layers of the neural policy
        a=self.state
        auxvar=[]
        Ak = SX.sym('Ak', layers[0], self.n_state)  # weights matrix
        bk = SX.sym('bk', layers[0])  # bias vector
        auxvar += [Ak.reshape((-1, 1))]
        auxvar += [bk]
        a=mtimes(Ak, a) + bk
        for i in range(len(layers)-1):
            a=tanh(a)
            Ak = SX.sym('Ak', layers[i+1],layers[i] )  # weights matrix
            bk = SX.sym('bk', layers[i+1])  # bias vector
            auxvar += [Ak.reshape((-1, 1))]
            auxvar += [bk]
            a = mtimes(Ak, a) + bk
        self.auxvar=vcat(auxvar)
        self.n_auxvar = self.auxvar.numel()
        neural_policy=a
        self.policy_fn = casadi.Function('policy_fn', [self.t, self.state, self.auxvar], [neural_policy])

        # Differentiate control policy function
        dpolicy_dx = casadi.jacobian(neural_policy, self.state)
        self.dpolicy_dx_fn = casadi.Function('dpolicy_dx', [self.t, self.state, self.auxvar], [dpolicy_dx])
        dpolicy_de = casadi.jacobian(neural_policy, self.auxvar)
        self.dpolicy_de_fn = casadi.Function('dpolicy_de', [self.t, self.state, self.auxvar], [dpolicy_de])

    # The following are to solve PDP control and planning with polynomial control policy

    def integrateSys(self, ini_state, horizon, auxvar_value):
        assert hasattr(self, 'dyn_fn'), "Set the dynamics first!"
        assert hasattr(self, 'policy_fn'), "Set the control policy first, you may use [setPolicy_polyControl] "

        if type(ini_state) == list:
            ini_state = numpy.array(ini_state)

        # do the system integration
        control_traj = numpy.zeros((horizon, self.n_control))
        state_traj = numpy.zeros((horizon + 1, self.n_state))
        state_traj[0, :] = ini_state
        cost = 0
        for t in range(horizon):
            curr_x = state_traj[t, :]
            curr_u = self.policy_fn(t, curr_x, auxvar_value).full().flatten()
            state_traj[t + 1, :] = self.dyn_fn(curr_x, curr_u).full().flatten()
            control_traj[t, :] = curr_u
            cost += self.path_cost_fn(curr_x, curr_u).full()
        cost += self.final_cost_fn(state_traj[-1, :]).full()

        traj_sol = {'state_traj': state_traj,
                    'control_traj': control_traj,
                    'cost': cost.item()}
        return traj_sol

    def getAuxSys(self, state_traj, control_traj, auxvar_value):
        # check the pre-requisite conditions
        assert hasattr(self, 'dfx_fn'), "Set the dynamics equation first!"
        assert hasattr(self, 'dpolicy_de_fn'), "Set the policy first, you may want to use method [setPolicy_]"
        assert hasattr(self, 'dpolicy_dx_fn'), "Set the policy first, you may want to use method [setPolicy_]"

        # Initialize the coefficient matrices of the auxiliary control system: note that all the notations used here are
        # consistent with the notations defined in the PDP paper.
        dynF, dynG = [], []
        dUx, dUe = [], []
        for t in range(numpy.size(control_traj, 0)):
            curr_x = state_traj[t, :]
            curr_u = control_traj[t, :]
            dynF += [self.dfx_fn(curr_x, curr_u).full()]
            dynG += [self.dfu_fn(curr_x, curr_u).full()]
            dUx += [self.dpolicy_dx_fn(t, curr_x, auxvar_value).full()]
            dUe += [self.dpolicy_de_fn(t, curr_x, auxvar_value).full()]

        auxSys = {"dynF": dynF,
                  "dynG": dynG,
                  "dUx": dUx,
                  "dUe": dUe}

        return auxSys

    def integrateAuxSys(self, dynF, dynG, dUx, dUe, ini_condition):

        # pre-requisite check
        if type(dynF) != list or type(dynG) != list or type(dUx) != list or type(dUe) != list:
            assert False, "The input dynF, dynE, dUx, and dUe should be list of numpy.array!"
        if len(dynG) != len(dynF) or len(dUe) != len(dUx) or len(dUe) != len(dynG):
            assert False, "The length of dynF, dynE, dUx, and dUe should be the same"
        if type(ini_condition) is not numpy.ndarray:
            assert False, "The initial condition should be numpy.array"

        horizon = len(dynF)
        state_traj = [ini_condition]
        control_traj = []
        for t in range(horizon):
            F_t = dynF[t]
            G_t = dynG[t]
            Ux_t = dUx[t]
            Ue_t = dUe[t]
            X_t = state_traj[t]
            U_t = numpy.matmul(Ux_t, X_t) + Ue_t
            state_traj += [numpy.matmul(F_t, X_t) + numpy.matmul(G_t, U_t)]
            control_traj += [U_t]

        aux_sol = {'state_traj': state_traj,
                   'control_traj': control_traj}
        return aux_sol

    def init_step(self, horizon, n_poly=5):
        # set the control polynomial policy
        pivots = numpy.linspace(0, horizon, n_poly + 1)
        self.setPolyControl(pivots)

    def init_step_neural_policy(self, hidden_layers=None):
        if hidden_layers is None:
            hidden_layers=[self.n_state]
        self.setNeuralPolicy(hidden_layers)

    def step(self, ini_state, horizon, auxvar_value):

        assert hasattr(self, 'policy_fn'), 'please set the control policy by running the init_step method first!'

        # generate the system trajectory using the current policy
        sol = self.integrateSys(ini_state=ini_state, horizon=horizon, auxvar_value=auxvar_value)
        state_traj = sol['state_traj']
        control_traj = sol['control_traj']
        loss = sol['cost']

        #  establish the auxiliary control system
        aux_sys = self.getAuxSys(state_traj=state_traj, control_traj=control_traj, auxvar_value=auxvar_value)
        # solve the auxiliary control system
        aux_sol = self.integrateAuxSys(dynF=aux_sys['dynF'], dynG=aux_sys['dynG'],
                                       dUx=aux_sys['dUx'], dUe=aux_sys['dUe'],
                                       ini_condition=numpy.zeros((self.n_state, self.n_auxvar)))
        dxdauxvar_traj = aux_sol['state_traj']
        dudauxvar_traj = aux_sol['control_traj']

        # Evaluate the current loss and the gradients
        dauxvar = numpy.zeros(self.n_auxvar)
        for t in range(horizon):
            # chain rule
            dauxvar += (numpy.matmul(self.dcx_fn(state_traj[t, :], control_traj[t, :]).full(), dxdauxvar_traj[t]) +
                        numpy.matmul(self.dcu_fn(state_traj[t, :], control_traj[t, :]).full(),
                                     dudauxvar_traj[t])).flatten()
        dauxvar += numpy.matmul(self.dhx_fn(state_traj[-1, :]).full(), dxdauxvar_traj[-1]).flatten()

        return loss, dauxvar

    # The following are to solve PDP control and planning with polynomial control policy by warping techniques

    def warp_dynCost(self, time_grid):

        assert hasattr(self, 'dyn_fn'), 'Please set the dynamics first!'
        assert hasattr(self, 'path_cost_fn'), 'Please set the path cost first!'
        assert hasattr(self, 'final_cost_fn'), 'Please set the final cost first!'

        # define warped dynamics and cost function
        self.wdyn_fns = []
        self.wdfx_fns = []
        self.wdfu_fns = []
        self.wpath_cost_fns = []
        self.wdcx_fns = []
        self.wdcu_fns = []

        # warp dynamics and path cost function
        for wt in range(len(time_grid) - 1):
            X = self.state
            U = self.control
            path_cost = 0
            for t in range(time_grid[wt], time_grid[wt + 1]):
                path_cost += self.path_cost_fn(X, U)
                X = self.dyn_fn(X, U)
            self.wdyn_fns += [Function('wdyn_fn' + str(wt), [self.state, self.control], [X])]
            self.wdfx_fns += [Function('wdfx_fn' + str(wt), [self.state, self.control], [jacobian(X, self.state)])]
            self.wdfu_fns += [Function('wdfu_fn' + str(wt), [self.state, self.control], [jacobian(X, self.control)])]
            self.wpath_cost_fns += [Function('wpath_cost_fn' + str(wt), [self.state, self.control], [path_cost])]
            self.wdcx_fns += [
                Function('wdcx_fn' + str(wt), [self.state, self.control], [jacobian(path_cost, self.state)])]
            self.wdcu_fns += [
                Function('wdcu_fn' + str(wt), [self.state, self.control], [jacobian(path_cost, self.control)])]

        # warp final cost function
        self.wfinal_cost_fn = self.final_cost_fn
        self.wdhx_fn = self.dhx_fn

    def warp_integrateSys(self, ini_state, whorizon, auxvar_value):
        assert hasattr(self, 'wdyn_fns'), "Warp the dynamics first by runing the method of warp_init_step! "

        if type(ini_state) == list:
            ini_state = numpy.array(ini_state)

        # do the system integration
        wcontrol_traj = numpy.zeros((whorizon, self.n_control))
        wstate_traj = numpy.zeros((whorizon + 1, self.n_state))
        wstate_traj[0, :] = ini_state
        wcost = 0
        for wt in range(whorizon):
            curr_x = wstate_traj[wt, :]
            curr_u = self.policy_fn(wt, curr_x, auxvar_value).full().flatten()
            wstate_traj[wt + 1, :] = self.wdyn_fns[wt](curr_x, curr_u).full().flatten()
            wcontrol_traj[wt, :] = curr_u
            wcost += self.wpath_cost_fns[wt](curr_x, curr_u).full()
        wcost += self.wfinal_cost_fn(wstate_traj[-1, :])

        wsol = {'wstate_traj': wstate_traj,
                'wcontrol_traj': wcontrol_traj,
                'wcost': wcost}
        return wsol

    def warp_getAuxSys(self, wstate_traj, wcontrol_traj, auxvar_value):

        wdynF, wdynG = [], []
        wdUx, wdUe = [], []
        for wt in range(numpy.size(wcontrol_traj, 0)):
            wcurr_x = wstate_traj[wt, :]
            wcurr_u = wcontrol_traj[wt, :]
            wdynF += [self.wdfx_fns[wt](wcurr_x, wcurr_u).full()]
            wdynG += [self.wdfu_fns[wt](wcurr_x, wcurr_u).full()]
            wdUx += [self.dpolicy_dx_fn(wt, wcurr_x, auxvar_value).full()]
            wdUe += [self.dpolicy_de_fn(wt, wcurr_x, auxvar_value).full()]

        wAuxSys = {"wdynF": wdynF,
                   "wdynG": wdynG,
                   "wdUx": wdUx,
                   "wdUe": wdUe}

        return wAuxSys

    def warp_init_step(self, horizon, time_grid=None):
        if time_grid is None:
            time_grid = numpy.linspace(0, 1, numpy.amin([horizon + 1, 11]))

        if type(time_grid) == list:
            time_grid = numpy.array(time_grid)

        if numpy.isscalar(time_grid) and time_grid == -1:
            time_grid = numpy.linspace(0, horizon - 1, horizon)

        self.time_grid = numpy.rint(horizon * time_grid / time_grid[-1]).astype(int)

        # warp dynamics and cost functions
        self.warp_dynCost(self.time_grid)
        self.whorizon = len(self.time_grid) - 1

        # set policy which is consistent with the warped dynamics and cost function
        pivots = numpy.linspace(0, self.whorizon, self.whorizon + 1)
        self.setPolyControl(pivots)

    def warp_step(self, ini_state, horizon, auxvar_value):

        assert hasattr(self, 'time_grid'), "Run warp_init_step first!"
        # generate the current trajectory
        wsol = self.warp_integrateSys(ini_state=ini_state, whorizon=self.whorizon, auxvar_value=auxvar_value)

        wstate_traj = wsol['wstate_traj']
        wcontrol_traj = wsol['wcontrol_traj']
        loss = wsol['wcost']

        # Establish the auxiliary control system
        waux_sys = self.warp_getAuxSys(wstate_traj=wstate_traj, wcontrol_traj=wcontrol_traj, auxvar_value=auxvar_value)
        # Solve the auxiliary control system
        aux_sol = self.integrateAuxSys(dynF=waux_sys['wdynF'], dynG=waux_sys['wdynG'], dUx=waux_sys['wdUx'],
                                       dUe=waux_sys['wdUe'], ini_condition=np.zeros((self.n_state, self.n_auxvar)))
        dxdauxvar_traj = aux_sol['state_traj']
        dudauxvar_traj = aux_sol['control_traj']

        # Evaluate the gradients
        dauxvar = numpy.zeros(self.n_auxvar)
        for wt in range(self.whorizon):
            # chain rule
            dauxvar += (numpy.matmul(self.wdcx_fns[wt](wstate_traj[wt, :], wcontrol_traj[wt, :]).full(),
                                     dxdauxvar_traj[wt]) +
                        numpy.matmul(self.wdcu_fns[wt](wstate_traj[wt, :], wcontrol_traj[wt, :]).full(),
                                     dudauxvar_traj[wt])).flatten()
        dauxvar += numpy.matmul(self.wdhx_fn(wstate_traj[-1, :]).full(), dxdauxvar_traj[-1]).flatten()

        return loss, dauxvar

    def warp_unwarp(self, ini_state, horizon, auxvar_value):

        if type(ini_state) == list:
            ini_state = numpy.array(ini_state)

        control_traj = numpy.zeros((horizon, self.n_control))

        for wt in range(self.whorizon):
            for t in range(self.time_grid[wt], self.time_grid[wt + 1]):
                control_traj[t, :] = self.policy_fn(wt, numpy.zeros(self.n_state), auxvar_value).full().flatten()

        state_traj = numpy.zeros((horizon + 1, self.n_state))
        state_traj[0, :] = ini_state
        cost = 0
        for t in range(horizon):
            curr_x = state_traj[t, :]
            curr_u = control_traj[t, :]
            state_traj[t + 1, :] = self.dyn_fn(curr_x, curr_u).full().flatten()
            cost += self.path_cost_fn(curr_x, curr_u).full().flatten()
        cost += self.final_cost_fn(state_traj[-1, :]).full().flatten()

        sol = {'state_traj': state_traj,
               'control_traj': control_traj,
               'cost': cost}

        return sol

    # The following are recovery matrix method to solve PDP control and planning

    def recmat_recoveryMatrix(self, whorizon):
        assert hasattr(self, 'wdyn_fns'), 'Please warp the dynamics and cost function first by running warp_init_step!'

        controls = []
        ini_state = SX.sym('X0', self.n_state)
        X_t = ini_state
        U_t = SX.sym('U_' + str(0), self.n_control)
        G_t = self.wdfu_fns[0](X_t, U_t)
        Cu_t = self.wdcu_fns[0](X_t, U_t)
        controls += [U_t]
        X_next = self.wdyn_fns[0](X_t, U_t)
        U_next = SX.sym('U_' + str(1), self.n_control)
        F_next = self.wdfx_fns[1](X_next, U_next)
        Cx_next = self.wdcx_fns[1](X_next, U_next)
        H1 = [mtimes(Cx_next, G_t) + Cu_t]
        H2 = [mtimes(F_next, G_t)]
        U_t = U_next
        X_t = X_next
        for wt in range(1, whorizon - 1):
            G_t = self.wdfu_fns[wt](X_t, U_t)
            Cu_t = self.wdcu_fns[wt](X_t, U_t)
            controls += [U_t]
            X_next = self.wdyn_fns[wt](X_t, U_t)
            U_next = SX.sym('U_' + str(wt + 1), self.n_control)
            F_next = self.wdfx_fns[wt + 1](X_next, U_next)
            Cx_next = self.wdcx_fns[wt + 1](X_next, U_next)
            H1 = [hcat(H1) + mtimes(Cx_next, hcat(H2))] + [mtimes(Cx_next, G_t) + Cu_t]
            H2 = [mtimes(F_next, hcat(H2))] + [mtimes(F_next, G_t)]
            U_t = U_next
            X_t = X_next
        G_t = self.wdfu_fns[whorizon - 1](X_t, U_t)
        Cu_t = self.wdcu_fns[whorizon - 1](X_t, U_t)
        controls += [U_t]
        X_next = self.wdyn_fns[whorizon - 1](X_t, U_t)
        Cx_next = self.wdhx_fn(X_next)
        H1 = [hcat(H1) + mtimes(Cx_next, hcat(H2))] + [mtimes(Cx_next, G_t) + Cu_t]
        recovery_matrix = hcat(H1)

        self.auxvar = vcat(controls)
        self.n_auxvar = self.auxvar.numel()
        self.recovery_matrix_fn = Function('recovery_matrix_fn', [ini_state, self.auxvar], [transpose(recovery_matrix)])

    def recmat_init_step(self, horizon, time_grid=None):
        if time_grid is None:
            time_grid = numpy.linspace(0, 1, numpy.amin([horizon + 1, 11]))

        if numpy.isscalar(time_grid) and time_grid == -1:
            time_grid = numpy.linspace(0, horizon, horizon + 1)

        if type(time_grid) == list:
            time_grid = numpy.array(time_grid)

        self.time_grid = numpy.rint(horizon * time_grid / time_grid[-1]).astype(int)

        # warp dynamics and cost functions
        self.warp_dynCost(self.time_grid)
        self.whorizon = len(self.time_grid) - 1

        # generate the recovery matrix
        self.recmat_recoveryMatrix(self.whorizon)

    def recmat_step(self, ini_state, horizon, auxvar_value):

        # integrate the control system
        wcost = 0
        curr_x = ini_state
        for wt in range(self.whorizon):
            curr_u = auxvar_value[wt * self.n_control:wt * self.n_control + self.n_control]
            wcost += self.wpath_cost_fns[wt](curr_x, curr_u).full().flatten().item()
            curr_x = self.wdyn_fns[wt](curr_x, curr_u).full().flatten()
        wcost += self.wfinal_cost_fn(curr_x).full().flatten().item()

        # compute the gradient for parameters
        dauxvar = self.recovery_matrix_fn(ini_state, auxvar_value).full().flatten()

        return wcost, dauxvar

    def recmat_unwarp(self, ini_state, horizon, auxvar_value):

        if type(ini_state) == list:
            ini_state = numpy.array(ini_state)

        control_traj = numpy.zeros((horizon, self.n_control))

        for wt in range(self.whorizon):
            for t in range(self.time_grid[wt], self.time_grid[wt + 1]):
                control_traj[t, :] = auxvar_value[wt * self.n_control:wt * self.n_control + self.n_control]

        state_traj = numpy.zeros((horizon + 1, self.n_state))
        state_traj[0, :] = ini_state
        cost = 0
        for t in range(horizon):
            curr_x = state_traj[t, :]
            curr_u = control_traj[t, :]
            state_traj[t + 1, :] = self.dyn_fn(curr_x, curr_u).full().flatten()
            cost += self.path_cost_fn(curr_x, curr_u).full().flatten()
        cost += self.final_cost_fn(state_traj[-1, :]).full().flatten()

        sol = {'state_traj': state_traj,
               'control_traj': control_traj,
               'cost': cost}

        return sol


'''
# =============================================================================================================
# This class is used to solve the system identification problems
# The standard form for the dynamics model is
# x_k+1= f（x_k, u_k, auxvar), 
# where auxvar is the unknown system parameter that needs to be identified
# To do the system ID, the user needs to provide a sequence of inputs-states
# 
# Note that most of the notations used below are consistent with the notations defined in the PDP paper
'''


class SysID:

    def __init__(self, project_name='my system identification'):
        self.project_name = project_name

    def setAuxvarVariable(self, auxvar):
        self.auxvar = auxvar
        self.n_auxvar = self.auxvar.numel()

    def setStateVariable(self, state):
        self.state = state
        self.n_state = self.state.numel()
        self.state_lb = self.n_state * [-1e20]
        self.state_ub = self.n_state * [1e20]

    def setControlVariable(self, control):
        self.control = control
        self.n_control = self.control.numel()
        self.control_lb = self.n_control * [-1e20]
        self.control_ub = self.n_control * [1e20]

    def setDyn(self, ode):
        self.dyn = ode
        self.dyn_fn = casadi.Function('dyn_fn', [self.state, self.control, self.auxvar], [self.dyn])

        # Differentiate the system dynamics model
        self.dfx = jacobian(self.dyn, self.state)
        self.dfx_fn = casadi.Function('dfx', [self.state, self.control, self.auxvar], [self.dfx])
        self.dfu = jacobian(self.dyn, self.control)
        self.dfu_fn = casadi.Function('dfu', [self.state, self.control, self.auxvar], [self.dfu])
        self.dfe = jacobian(self.dyn, self.auxvar)
        self.dfe_fn = casadi.Function('dfe', [self.state, self.control, self.auxvar], [self.dfe])

    def getRandomInputs(self, horizon=10, n_batch=1, lb=None, ub=None):
        # set the upper bound and lower bound
        if lb is None:
            lb = self.n_control * [-1]
        if ub is None:
            ub = self.n_control * [1]

        # generate random inputs
        batch_inputs = []
        for n in range(n_batch):
            inputs = numpy.zeros((horizon, self.n_control))
            for i in range(self.n_control):
                i_lb = lb[i]
                i_ub = ub[i]
                inputs[:, i] = (i_ub - i_lb) * numpy.random.random(horizon) + i_lb
            batch_inputs += [inputs]

        return batch_inputs

    def integrateDyn(self, ini_state, inputs, auxvar_value):
        # check the pre-requisite conditions
        assert hasattr(self, 'dyn_fn'), "set the dynamics first!"

        if type(ini_state) == list:
            ini_state = numpy.array(ini_state)

        # do the system integration
        horizon = numpy.size(inputs, 0)
        states = numpy.zeros((horizon + 1, self.n_state))
        states[0, :] = ini_state
        for t in range(horizon):
            states[t + 1, :] = self.dyn_fn(states[t], inputs[t], auxvar_value).full().flatten()

        return states

    def getAuxSys(self, state_traj, control_traj, auxvar_value):

        # Initialize the coefficient matrices of the auxiliary control system: note that all the notations used here are
        # consistent with the notations defined in the PDP paper.
        dynF, dynE = [], []
        for t in range(numpy.size(control_traj, 0)):
            curr_x = state_traj[t, :]
            curr_u = control_traj[t, :]
            dynF += [self.dfx_fn(curr_x, curr_u, auxvar_value).full()]
            dynE += [self.dfe_fn(curr_x, curr_u, auxvar_value).full()]

        auxSys = {"dynF": dynF,
                  "dynE": dynE}

        return auxSys

    def integrateAuxSys(self, dynF, dynE, ini_condition):

        # pre-requisite check
        if type(dynF) != list or type(dynE) != list:
            assert False, "The input dynF and dynE should be list of numpy.array!"
        if len(dynE) != len(dynF):
            assert False, "The length of dynF and dynE should be the same"
        if type(ini_condition) is not numpy.ndarray:
            assert False, "The initial condition should be numpy.array"

        horizon = len(dynF)
        state_traj = [ini_condition]
        for t in range(horizon):
            Ft = dynF[t]
            Fe = dynE[t]
            state_traj += [numpy.matmul(Ft, state_traj[t]) + Fe]

        aux_sol = {'state_traj': state_traj}
        return aux_sol

    def step(self, batch_inputs, batch_states, auxvar_value):

        n_batch = len(batch_inputs)
        loss = 0
        dauxvar = np.zeros(self.n_auxvar)
        for i in range(n_batch):

            # for each trajectory, extract the information
            input_traj = batch_inputs[i]
            ini_state = batch_states[i][0, :]
            horizon = np.size(batch_inputs[i], 0)
            ob_state_traj = batch_states[i]

            # current trajectory based on current parameter guess
            state_traj = self.integrateDyn(ini_state=ini_state, inputs=input_traj, auxvar_value=auxvar_value)
            aux_sys = self.getAuxSys(state_traj=state_traj, control_traj=input_traj, auxvar_value=auxvar_value)
            aux_sol = self.integrateAuxSys(dynF=aux_sys['dynF'],
                                           dynE=aux_sys['dynE'],
                                           ini_condition=np.zeros((self.n_state, self.n_auxvar)))

            # take the solution of auxiliary control system
            dxdauxvar = aux_sol['state_traj']

            # evaluate the loss
            dldx_traj = state_traj - ob_state_traj
            loss = loss + numpy.linalg.norm(dldx_traj) ** 2
            # chain rule
            for t in range(horizon):
                dauxvar += np.matmul(dldx_traj[t, :], dxdauxvar[t])
            dauxvar += np.matmul(dldx_traj[-1, :], dxdauxvar[-1])

        # take the expectation (average)
        dauxvar = dauxvar / n_batch
        loss = loss / n_batch

        return loss, dauxvar
