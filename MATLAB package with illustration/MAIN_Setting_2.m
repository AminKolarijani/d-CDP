% This is an illustration of the usage of the developed MATLAB pakage for
%   solving finite-horizon optimal control problems of SETTING 2 using the 
%   d-CDP ALGORTIHM 2. Also included is the implementations of the 
%   corresponding d-DP algorithm. 
%   Please see the following article for more details:
%   
%    KOLARIJANI & MOHAJERIN ESFAHANI (2020), 
%      "FAST APPROXIMATE DYNAMIC PROGRAMMING FOR INPUT-AFFINE DYNAMICS"
%

clc
clear all

%% (1) Problem data (inputs given by the user) 

%--------------------------------------------------------------------------
% Dynamics: f(x,u) = f_s(x) + B*u [input-affine with possibly nonlinear 
%   state dynamics]. Provide the functaion handle for the state dynamics, 
%   and the matrix B for the input dynamics. E.g., for the linear dyanmics 
%   f(x,u) = A*x + B*u:
A = [-.5 2;1 3]; state_dynamics = @(x) (A*x); 
B = [1 .5;1 1]; input_matrix = B;
%--------------------------------------------------------------------------
% Horizon
T = 10; 
%--------------------------------------------------------------------------
% Constraints: provided by function handles as "x_constraints(x) <= 0" for 
%   state constraints, as "u_constraints(u) <= 0" for input constraints.
%   E.g.:
x_constraints = @(x) ([1 0;-1 0; 0 1; 0 -1]*x - ones(4,1)); % state constraints: x \in [-1,1]^2
u_constraints = @(u) ([1 0;-1 0; 0 1; 0 -1]*u - 2*ones(4,1)); % input constraints: u \in [-2,2]^2
%--------------------------------------------------------------------------
% Cost functions: Provide the stage and terminal cost functions by the 
%   proper function handels.  In particular, provide the state-dependent
%   input-dependent part of the stage cost separately. E.g., 
state_stage_cost = @(x) (x'*x); % quadratic stage cost (state)
input_stage_cost = @(u) (u'*u); % quadratic stage cost (input)
term_cost = @(x) (x'*x); % quadratic terminal cost
%--------------------------------------------------------------------------
% Conjugate of input-dependent stage cost: First, determine if the 
%   conjugate is to be computed numerically or availabe analytically:
Num_Conj = true;
%   If "Num_Conj == false", provide the function handle for the conjugate 
%   of the input-dependent stage cost. E.g., for the quadratic cost given 
%   above, we have (notice that the conjugation takes into account the 
%   input constraints)
if ~Num_Conj
    Delta_u = [-2,2; -2,2]; % i-th row determines the range of i-th input
    conj_input_cost = @(v) (conj_Quad_box(eye(2),Delta_u(:,1),Delta_u(:,2),v));
end
%--------------------------------------------------------------------------
% Stochasticity 
%
%   NOTE: The current implementation of the proposed algorithms can handle 
%      addtitive distrurbance of the forrm "x^+ = f(x,u)+w", where the 
%      disturbance "w" belongs to a FINITE set "W" with a given probability
%      mass function (p.m.f). Then, the expectation with respect to "w" is 
%      simply a weighted average. We note that the algorithm can be easily 
%      modified to handle other forms of approximate expectation (e.g., 
%      Monte Carlo). To do so, one must update the MATLAB function 
%                              "ext_constr_expect.m"
%      accordingly. 
%      
%   First, determine if the dynamics is stochastic or deterministic: 
Stoch = ture;
%   If "Stoch == true", the user must also provide the finite set of 
%   distrurabnces along with the corresponding probability mass function:
if Stoch
    W1 = [0 .1 -.1]; W = combvec(W1,W1); % discrete set of disturbance (column vectors)
    pmf_W = ones(1,size(W,2))/size(W,2); % disturbance probability mass function
end
%--------------------------------------------------------------------------
% Optimal control problem instances (initial states): provide THE SET of
%   initial states for which the optimal control porblem is to be solved, 
%   as a CELL array; e.g., here we consider two instances randomly chosen 
%   from the provided state space X = [-1,1]^2:
NumInstances = 2; % the number of instances (initial states)
initial_state_set = cell(NumInstances,1); % the set of initial states (allocation)
Delta_x = [-1,1; -1,1]; % i-th row determines the range of i-th state
for i_Inst = 1:NumInstances
    initial_state_set{i_Inst} = Delta_x(:,1) + (Delta_x(:,2) - Delta_x(:,1)).*rand(size(Delta_x,1),1);
end

%   If the system is stochastic, then the user must also define a sequence 
%   of disturbances for the forward simulation of the system. The default 
%   mode is to generate a sample sequance using the provided data on the
%   disturbance (i.e., the disturbance set "W" and its p.m.f.)
if Stoch
    disturb_seq = cell(NumInstances,1);
    for i_Inst = 1:NumInstances
        ind_w_t = randsample(length(pmf_W),T,true,pmf_W);
        disturb_seq{i_Inst} = W(:,ind_w_t);
    end
end 
%--------------------------------------------------------------------------
% Discretization of the state and input (and their dual) spaces: Provide 
%   the GRID-LIKE discretizations of the state and input spaces, as "CELLS".
%   E.g., the state space grid is a cell of size (dimension of x, 1) with
%   each component of the cell inclduing a column vector corresponding to 
%   the discrete points along that dimension of the state space. 

%   NOTE: Here we consider UNIFORM grids for the discritization of the 
%      state and input spaces, but this need not be the case in general.

%   State space discretization (uniform grid)
N_x = [11, 11]; % vector of number of data points in the discretization of state space in EACH dimension
Delta_x = [-1,1; -1,1]; % i-th row determines the range of i-th state
state_grid = unif_grid(Delta_x,N_x); % so the discretization of the state space is a uniform grid of size 11^2 over [-1,1]^2 

%   Input space discretization (uniform grid) - similar to state space discretization described above
N_u = [11, 11]; 
Delta_u = [-2,2; -2,2]; 
input_grid = unif_grid(Delta_u,N_u); 

%   NOTE: The discrete sets X and U must satisfy the feasibility condition, 
%      i.e., for each x in X, there must exist u in U, such that Ax+Bu 
%      satisfies the state constraints. You will recieve a warning if this 
%      is not the case, with an example of a state for which the 
%      feasibility condition is not satisfied.

%   The state dual grid (Y_g):
%     Here, we set the the SIZE of the dual grid Y equal to the size of 
%     the state grid, but this need not be the case in general.
N_y = N_x; 
%   The grid (Z_g):
%     Here, we set the the SIZE of the grid Z equal to the size of 
%     the state grid, but this need not be the case in general.
N_z = N_x;
%   Set the value of the coefficeint "alpha" for the construction of the
%     state dual grid (please consult the Section 4.2.2 of the manuscript 
%     for more details. In particular, take into account the dimension of 
%     the state space in setting the value of alpha. As a rule of thumb, 
%     alpha should be inversly related to the dimension of the state space.
alpha = 1;

%   Size of the input DUAL grid: If the conjugate of the input stage cost 
%      is to be is to be computed numerically ("Num_Conj == true"), then
%      the user must determine the size of the input dual grids V.
%      Here, we again set the the size of the dual grids V equal to the 
%      size of the input grid, but this need not be the case in general.
if Num_Conj
    N_v = N_u;
end

%   NOTE the followings:
%
% (a) The current implementation of the d-CDP algorithms considers UNIFORM
%     grids for the discretization of the dual spaces. However, in general, 
%     this need not be the case. In order to modify the current 
%     implementation for NON-UNIFORM grid-like discretization of the dual
%     spaces, one needs to modify the correspondig parts of the following
%     MATLAM function
%                               "d_CDP_Alg_2.m".
%  
% (b) The vectors "N_x, N_u, N_y, N_z, N_v" determine the number of points in 
%     the grid-like discretization for EACH dimnesion. So, the grid size is
%     equal to the product of entries of these vecotrs. Particularly, avoid
%     using a large numbers for the d-DP algorithm. Please check the time 
%     complexities of the d-DP and d-CDP algorithms given, e.g., in 
%     Remark 6.1 of the manuscript. 
%--------------------------------------------------------------------------
% Configuration of the d-DP and d-CDP algorithms
%   Determine if you want to also run the d-DP algorithm, e.g., in order
%   to compare the performance of the two algorithms
DP_implementation = true;
%   If "dp_implementation = true" and "Stoch == true" (i.e., the system is
%   stochastic), then the user has the option to use the fast 
%   approximation of the stochastic DP opearation; see Section 4.3.1 of 
%   the manuscript for more details. Use the following to do so:
if DP_implementation && Stoch
    fast_stoch_DP = true;
end

%   NOTE: Considering grid-like discretization of the state space in the
%      d-DP algorithm, the extension operation are handeled via the MATALB
%      function 
%                           "griddedInterpolant". 
%      In particular, both the interpolation and extrapolation methods are 
%      set to "linear" (see below). However, the user can use other methods
%      that come with MATLAB function "griddedInterpolant". 
%      You can use the following lines to modify the methods for 
%      interpolation and extrapolation:
if DP_implementation 
    DP_interpol_mehtod = 'linear';
    DP_extrapol_mehtod = 'linear';
end

%   NOTE: If the dynamics is stochastic, then the d-CDP algorithm also uses
%      extension of the cost function for computing expectation. Once again, 
%      since we use grid-like discretization of the state space the 
%      extension operation uses the MATALB function "griddedinterpolant". 
%      The defualt mode for interpolation and extrapolation methods are 
%      set to "linear" (see below). However, the user can use other methods
%      that come with MATLAB function "griddedinterpolant" to modify them: 

if Stoch 
    CDP_interpol_mehtod = 'linear';
    CDP_extrapol_mehtod = 'linear';
end

%% (2) Problem data (data objects defined for internal use)

ProblemData = struct;

% Dynamics
ProblemData.StateDynamics = state_dynamics;
ProblemData.InputMatrix = input_matrix;
ProblemData.InputDynamics = @(x) (input_matrix); 
ProblemData.Dynamics = @(x,u) (ProblemData.StateDynamics(x) + ProblemData.InputDynamics(x) * u);

% Horizon
ProblemData.Horizon = T; 

% Constraints
ProblemData.StateConstraints = x_constraints; 
ProblemData.InputConstraints = u_constraints; 

% Cost functions and conjugates
ProblemData.StateCost = state_stage_cost;
ProblemData.InputCost = input_stage_cost;
ProblemData.TerminalCost = term_cost; 
if ~Num_Conj 
    ProblemData.ConjugateInputCost = conj_input_cost; 
end

% Disturbance
ProblemData.Stochastic = Stoch;
if Stoch
    ProblemData.DiscreteDisturbance = W; 
    ProblemData.DisturbancePMF = pmf_W; 
end

% Discretizations
ProblemData.StateGridSize = N_x; 
ProblemData.InputGridSize = N_u; 
ProblemData.StateDualGridSize = N_y;
ProblemData.StateDynamicsGridSize = N_z;
if Num_Conj
 ProblemData.InputDualGridSize = N_v; 
end

ProblemData.StateGrid = state_grid;
ProblemData.InputGrid = input_grid;

% Configuration of d-DP and d-CDP
if DP_implementation
    dpData = struct;
    if Stoch
        dpData.FastStochastic = fast_stoch_DP ;
    end
    dpData.ExtensionInterpolationMethod = DP_interpol_mehtod;
    dpData.ExtensionExtrapolationMethod = DP_extrapol_mehtod;
end

cdpData = struct;
cdpData.NumericalConjugateCost = Num_Conj;
cdpData.CoarseDualGridConstructionAlpha = alpha;
if Stoch 
    cdpData.ExpectationInterpolationMethod = CDP_interpol_mehtod;
    cdpData.ExpectationExtrapolationMethod = CDP_extrapol_mehtod;
end
 

%% (3) Implementation and Results 
%      NOTE: ALL the outputs are available in the data structure "Result".

%  Output data
Result = struct;
Result.InitialState = initial_state_set;
if Stoch
    Result.DisturbanceSequence = disturb_seq; 
end

%--------------------------------------------------------------------------

% Feasibility check of discretization scheme: check your command window
%   for possile warnings; the algorithm works even if the discretization of
%   the state and input spaces does not satisfy the feasibility condition
feasibility_check_2(ProblemData)

%--------------------------------------------------------------------------

% d-CDP Algorithm 2
%
%   (1) Backward Value Iteration: The outputs are
%      (a) Result.J_cdp: cell of size (T+1,1) for the discrete cost-to-go
%          functions over the discrete state space for each step in the 
%          horizon
%      (b) Result.run_time_B_cdp: the run-time of the algorithm (in sec)

[Result.J_cdp, Result.run_time_B_cdp] = d_CDP_Alg_2(ProblemData, cdpData);
Result.run_time_B_cdp = sum(Result.run_time_B_cdp);

%   (2) Optimal Control Problem (forward iteration for solving the given 
%       instances): The outputs are
%      (a) Result.x_cdp: cell of size (NumInstances,1) for the state trajectiory 
%          for each instance (initial state)
%      (b) Result.u_cdp: cell of size (NumInstances,1) for the optimal input  
%          sequence for each instance (initial state)
%      (c) Result.traj_cost_cdp: vector of size (NumInstances,1) for the total
%          cost of the generated trajectory using optimal input for each instance
%      (d) Result.run_time_F_cdp: vector of size (NumInstances,1) for the
%          run-time of the algorithm in forward iterations or each instance

Result.x_cdp_J = cell(NumInstances,1); Result.u_cdp_J = cell(NumInstances,1); 
Result.traj_cost_cdp_J = zeros(NumInstances,1); Result.run_time_F_cdp_J = zeros(NumInstances,1);

for i_Inst = 1:NumInstances % Iteraton over control problem instances
    
    ProblemData.InitialState = Result.InitialState{i_Inst}; 
    if Stoch
        ProblemData.DisturbanceSequence = Result.DisturbanceSequence{i_Inst}; 
    end
    
    [Result.x_cdp_J{i_Inst},Result.u_cdp_J{i_Inst},Result.traj_cost_cdp_J(i_Inst),Result.run_time_F_cdp_J(i_Inst)] = forward_iter_J_2(ProblemData, Result.J_cdp);

end

%--------------------------------------------------------------------------

if DP_implementation

% d-DP Algorithm (for Setting 2)
%
%   (1) Backward Value Iteration: The outputs are
%      (a) Result.J_dp: cell of size (T+1,1) for the discrete cost-to-go
%          functions over the discrete state space for each step in the 
%          horizon
%      (b) Result.Pi_dp: cell of size (T+1,1) for the discrete optimal
%          control laws over the discrete state space for each step in the 
%          horizon
%      (c) Result.run_time_B_dp: the run-time of the algorithm (in sec) 

[Result.J_dp, Result.Pi_dp, Result.run_time_B_dp] = d_DP_Alg_2(ProblemData, dpData); 

%   (2) Optimal Control Problem (forward iteration for solving the given 
%       instances): The outputs are
%      (a) Result.x_*: cell of size (NumInstances,1) for the state trajectiory 
%          for each instance (initial state)
%      (b) Result.u_*: cell of size (NumInstances,1) for the optimal input  
%          sequence for each instance (initial state)
%      (c) Result.traj_cost_*: vector of size (NumInstances,1) for the total
%          cost of the generated trajectory using optimal input for each instance
%      (d) Result.run_time_F_*: vector of size (NumInstances,1) for the
%          run-time of the algorithm in forward iterations or each instance
%   where the "*" is (see Section 6.1 of the manuscript for more details)
%      (a) dp_J: for the d-DP algorithm, with the control input computed 
%          using the optimal cost-to-go functions derived from value iteration
%      (b) dp_Pi: for the d-DP algorithm, with the control input computed 
%          using the optimal control laws derived from value iteration

Result.x_dp_J = cell(NumInstances,1); Result.u_dp_J = cell(NumInstances,1); 
Result.traj_cost_dp_J = zeros(NumInstances,1); Result.run_time_F_dp_J = zeros(NumInstances,1);
Result.x_dp_Pi = cell(NumInstances,1); Result.u_dp_Pi = cell(NumInstances,1); 
Result.traj_cost_dp_Pi = zeros(NumInstances,1); Result.run_time_F_dp_Pi = zeros(NumInstances,1);

for i_Inst = 1:NumInstances % Iteraton over control problem instances
    
    ProblemData.InitialState = Result.InitialState{i_Inst};
    if Stoch
        ProblemData.DisturbanceSequence = Result.DisturbanceSequence{i_Inst}; 
    end
    
    [Result.x_dp_J{i_Inst},Result.u_dp_J{i_Inst},Result.traj_cost_dp_J(i_Inst),Result.run_time_F_dp_J(i_Inst)] = forward_iter_J_2(ProblemData, Result.J_dp); 
    [Result.x_dp_Pi{i_Inst},Result.u_dp_Pi{i_Inst},Result.traj_cost_dp_Pi(i_Inst),Result.run_time_F_dp_Pi(i_Inst)] = forward_iter_Pi_2(ProblemData, Result.Pi_dp);

end

end



