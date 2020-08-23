function [x,u,traj_cost,run_time] = forward_iter_J_1(ProblemData, J_t)
% This function solves an instance (determined by the given initial state) 
% of optimal control problem of 
%                              "SETTING 1"
% using the optimal cost functions "J_t" given over the discrete state 
% space, via a brute force minimization over the discrete input space.
% Please see the following article for more details:
%   
%    KOLARIJANI & MOHAJERIN ESFAHANI (2020), "FAST DYNAMIC PROGRAMMING: 
%    APPLICATION OF CONJUGATE DUALITY IN APPROXIMATE DYNAMIC PROGRAMMING"
%
% Inputs: 
%   Data structure containing the problem data (see the block "local variables" below)
%   J_t: cell of size (T+1,1) for the discrete optimal costs-to-go over the 
%        discrete state space for each step in the horizon
%
% Outputs: 
%   x: vector of size (n_x,T+1) for the controlled state trajectiory 
%   u: vector of size (n_u,T) for the optimal input sequence 
%   traj_cost: a scalar for the total cost of the generated trajectory
%   run_time: a scalar for the run-time (in seconds) of solving the
%             minimization probelms in the forward iteration
%
% NOTE: The extension operation is handeled via the MATALB function 
%                           "griddedInterpolant". 
%      In particular, both the interpolation and extrapolation methods are 
%      set to "linear" (see line 58). However, the user can use other methods
%      that come with MATLAB function "griddedInterpolant".
%

%==========================================================================

% local variables (begins) ------------------------------------------------
dyn = ProblemData.Dynamics;

T = ProblemData.Horizon; 

cost = ProblemData.StageCost;
cost_T = ProblemData.TerminalCost; 
constr_x = ProblemData.StateConstraints;
constr_u = ProblemData.InputConstraints;
constr_xu = ProblemData.StateInputConstraints;

X = ProblemData.StateGrid;
U = ProblemData.InputGrid;

x_0 = ProblemData.InitialState;

stoch = ProblemData.Stochastic;
if stoch
    W = ProblemData.DiscreteDisturbance;
    P = ProblemData.DisturbancePMF;
    w_t = ProblemData.DisturbanceSequence;
end
% local variables (ends) --------------------------------------------------

% interpolation and extrapolation methods
interpol = 'linear';
extrapol = 'linear';

% constraints
feasibility_t = @(x,u)  max( max(max(constr_x(x)) , max(constr_u(u)) ), max(constr_xu(x,u)) );

n_x = length(x_0); % dimension of the state space
n_u = size(U,1); % dimension of the input space

% allocations
x = zeros(n_x,T+1); % state trajectory
u = zeros(n_u,T); % control input
x(:,1) = x_0;
run_time = 0;
traj_cost = 0;

%==========================================================================

% forward iteration

for t = 1:T
   
    tic % minimization time (begins)
    % minimization over u -------------------------------------------------
    % computing the LERP of the cost-to-go J at (x,u) for u \in U
    x_t = x(:,t);
    if stoch 
        J_at_U = ext_constr_expect(X,J_t{t+1},U,@(u) (dyn(x_t,u)),constr_x,W,P,interpol,extrapol);
    else
        J_at_U = ext_constr(X,J_t{t+1},U,@(u) (dyn(x_t,u)),constr_x,interpol,extrapol);
    end
       
    cost_at_U = eval_func_constr(@(u) (cost(x_t,u)), U, @(u) feasibility_t(x_t,u));
    
    Q = cost_at_U{1} + J_at_U;
    [dummy, temp2] = min(Q(:));
    ind_opt = cell(1,n_u);
    [ind_opt{:}] = ind2sub(size(Q),temp2);
    for i=1:n_u
        u(i,t) = U{i}(ind_opt{i});
    end
    % ---------------------------------------------------------------------
    run_time = run_time + toc; % minimization time (ends)
    
    % applying the control input
    if stoch
        x(:,t+1) = dyn(x(:,t),u(:,t))+w_t(:,t);
    else
        x(:,t+1) = dyn(x(:,t),u(:,t));
    end
    
    if (feasibility_t(x(:,t),u(:,t)) <=0)
        cost_t = cost(x(:,t),u(:,t));
    else
        cost_t = inf;
    end
    traj_cost = traj_cost + cost_t;
end

if (constr_x(x(:,T+1)) <=0) 
    cost_Tplus1 = cost_T(x(:,T+1));
else
    cost_Tplus1 = inf;
end
traj_cost = traj_cost + cost_Tplus1;

