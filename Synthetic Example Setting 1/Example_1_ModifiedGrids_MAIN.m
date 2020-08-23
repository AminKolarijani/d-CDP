% This file is an example of implementation of the d-CDP Algorithm 1 for 
%   solving finite-horizon optimal control problems of SETTING 1. Please 
%   see Section 6.1 of the article for more details:
%   
%    KOLARIJANI & MOHAJERIN ESFAHANI (2020), 
%      "FAST APPROXIMATE DYNAMIC PROGRAMMING FOR INPUT-AFFINE DYNAMICS"
% 
clc
clear all

%% Problem data

ProblemData = struct;

% Dynamics:
A = [-.5 2;1 3]; 
B = [1 .5;1 1]; 
ProblemData.StateDynamics = @(x) (A*x); 
ProblemData.InputDynamics = @(x) (B); 
ProblemData.Dynamics = @(x,u) (ProblemData.StateDynamics(x) + ProblemData.InputDynamics(x) * u);

% Horizon
ProblemData.Horizon = 10;

% Constraints
ProblemData.StateConstraints = @(x) ([1 0;-1 0; 0 1; 0 -1]*x - ones(4,1)); % box [-1,1]^2
ProblemData.InputConstraints = @(u) ([1 0;-1 0; 0 1; 0 -1]*u - 2*ones(4,1)); % box [-2,2]^2
C = [1 0;0 1]; D = [1 0; 0 1]; d = [2; 2];  
ProblemData.StateInputConstraints = @(x,u) (C*x+D*u-d); % state-dependent input constraints "C*x+D*u <= d"

% Cost functions
Q = [1 0;0 1]; ProblemData.StageCost = @(x,u) ( x'*Q*x + exp(abs(u(1)))+exp(abs(u(2)))-2 ); % quadratic state cost + exponential input cost
Q_T = [1 0;0 1]; ProblemData.TerminalCost = @(x) (x'*Q_T*x); % quadratic terminal cost

% Conjugate cost functions
Delta_u = [-2,2; -2,2]; % i-th row determines the range of i-th input (input constraints)
ProblemData.CojugateStageCost = @(x,v) ( -x'*Q*x + conj_ExpL1_box(Delta_u(:,1),min(Delta_u(:,2),d-x),v) );

% Stochasticity
ProblemData.Stochastic = false;

% Output data
Result = struct;

% Optimal control problem instances - data
load 'InititialState_DisturbanceSeq';
NumInstances = 10;
Result.InitialState = InitStat;

% DP and CDP configurations
dpData = struct;
dpData.ExtensionInterpolationMethod = 'linear';
dpData.ExtensionExtrapolationMethod = 'linear';

cdpData = struct;
cdpData.NumericalConjugateCost = false;   
cdpData.CoarseDualGridConstructionAlpha = 1;

%--------------------------------------------------------------------------
N_set = [11 7 11; 21 11 21; 41 21 41;...
    11 7 21; 21 11 41; 41 21 81]; % grid sizes
N_star = 81; % for the reference in the error analysis
%--------------------------------------------------------------------------

%% Implementation

for i_N = 1:size(N_set,1) % iteraton over grid sizes
    
    % Problem data --------------------------------------------------------
    
    % Discretization of the state and input (and their dual) spaces
    ProblemData.StateGridSize = N_set(i_N,1)*[1,1]; % size of the grid X
    ProblemData.InputGridSize = N_set(i_N,3)*[1,1]; % size of the grid U
    ProblemData.StateDualGridSize = N_set(i_N,2)*[1,1]; % size of the grid Y
    
    % State space discretization (uniform grid)
    Delta_x = [-1,1; -1,1]; % i-th row determines the range of i-th state
    ProblemData.StateGrid = unif_grid(Delta_x,ProblemData.StateGridSize);
    
    % Input space discretization (uniform grid)
    Delta_u = [-2,2; -2,2]; % i-th row determines the range of i-th input
    ProblemData.InputGrid = unif_grid(Delta_u,ProblemData.InputGridSize);
    %----------------------------------------------------------------------
    
    % Feasibility check of discretization scheme --------------------------
    feasibility_check_1(ProblemData)
    %----------------------------------------------------------------------

    % Value iteration (backward iteration) --------------------------------

    % CDP implemetation
    [Result.J_cdp, Result.run_time_B_cdp] = d_CDP_Alg_1(ProblemData, cdpData);
    %----------------------------------------------------------------------
    
    % Solving the optimal control problem instances (forward iteration)----

    Result.x_cdp_J = cell(NumInstances,1); 
    Result.u_cdp_J = cell(NumInstances,1); 
    Result.traj_cost_cdp_J = zeros(NumInstances,1); 
    Result.run_time_F_cdp_J = zeros(NumInstances,1);

    for i_Inst = 1:NumInstances % iteraton over control problem instances

        ProblemData.InitialState = Result.InitialState{i_Inst};

        % CDP 
        [Result.x_cdp_J{i_Inst},Result.u_cdp_J{i_Inst},Result.traj_cost_cdp_J(i_Inst),Result.run_time_F_cdp_J(i_Inst)] = forward_iter_J_1(ProblemData, Result.J_cdp);

    end
    %----------------------------------------------------------------------

    % Saving the results for each grid size -------------------------------
    str = sprintf('Ex_1_ModifiedGrids_Data_%1.0f',i_N);
    save(str,'ProblemData','dpData','cdpData','Result')
    %----------------------------------------------------------------------
    
end 

%% Analysis

%==========================================================================

% Control Problem Instances: Time complexity and trajectory cost for 
% different grid sizes

N_num = size(N_set,1);

% The reference (J_star): DP with high-resolution grid (N_star = 81) 
str = sprintf('Ex_1_Data_N%2.0f',N_star);
load(str)
X_star = ProblemData.StateGrid;
J_star = Result.J_dp;
J_star_instance = Result.traj_cost_dp_Pi; 
 
% The outputs are: 
Avg_RunTime_cdp = zeros(1,N_num); % running time of d-DP algorithm (control generated using the optimal costs "J_cdp")
Avg_Cost_cdp = zeros(1,N_num); %(relative) cost of d-DP algorithm (control generated using the optimal costs "J_cdp")

for i_N = 1:N_num
    
    str = sprintf('Ex_1_ModifiedGrids_Data_%1.0f',i_N);
    load(str)
    
    Avg_RunTime_cdp(i_N) = sum(Result.run_time_B_cdp) + mean(Result.run_time_F_cdp_J);
    
    ind = 1:10;
    temp = Result.traj_cost_cdp_J(ind)./J_star_instance(ind); Avg_Cost_cdp(i_N) = mean(temp(~isnan(temp)));
    
end

%==========================================================================

% Saving the outputs:
str = sprintf('Ex_1_ModifiedGrids_Analysis');
save(str,'N_set','Avg_RunTime_cdp','Avg_Cost_cdp');

