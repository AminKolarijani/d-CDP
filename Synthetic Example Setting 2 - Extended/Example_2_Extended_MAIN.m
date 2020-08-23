% This file is an example of implementation of the extended d-CDP Algorithm 4 
%   for solving finite-horizon optimal control problems of SETTING 2. Please 
%   see Appendix C.1 of the article for more details:
%   
%    KOLARIJANI & MOHAJERIN ESFAHANI (2020), 
%      "FAST APPROXIMATE DYNAMIC PROGRAMMING FOR INPUT-AFFINE DYNAMICS"
% 

clc
clear all

%% Problem data

%% Problem data

ProblemData = struct;

% Dynamics: f(x,u) = Ax+Bu (linear dynamics)
A = [-.5 2;1 3]; 
B = [1 .5;1 1]; 
ProblemData.StateDynamics = @(x) (A*x); 
ProblemData.InputMatrix = B;
ProblemData.InputDynamics = @(x) (B); 
ProblemData.Dynamics = @(x,u) (ProblemData.StateDynamics(x) + ProblemData.InputDynamics(x) * u);

% Horizon
ProblemData.Horizon = 10;

% Constraints
ProblemData.StateConstraints = @(x) ([1 0;-1 0; 0 1; 0 -1]*x - ones(4,1)); % box [-1,1]^2
ProblemData.InputConstraints = @(u) ([1 0;-1 0; 0 1; 0 -1]*u - 2*ones(4,1)); % box [-2,2]^2

% Cost functions
Q = [1 0;0 1]; ProblemData.StateCost = @(x) (x'*Q*x); % quadratic stage cost (state) 
ProblemData.InputCost = @(u) ( exp(abs(u(1))) + exp(abs(u(2))) - 2 ); % exponential stage cost (input)
Q_T = [1 0;0 1]; ProblemData.TerminalCost = @(x) (x'*Q_T*x); % quadratic terminal cost

% Stochasticity
ProblemData.Stochastic = true; 

% Disturbance
W1 = [0 .1 -.1]; 
W = combvec(W1,W1); % discrete set of disturbance (column vectors)
ProblemData.DiscreteDisturbance = W;
ProblemData.DisturbancePMF = ones(1,size(W,2))/size(W,2); % disturbance probability mass function 

% Output data
Result = struct;

% Optimal control problem instances - data
load 'InititialState_DisturbanceSeq';
NumInstances = 10;
Result.DisturbanceSequence = DistSeq;
Result.InitialState = InitStat;

% DP and CDP configurations
dpData = struct;
dpData.FastStochastic = true;
dpData.ExtensionInterpolationMethod = 'linear';
dpData.ExtensionExtrapolationMethod = 'linear';

cdpData = struct;
cdpData.NumericalConjugateCost = true;
cdpData.ExpectationInterpolationMethod = 'linear';
cdpData.ExpectationExtrapolationMethod = 'linear'; 
cdpData.CoarseDualGridConstructionAlpha = 1;

%--------------------------------------------------------------------------
N_set = [11 21 41 81]; % grid sizes 11^2, 21^2, ...
N_star = 81; % for the reference in the error analysis
%--------------------------------------------------------------------------

%% Implementation 

for i_N = 1:length(N_set) % Iteraton over grid sizes
    
    % Problem data --------------------------------------------------------

    % Discretization of the state and input (and their dual) spaces
    N = N_set(i_N)*[1,1];
    ProblemData.StateGridSize = N; % size of the grid X
    ProblemData.InputGridSize = N; % size of the grid U
    ProblemData.StateDualGridSize = ProblemData.StateGridSize; % size of the grid Y
    ProblemData.InputDualGridSize = ProblemData.InputGridSize; % size of the grid V
    ProblemData.StateDynamicsGridSize = ProblemData.StateGridSize; % size of the grid Z

    % State space discretization (uniform grid)
    Delta_x = [-1,1; -1,1]; % i-th row determines the range of i-th state
    ProblemData.StateGrid = unif_grid(Delta_x,ProblemData.StateGridSize);
    
    % Input space discretization (uniform grid)
    Delta_u = [-2,2; -2,2]; % i-th row determines the range of i-th input
    ProblemData.InputGrid = unif_grid(Delta_u,ProblemData.InputGridSize);
    %----------------------------------------------------------------------

    % Feasibility check of discretization scheme --------------------------
    feasibility_check_2(ProblemData)
    %----------------------------------------------------------------------

    % Value iteration (backward iteration) --------------------------------

    % DP implementation 
    [Result.J_dp, Result.Pi_dp, Result.run_time_B_dp] = d_DP_Alg_2(ProblemData, dpData); 

    % CDP implemetation
    [Result.J_cdp, Result.run_time_B_cdp] = d_CDP_Alg_2(ProblemData, cdpData);
    %----------------------------------------------------------------------

    % Solving the optimal control problem instances (forward iteration)----

    Result.x_dp_J = cell(NumInstances,1); Result.u_dp_J = cell(NumInstances,1); Result.traj_cost_dp_J = zeros(NumInstances,1); Result.run_time_F_dp_J = zeros(NumInstances,1);
    Result.x_dp_Pi = cell(NumInstances,1); Result.u_dp_Pi = cell(NumInstances,1); Result.traj_cost_dp_Pi = zeros(NumInstances,1); Result.run_time_F_dp_Pi = zeros(NumInstances,1);
    Result.x_cdp_J = cell(NumInstances,1); Result.u_cdp_J = cell(NumInstances,1); Result.traj_cost_cdp_J = zeros(NumInstances,1); Result.run_time_F_cdp_J = zeros(NumInstances,1);

    for i_Inst = 1:NumInstances % Iteraton over control problem instances

        ProblemData.DisturbanceSequence = Result.DisturbanceSequence{i_Inst};
        ProblemData.InitialState = Result.InitialState{i_Inst};

        % DP 
        [Result.x_dp_J{i_Inst},Result.u_dp_J{i_Inst},Result.traj_cost_dp_J(i_Inst),Result.run_time_F_dp_J(i_Inst)] = forward_iter_J_2(ProblemData, Result.J_dp); 
        [Result.x_dp_Pi{i_Inst},Result.u_dp_Pi{i_Inst},Result.traj_cost_dp_Pi(i_Inst),Result.run_time_F_dp_Pi(i_Inst)] = forward_iter_Pi_2(ProblemData, Result.Pi_dp);

        % CDP 
        [Result.x_cdp_J{i_Inst},Result.u_cdp_J{i_Inst},Result.traj_cost_cdp_J(i_Inst),Result.run_time_F_cdp_J(i_Inst)] = forward_iter_J_2(ProblemData, Result.J_cdp);

    end
    %----------------------------------------------------------------------
    
    % Saving the results for each grid size -------------------------------
    str = sprintf('Ex_2_Extended_Data_N%2.0f',N(1));
    save(str,'ProblemData','dpData','cdpData','Result')
    %----------------------------------------------------------------------

end 
%% ANALYSIS: 

%==========================================================================

% The error in the costs-to-go for different grid sizes

% The reference (J_star): DP with high-resolution grid (N_star = 81) 
str = sprintf('Ex_2_Extended_Data_N%2.0f',N_star);
load(str)
X_star = ProblemData.StateGrid;
J_star = Result.J_dp;

J_star_instance = Result.traj_cost_dp_Pi;

% other local variables:
T = ProblemData.Horizon;
interpol = dpData.ExtensionInterpolationMethod;
extrapol = dpData.ExtensionExtrapolationMethod;

% Data
error_dp_J = zeros(T,length(N_set)); % maximum absolute error between J_dp and J_star
error_cdp_J = zeros(T,length(N_set)); % maximum absolute error between J_cdp and J_star

for i_N = 1:length(N_set)
    
    N = N_set(i_N);
    str = sprintf('Ex_2_Extended_Data_N%2.0f',N);
    load(str)
    
    J_dp = Result.J_dp;
    J_cdp = Result.J_cdp;
    
    X = ProblemData.StateGrid;

    for t = 1:T

        % Computing the reference via the "exact" implementation of DP
        idnt = @(x) (x);
        J_star_X_t = ext_constr(X_star,J_star{t},X,@(x) (x),@(x) (0),interpol,extrapol);
           
        % Computing errors
        temp = ( J_star_X_t ) - ( J_dp{t} );
        error_dp_J(t,i_N) = max(abs(temp(:)));
        temp = ( J_star_X_t ) - ( J_cdp{t} );
        error_cdp_J(t,i_N) = max(abs(temp(:)));

    end

end 

%==========================================================================

% Control Problem Instances: Time complexity and trajectory cost for
% different grid sizes 

% The outputs are: 
Avg_RunTime_dp_J = zeros(1,length(N_set)); % running time of d-DP algorithm, control generated using the optimal costs "J_dp"
Avg_RunTime_dp_Pi = zeros(1,length(N_set)); % running time of d-DP algorithm, control generated using the optimal policy "Pi_dp"
Avg_RunTime_cdp = zeros(1,length(N_set)); % running time of d-DP algorithm (control generated using the optimal costs "J_cdp")
Avg_Cost_dp_J = zeros(1,length(N_set));  %(relative) cost of d-DP algorithm, control generated using the optimal costs "J_dp"
Avg_Cost_dp_Pi = zeros(1,length(N_set)); %(relative) cost of d-DP algorithm, control generated using the optimal policy "Pi_dp"
Avg_Cost_cdp = zeros(1,length(N_set)); %(relative) cost of d-DP algorithm (control generated using the optimal costs "J_cdp")

for i_N = 1:length(N_set)
    
    N = N_set(i_N);
    str = sprintf('Ex_2_Extended_Data_N%2.0f',N);
    load(str)
    
    % Computing the average of total running time over problem instances
    Avg_RunTime_dp_J(i_N) = Result.run_time_B_dp + mean(Result.run_time_F_dp_J);
    Avg_RunTime_dp_Pi(i_N) = Result.run_time_B_dp + mean(Result.run_time_F_dp_Pi);
    Avg_RunTime_cdp(i_N) = sum(Result.run_time_B_cdp) + mean(Result.run_time_F_cdp_J);
    
    ind = 1:10;
    % Computing the the average of relative trajectory costs over problem instances
    temp = Result.traj_cost_dp_J(ind)./J_star_instance(ind); Avg_Cost_dp_J(i_N) = mean(temp(~isnan(temp)));
    temp = Result.traj_cost_dp_Pi(ind)./J_star_instance(ind); Avg_Cost_dp_Pi(i_N) = mean(temp(~isnan(temp)));
    temp = Result.traj_cost_cdp_J(ind)./J_star_instance(ind); Avg_Cost_cdp(i_N) = mean(temp(~isnan(temp)));
    
end 

%==========================================================================

% Saving the outputs:
str = sprintf('Ex_2_Extended_Analysis');
save(str,'N_set','error_dp_J','error_cdp_J',...
    'Avg_RunTime_dp_J','Avg_RunTime_dp_Pi','Avg_RunTime_cdp',...
    'Avg_Cost_dp_J','Avg_Cost_dp_Pi','Avg_Cost_cdp');
