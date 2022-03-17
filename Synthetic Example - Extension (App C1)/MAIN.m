% This file is an example of implementation of the "extended" d-CDP 
%    Algorithms 3 and 4 and the d-DP algorithm for solving finite-horizon
%    optimal control problems. For more details, see Appendix C.1 of the 
%    following article:
%
%                KOLARIJANI & MOHAJERIN ESFAHANI (2022), 
%    "FAST APPROXIMATE DYNAMIC PROGRAMMING FOR INPUT-AFFINE DYNAMICS"
%

clc
clear all

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
ProblemData.StateInputConstraints = @(x,u) (0); % state-dependent input constraints "C*x+D*u <= d"

% Cost functions
Q = [1 0;0 1]; ProblemData.StateCost = @(x) (x'*Q*x); % quadratic stage cost (state) 
ProblemData.InputCost = @(u) ( exp(abs(u(1))) + exp(abs(u(2))) - 2 ); % exponential stage cost (input)
ProblemData.StageCost = @(x,u) ( x'*Q*x + exp(abs(u(1)))+exp(abs(u(2)))-2 ); % quadratic state cost + exponential input cost
Q_T = [1 0;0 1]; ProblemData.TerminalCost = @(x) (x'*Q_T*x); % quadratic terminal cost

% Conjugate cost function
Delta_u = [-2,2; -2,2]; % i-th row determines the range of i-th input (input constraints)
ProblemData.ConjugateInputCost = @(v) (  conj_ExpL1_box(Delta_u(:,1),Delta_u(:,2),v) );
ProblemData.CojugateStageCost = @(x,v) ( -x'*Q*x +  conj_ExpL1_box(Delta_u(:,1),Delta_u(:,2),v) );

% Stochasticity
ProblemData.Stochastic = true; 

% Disturbance
W1 = [0 .1 -.1]; 
W = combvec(W1,W1); % discrete set of disturbance (column vectors)
ProblemData.DiscreteDisturbance = W;
ProblemData.DisturbancePMF = ones(1,size(W,2))/size(W,2); % disturbance probability mass func

% Output data
Result = struct;

% Optimal control problem instances - initial states
NumInstances = 100;

initial_state_set = cell(NumInstances,1); % the set of initial states (allocation)
disturb_seq_set = cell(NumInstances,1); % the set of disturbance eqauences (allocation)
Delta_x = [-1,1; -1,1]; % i-th row determines the range of i-th state
for i_Inst = 1:NumInstances
    initial_state_set{i_Inst} = Delta_x(:,1) + (Delta_x(:,2) - Delta_x(:,1)).*rand(size(Delta_x,1),1);
    ind_w_t = randsample(length(ProblemData.DisturbancePMF),ProblemData.Horizon,true,ProblemData.DisturbancePMF);
    disturb_seq_set{i_Inst} = ProblemData.DiscreteDisturbance(:,ind_w_t);
end

Result.InitialState = initial_state_set;
Result.DisturbanceSequence = disturb_seq_set;

% DP and CDP configurations
dpData = struct;
dpData.FastStochastic = false;
dpData.ExtensionInterpolationMethod = 'linear';
dpData.ExtensionExtrapolationMethod = 'linear';

cdpData = struct;
cdpData.NumericalConjugateCost = true;
cdpData.ExpectationInterpolationMethod = 'linear';
cdpData.ExpectationExtrapolationMethod = 'linear'; 
cdpData.CoarseDualGridConstructionAlpha = 1;


%% Implemenatation 

% The three algorithms with different sizes of discretizations

N_set = [11 21 31 41]; % grid sizes 11^2, 21^2, ...

for i_N = 1:length(N_set) % Iteraton over grid sizes
    
    % Problem data 

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

    % Feasibility check of discretization scheme 
    feasibility_check_2(ProblemData)
    %----------------------------------------------------------------------

    % Value iteration (backward iteration) 

    % DP implementation 
    [Result.J_dp, Result.Pi_dp, Result.run_time_B_dp] = d_DP_Alg_1(ProblemData, dpData); 

    % CDP2 implemetation 
    [Result.J_cdp2, Result.run_time_B_cdp2] = d_CDP_Alg_2(ProblemData, cdpData);
    
    % CDP1 implemetation 
    [Result.J_cdp1, Result.run_time_B_cdp1] = d_CDP_Alg_1(ProblemData, cdpData);
    %----------------------------------------------------------------------

    % Solving the optimal control problem instances (forward iteration)

    Result.x_dp_J = cell(NumInstances,1); Result.u_dp_J = cell(NumInstances,1); Result.traj_cost_dp_J = zeros(NumInstances,1); Result.run_time_F_dp_J = zeros(NumInstances,1);
    Result.x_cdp2_J = cell(NumInstances,1); Result.u_cdp2_J = cell(NumInstances,1); Result.traj_cost_cdp2_J = zeros(NumInstances,1); Result.run_time_F_cdp2_J = zeros(NumInstances,1);
    Result.x_cdp1_J = cell(NumInstances,1); Result.u_cdp1_J = cell(NumInstances,1); Result.traj_cost_cdp1_J = zeros(NumInstances,1); Result.run_time_F_cdp1_J = zeros(NumInstances,1);

    for i_Inst = 1:NumInstances % Iteraton over control problem instances

        ProblemData.DisturbanceSequence = Result.DisturbanceSequence{i_Inst};
        ProblemData.InitialState = Result.InitialState{i_Inst};

        % DP 
        [Result.x_dp_J{i_Inst},Result.u_dp_J{i_Inst},Result.traj_cost_dp_J(i_Inst),Result.run_time_F_dp_J(i_Inst)] = forward_iter_J_1(ProblemData, Result.J_dp); 
        
        % CDP2 
        [Result.x_cdp2_J{i_Inst},Result.u_cdp2_J{i_Inst},Result.traj_cost_cdp2_J(i_Inst),Result.run_time_F_cdp2_J(i_Inst)] = forward_iter_J_2(ProblemData, Result.J_cdp2);
        
        % CDP1 
        [Result.x_cdp1_J{i_Inst},Result.u_cdp1_J{i_Inst},Result.traj_cost_cdp1_J(i_Inst),Result.run_time_F_cdp1_J(i_Inst)] = forward_iter_J_1(ProblemData, Result.J_cdp1);

    end
    %----------------------------------------------------------------------
    
    % Saving the results for each grid size 
    str = sprintf('Ex_Data_N%2.0f',N(1));
    save(str,'ProblemData','dpData','cdpData','Result')

end

%% ANALYSIS: 

% Control Problem Instances: Time complexity and trajectory cost for
%      different grid sizes

% The outputs are: 
RunTime_dp_J = zeros(1,length(N_set)); % total running time of d-DP algorithm
RunTime_cdp1 = zeros(1,length(N_set));
RunTime_cdp2 = zeros(1,length(N_set));
Avg_Cost_dp_J = zeros(1,length(N_set)); % cost of d-DP algorithm (using greedy actions)
Avg_Cost_cdp1 = zeros(1,length(N_set));
Avg_Cost_cdp2 = zeros(1,length(N_set));

for i_N = 1:length(N_set)
    
    N = N_set(i_N);
    str = sprintf('Ex_Data_N%2.0f',N);
    load(str)
    
    % Computing the average of total running time over problem instances
    RunTime_dp_J(i_N) = Result.run_time_B_dp + Result.run_time_F_dp_J(1);
    RunTime_cdp1(i_N) = sum(Result.run_time_B_cdp1) + Result.run_time_F_cdp1_J(1);
    RunTime_cdp2(i_N) = sum(Result.run_time_B_cdp2) + Result.run_time_F_cdp2_J(1);
    
    ind = 1:100;
    % Computing the the average of relative trajectory costs over problem instances
    temp = Result.traj_cost_dp_J(ind); Avg_Cost_dp_J(i_N) = mean(temp(~isnan(temp)));
    temp = Result.traj_cost_cdp1_J(ind); Avg_Cost_cdp1(i_N) = mean(temp(~isnan(temp)));
    temp = Result.traj_cost_cdp2_J(ind); Avg_Cost_cdp2(i_N) = mean(temp(~isnan(temp)));
    
end 


% Saving the outputs of analyis
str = sprintf('Ex_Analysis');
save(str,'N_set','RunTime_dp_J','RunTime_cdp1','RunTime_cdp2',...
    'Avg_Cost_dp_J','Avg_Cost_cdp1','Avg_Cost_cdp2');

%% Plotting the outputs

% Plotting the runtimes
h2 = figure;
N_vec = (N_set.^2);
loglog(N_vec,RunTime_dp_J,'Color',[.0 .45 .74],'LineStyle','-','Marker','s','LineWidth',1)
hold on
loglog(N_vec,RunTime_cdp1,'Color',[0 0.5 0],'LineStyle','-','Marker','d','LineWidth',1)
loglog(N_vec,RunTime_cdp2,'Color',[.85 .33 .1],'LineStyle','-','Marker','o','LineWidth',1)
hold off
ylabel('Run-time (sec)','FontSize',10,'Interpreter','latex')
xlabel('Grid size, $N$','FontSize',10,'Interpreter','latex')
txt1 = 'd-DP';
txt2 = 'd-CDP 3';
txt3 = 'd-CDP 4';
legend({txt1, txt2, txt3},'Interpreter','latex','Orientation','vertical','Location','northwest','FontSize',10)
ax = gca;
ax.Box = 'on';
grid on
%----------------------------------------------------------------------

% Plotting the costs
h3 = figure;
N_vec = (N_set.^2);
loglog(N_vec,Avg_Cost_dp_J,'Color',[.0 .45 .74],'LineStyle','-','Marker','s','LineWidth',1)
hold on
loglog(N_vec,Avg_Cost_cdp1,'Color',[0 0.5 0],'LineStyle','-','Marker','d','LineWidth',1)
loglog(N_vec,Avg_Cost_cdp2,'Color',[.85 .33 .1],'LineStyle','-','Marker','o','LineWidth',1)
hold off
ylabel('Avg. cost','FontSize',10,'Interpreter','latex')
xlabel('Grid size, $N$','FontSize',10,'Interpreter','latex')
txt1 = 'd-DP';
txt2 = 'd-CDP 3';
txt3 = 'd-CDP 4';
legend({txt1, txt2, txt3},'Interpreter','latex','Orientation','vertical','Location','northwest','FontSize',10)
ax = gca;
ax.Box = 'on';
grid on

