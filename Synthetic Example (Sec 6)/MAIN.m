% This file is an example of implementation of the d-CDP Algorithms 1 and 2 
%   and the d-DP algorithm for solving finite-horizon optimal control
%   problems. For more details, see Section 6 of the following article:
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
ProblemData.Stochastic = false;

% Output data
Result = struct;

% Optimal control problem instances - initial states
NumInstances = 100;
initial_state_set = cell(NumInstances,1); % the set of initial states (allocation)
Delta_x = [-1,1; -1,1]; % i-th row determines the range of i-th state
for i_Inst = 1:NumInstances
    initial_state_set{i_Inst} = Delta_x(:,1) + (Delta_x(:,2) - Delta_x(:,1)).*rand(size(Delta_x,1),1);
end
Result.InitialState = initial_state_set;

% DP and CDP configurations
dpData = struct;
dpData.ExtensionInterpolationMethod = 'linear';
dpData.ExtensionExtrapolationMethod = 'linear';

cdpData = struct;
cdpData.NumericalConjugateCost = false; 
cdpData.CoarseDualGridConstructionAlpha = 1;


%% Implemenatation 

% The reference costs in the error analysis (computed using d-DP)  

N_star = 81; % grid sizes 81^2 

% Problem data 

% Discretization of the state and input (and their dual) spaces
N = N_star*[1,1];
ProblemData.StateGridSize = N; % size of the grid X
ProblemData.InputGridSize = N; % size of the grid U

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

% Value iteration (backward iteration) using d-DP 
[Result.J_dp, Result.Pi_dp, Result.run_time_B_dp] = d_DP_Alg_1(ProblemData, dpData);
%----------------------------------------------------------------------

% Saving the results for each grid size 
str = sprintf('Ex_Data_N%2.0f',N(1));
save(str,'ProblemData','dpData','cdpData','Result')
%==========================================================================

% The three algorithms with different sizes of discretizations

N_set = [11 21 31 41]; % grid sizes 11^2, 21^2, ...

for i_N = 1:length(N_set) % Iteraton over grid sizes
    
    % Problem data 

    % Discretization of the state and input (and their dual) spaces
    N = N_set(i_N)*[1,1];
    ProblemData.StateGridSize = N; % size of the grid X
    ProblemData.InputGridSize = N; % size of the grid U
    ProblemData.StateDualGridSize = ProblemData.StateGridSize; % size of the grid Y
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
%==========================================================================

% d-CDP algorithm 1 with a smaller size for dual grid 

% Problem data 
% Discretization of the state and input (and their dual) spaces
ProblemData.StateGridSize = 41*[1,1]; % size of the grid X
ProblemData.InputGridSize = 41*[1,1]; % size of the grid U
ProblemData.StateDualGridSize = 21*[1,1]; % size of the grid Y

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

% Value iteration (backward iteration) using d-CDP Alg. 1 
[Result.J_cdp1, Result.run_time_B_cdp1] = d_CDP_Alg_1(ProblemData, cdpData);
%----------------------------------------------------------------------

% Solving the optimal control problem instances (forward iteration)
Result.x_cdp1_J = cell(NumInstances,1); Result.u_cdp1_J = cell(NumInstances,1); Result.traj_cost_cdp1_J = zeros(NumInstances,1); Result.run_time_F_cdp1_J = zeros(NumInstances,1);

for i_Inst = 1:NumInstances % Iteraton over control problem instances
    ProblemData.InitialState = Result.InitialState{i_Inst};
    [Result.x_cdp1_J{i_Inst},Result.u_cdp1_J{i_Inst},Result.traj_cost_cdp1_J(i_Inst),Result.run_time_F_cdp1_J(i_Inst)] = forward_iter_J_1(ProblemData, Result.J_cdp1);
end
%----------------------------------------------------------------------
    
% Saving the results for each grid size 
str = sprintf('Ex_Data_N41_2');
save(str,'ProblemData','cdpData','Result')
%----------------------------------------------------------------------


%% ANALYSIS: 

% The error in the costs-to-go for different grid sizes

% The reference (J_star): DP with high-resolution grid (N_star = 81) 
N_star = 81;
str = sprintf('Ex_Data_N%2.0f',N_star);
load(str)
X_star = ProblemData.StateGrid;
J_star = Result.J_dp;

% other local variables:
T = ProblemData.Horizon;
interpol = dpData.ExtensionInterpolationMethod;
extrapol = dpData.ExtensionExtrapolationMethod;

N_set = [11 21 31 41]; % grid sizes
% Data
error_dp_J = zeros(T,length(N_set)); % maximum absolute error between J_dp and J_star
error_cdp1_J = zeros(T,length(N_set)); 
error_cdp2_J = zeros(T,length(N_set)); 
for i_N = 1:length(N_set)
    
    N = N_set(i_N);
    str = sprintf('Ex_Data_N%2.0f',N);
    load(str)
    
    J_dp = Result.J_dp;
    J_cdp1 = Result.J_cdp1;
    J_cdp2 = Result.J_cdp2;
    
    X = ProblemData.StateGrid;

    for t = 1:T

        % Computing the reference via the "exact" implementation of DP
        idnt = @(x) (x);
        J_star_X_t = ext_constr(X_star,J_star{t},X,@(x) (x),@(x) (0),interpol,extrapol);
           
        % Computing errors
        temp = ( J_star_X_t ) - ( J_dp{t} );
        error_dp_J(t,i_N) = max(abs(temp(:)));
        temp = ( J_star_X_t ) - ( J_cdp1{t} );
        error_cdp1_J(t,i_N) = max(abs(temp(:)));
        temp = ( J_star_X_t ) - ( J_cdp2{t} );
        error_cdp2_J(t,i_N) = max(abs(temp(:)));

    end

end
%----------------------------------------------------------------------

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

load('Ex_Data_N41_2')
RunTime_cdp1_2 = sum(Result.run_time_B_cdp1) + Result.run_time_F_cdp1_J(1); % total running time of d-CDP Alg. 1 with smaller dual grid
ind = 1:100; temp = Result.traj_cost_cdp1_J(ind); 
Avg_Cost_cdp1_2 = mean(temp(~isnan(temp))); % cost of d-CDP Alg. 1 with smaller dual grid (using greedy actions)
%----------------------------------------------------------------------

% Saving the outputs of analyis
str = sprintf('Ex_Analysis');
save(str,'N_set','error_dp_J','error_cdp1_J','error_cdp2_J',...
    'RunTime_dp_J','RunTime_cdp1','RunTime_cdp2','RunTime_cdp1_2',...
    'Avg_Cost_dp_J','Avg_Cost_cdp1','Avg_Cost_cdp2','Avg_Cost_cdp1_2');

%% Plotting the outputs

% Plotting the errors
t_vec = 0:T;
h1 = figure;
subplot(1,2,1)
plot(t_vec,flip([error_dp_J(:,1);0]),'Color',[.0 .45 .74],'LineStyle','-','Marker','s','LineWidth',1)
hold on
plot(t_vec,flip([error_cdp1_J(:,1);0]),'Color',[0 0.5 0],'LineStyle','-','Marker','d','LineWidth',1)
plot(t_vec,flip([error_cdp2_J(:,1);0]),'Color',[.85 .33 .1],'LineStyle','-','Marker','o','LineWidth',1)
ylabel('$\Vert J_t - J^{\star}_t \Vert_{\infty}$','FontSize',10,'Interpreter','latex')
xlabel('$t$','FontSize',10,'Interpreter','latex')
xticks([0 5 10])
xticklabels({'10','5','0'})
txt1 = 'd-DP';
txt2 = 'd-CDP 1';
txt3 = 'd-CDP 2';
legend({txt1, txt2, txt3},'Interpreter','latex','Orientation','vertical','Location','northeast','FontSize',10)
ax = gca;
ax.Box = 'on';
ax.YGrid = 'on';
hold off
subplot(1,2,2)
hold on
plot(t_vec,flip([error_dp_J(:,4);0]),'Color',[.0 .45 .74],'LineStyle','-','Marker','s','LineWidth',1)
plot(t_vec,flip([error_cdp1_J(:,4);0]),'Color',[0 0.5 0],'LineStyle','-','Marker','d','LineWidth',1)
plot(t_vec,flip([error_cdp2_J(:,4);0]),'Color',[.85 .33 .1],'LineStyle','-','Marker','o','LineWidth',1)
ylabel('$\Vert J_t - J^{\star}_t \Vert_{\infty}$','FontSize',10,'Interpreter','latex')
xlabel('$t$','FontSize',10,'Interpreter','latex')
xticks([0 5 10])
xticklabels({'10','5','0'})
txt1 = 'd-DP';
txt2 = 'd-CDP 1';
txt3 = 'd-CDP 2';
legend({txt1, txt2, txt3},'Interpreter','latex','Orientation','vertical','Location','northeast','FontSize',10)
ax = gca;
ax.Box = 'on';
ax.YGrid = 'on';
hold off
%----------------------------------------------------------------------

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
txt2 = 'd-CDP 1';
txt3 = 'd-CDP 2';
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
txt2 = 'd-CDP 1';
txt3 = 'd-CDP 2';
legend({txt1, txt2, txt3},'Interpreter','latex','Orientation','vertical','Location','northwest','FontSize',10)
ax = gca;
ax.Box = 'on';
grid on
%----------------------------------------------------------------------

% Plotting the costs-to-go for specific N and t
N = 21;
str = sprintf('Ex_Data_N%2.0f',N);
load(str)
J_dp = Result.J_dp;
J_cdp1 = Result.J_cdp1;
J_cdp2 = Result.J_cdp2;
X = ProblemData.StateGrid;
h4 = figure;
subplot(2,4,1)
mesh(X_star{1},X_star{2},J_star{10})
xlabel('$x_1$','FontSize',10,'Interpreter','latex')
ylabel('$x_2$','FontSize',10,'Interpreter','latex')
zlabel('$J^{\star}_9$','FontSize',10,'Interpreter','latex')
subplot(2,4,2)
mesh(X{1},X{2},J_dp{10})
xlabel('$x_1$','FontSize',10,'Interpreter','latex')
ylabel('$x_2$','FontSize',10,'Interpreter','latex')
zlabel('$J^{DP}_{9}$','FontSize',10,'Interpreter','latex')
subplot(2,4,3)
mesh(X{1},X{2},J_cdp1{10})
xlabel('$x_1$','FontSize',10,'Interpreter','latex')
ylabel('$x_2$','FontSize',10,'Interpreter','latex')
zlabel('$J^{CDP1}_{9}$','FontSize',10,'Interpreter','latex')
subplot(2,4,4)
mesh(X{1},X{2},J_cdp2{10})
xlabel('$x_1$','FontSize',10,'Interpreter','latex')
ylabel('$x_2$','FontSize',10,'Interpreter','latex')
zlabel('$J^{CDP2}_{9}$','FontSize',10,'Interpreter','latex')
subplot(2,4,5)
mesh(X_star{1},X_star{2},J_star{1})
xlabel('$x_1$','FontSize',10,'Interpreter','latex')
ylabel('$x_2$','FontSize',10,'Interpreter','latex')
zlabel('$J^{\star}_{0}$','FontSize',10,'Interpreter','latex')
subplot(2,4,6)
mesh(X{1},X{2},J_dp{1})
xlabel('$x_1$','FontSize',10,'Interpreter','latex')
ylabel('$x_2$','FontSize',10,'Interpreter','latex')
zlabel('$J^{DP}_{0}$','FontSize',10,'Interpreter','latex')
subplot(2,4,7)
mesh(X{1},X{2},J_cdp1{1})
xlabel('$x_1$','FontSize',10,'Interpreter','latex')
ylabel('$x_2$','FontSize',10,'Interpreter','latex')
zlabel('$J^{CDP1}_{0}$','FontSize',10,'Interpreter','latex')
subplot(2,4,8)
mesh(X{1},X{2},J_cdp2{1})
xlabel('$x_1$','FontSize',10,'Interpreter','latex')
ylabel('$x_2$','FontSize',10,'Interpreter','latex')
zlabel('$J^{CDP2}_{0}$','FontSize',10,'Interpreter','latex')