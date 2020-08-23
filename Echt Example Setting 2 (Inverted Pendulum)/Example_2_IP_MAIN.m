% This file is an example of implementation of the extended d-CDP Algorithm 4 
%   for solving finite-horizon optimal control problems of a noisy inverted 
%   pendulum. Please see Appendix C.2.2 of the article for more details:
%   
%    KOLARIJANI & MOHAJERIN ESFAHANI (2020), 
%      "FAST APPROXIMATE DYNAMIC PROGRAMMING FOR INPUT-AFFINE DYNAMICS"
% 

clc
clear all

%% Problem data

ProblemData = struct;

% Dynamics - inverted pendulum
%    parameters -----------------------------------------------------------
J1 = 1.91e-4; m = .055; g = 9.81; el = .042; b = 3.0e-6; K = 53.6e-3; R = 9.50; tau = 0.05;
alpha = m*g*el/J1; beta = -(b+K^2/R)/J1; gamma = K/(J1*R);
%--------------------------------------------------------------------------
B = tau*[0;gamma]; 
ProblemData.InputMatrix = B;
ProblemData.StateDynamics = @(x) ( x + tau * [x(2); (alpha*sin(x(1)) + beta*x(2)) ]); 
ProblemData.InputDynamics = @(x) (B); 
ProblemData.Dynamics = @(x,u) (ProblemData.StateDynamics(x) + ProblemData.InputDynamics(x) * u);

% Horizon
ProblemData.Horizon = 50;

% Constraints
theta_lim = pi/4; thetadot_lim = pi; u_lim = 3; % state and input variables limits
ProblemData.StateConstraints = @(x) ([1 0; -1 0; 0 1; 0 -1]*x - [theta_lim; theta_lim; thetadot_lim; thetadot_lim]); %  the box [-pi/4,pi/4] * [-pi,pi]
ProblemData.InputConstraints = @(u) ([1;-1]*u - u_lim*ones(2,1)); % interval [-3,3]

% Cost functions
ProblemData.TerminalCost = @(x) (x'*x); % quadratic terminal cost (state)
ProblemData.InputCost = @(u) (u'*u); % quadratic stage cost (input) weight matrix
ProblemData.StateCost = @(x) (x'*x); % quadratic stage cost (state) weight matrix

% Conjugate cost function
ProblemData.ConjugateInputCost = @(v) ( conj_Quad_box(1,-u_lim,u_lim,v) );

% Stochasticity
ProblemData.Stochastic = true; % the dynamics is stochastic (with additive disturbance)

% Disturbance
W1 = theta_lim*[-.05, -.025, 0, .025, .05];
W2 = thetadot_lim*[-.05, -.025, 0, .025, .05];
W = combvec(W1,W1); % discrete set of disturbance (column vectors)
ProblemData.DiscreteDisturbance = W;
ProblemData.DisturbancePMF = ones(1,size(W,2))/size(W,2); % disturbance probability mass function 

% Discretization of the state and input (and their dual) spaces
ProblemData.StateGridSize = [21; 21]; % size of the grid X
ProblemData.InputGridSize = 21; % size of the grid U
ProblemData.StateDualGridSize = ProblemData.StateGridSize; % size of the grid Y
%ProblemData.InputDualGridSize = ProblemData.InputGridSize; % size of the grid V
ProblemData.StateDynamicsGridSize = ProblemData.StateGridSize; % size of the grid Z
%   State space discretization (uniform grid)
Delta_x = [-theta_lim,theta_lim; -thetadot_lim,thetadot_lim]; % i-th row determines the range of i-th state
ProblemData.StateGrid = unif_grid(Delta_x,ProblemData.StateGridSize);
%   Input space discretization (uniform grid)
Delta_u = [-u_lim,u_lim]; % i-th row determines the range of i-th input
ProblemData.InputGrid = unif_grid(Delta_u,ProblemData.InputGridSize);

% Output data
Result = struct;

% Optimal control problem instances - data
NumInstances = 1;
Result.InitialState = cell(NumInstances,1);
Result.InitialState{1} = [pi/12;0];
Result.DisturbanceSequence = cell(NumInstances,1);
ind_w_t = randsample(length(ProblemData.DisturbancePMF),ProblemData.Horizon,true,ProblemData.DisturbancePMF);
Result.DisturbanceSequence{1} = ProblemData.DiscreteDisturbance(:,ind_w_t);

% DP and CDP configuration
dpData = struct;
dpData.FastStochastic = true;
dpData.ExtensionInterpolationMethod = 'linear';
dpData.ExtensionExtrapolationMethod = 'linear';

cdpData = struct;
cdpData.NumericalConjugateCost = false; % the conjugate of the (input) stage cost is avalialble
cdpData.ExpectationInterpolationMethod = 'linear';
cdpData.ExpectationExtrapolationMethod = 'linear';
cdpData.CoarseDualGridConstructionAlpha = 1;

%% Implemenatation 

% Feasibility check of discretization scheme ------------------------------
feasibility_check_2(ProblemData)
%--------------------------------------------------------------------------

% Value iteration (backward iteration) ------------------------------------

% DP implementation 
[Result.J_dp, Result.Pi_dp, Result.run_time_B_dp] = d_DP_Alg_2(ProblemData, dpData); 

% CDP implemetation 
[Result.J_cdp, Result.run_time_B_cdp] = d_CDP_Alg_2(ProblemData, cdpData);
%--------------------------------------------------------------------------

% Solving the optimal control problem instances (forward iteration)--------

Result.x_dp_J = cell(NumInstances,1); Result.u_dp_J = cell(NumInstances,1); Result.traj_cost_dp_J = zeros(NumInstances,1); Result.run_time_F_dp_J = zeros(NumInstances,1);
Result.x_cdp_J = cell(NumInstances,1); Result.u_cdp_J = cell(NumInstances,1); Result.traj_cost_cdp_J = zeros(NumInstances,1); Result.run_time_F_cdp_J = zeros(NumInstances,1);

for i_Inst = 1:NumInstances % Iteraton over control problem instance
    
    ProblemData.DisturbanceSequence = Result.DisturbanceSequence{i_Inst};
    ProblemData.InitialState = Result.InitialState{i_Inst};

    % DP 
    [Result.x_dp_J{i_Inst},Result.u_dp_J{i_Inst},Result.traj_cost_dp_J(i_Inst),Result.run_time_F_dp_J(i_Inst)] = forward_iter_J_2(ProblemData, Result.J_dp); 
    
    % CDP 
    [Result.x_cdp_J{i_Inst},Result.u_cdp_J{i_Inst},Result.traj_cost_cdp_J(i_Inst),Result.run_time_F_cdp_J(i_Inst)] = forward_iter_J_2(ProblemData, Result.J_cdp);

end

% Saving the results ------------------------------------------------------
str = sprintf('Example_2_IP_Data');
save(str,'ProblemData','dpData','cdpData','Result')
%--------------------------------------------------------------------------

%% Plotting the outputs

% Cost-to-go  at t=0 ------------------------------------------------------
J_dp = Result.J_dp;
J_cdp = Result.J_cdp;
X = ProblemData.StateGrid;
infRGB = reshape([0 0 0],1,1,3);
infimgdata = repmat(infRGB, size(J_dp{1},1), size(J_dp{1},2));

h1 = figure;
subplot(1,2,1)
infimg = image('XData',X{1},'YData',X{2},'CData',infimgdata, 'alphadata', isinf(J_dp{1}));
hold on
test_image = imagesc('XData',X{1},'YData',X{2},'CData',J_dp{1},'alphadata', ~(isnan(J_dp{1})|isinf(J_dp{1})));
caxis([0, 100]);
colormap(jet());
colorbar
xlabel('$\theta$','FontSize',10,'Interpreter','latex')
ylabel('$\dot{\theta}$','FontSize',10,'Interpreter','latex')
zlabel('$J^{DP}_{0}$','FontSize',10,'Interpreter','latex')
hold off
subplot(1,2,2)
infimg = image('XData',X{1},'YData',X{2},'CData',infimgdata, 'alphadata', isinf(J_cdp{1}));
hold on
test_image = imagesc('XData',X{1},'YData',X{2},'CData',J_cdp{1},'alphadata', ~(isnan(J_cdp{1})|isinf(J_cdp{1})));
caxis([0, 100]);
colormap(jet());
colorbar
xlabel('$\theta$','FontSize',10,'Interpreter','latex')
ylabel('$\dot{\theta}$','FontSize',10,'Interpreter','latex')
zlabel('$J^{CDP}_{0}$','FontSize',10,'Interpreter','latex')
%--------------------------------------------------------------------------

% Controlled trajectory ---------------------------------------------------
x_dp = Result.x_dp_J;
u_dp = Result.u_dp_J;
x_cdp = Result.x_cdp_J;
u_cdp = Result.u_cdp_J;
T = ProblemData.Horizon;

h2 = figure;
subplot(2,3,1)
plot(0:T, x_dp{1}(1,:),'-r','LineWidth',1.1)
ylabel('$\theta^{DP}$','Interpreter','latex','FontSize',10)
subplot(2,3,2)
plot(0:T, x_dp{1}(2,:),'-r','LineWidth',1.1)
ylabel('$\dot{\theta}^{DP}$','Interpreter','latex','FontSize',10)
subplot(2,3,3)
plot(0:T-1, u_dp{1},'-r','LineWidth',1.1)
ylabel('$u^{DP}$','Interpreter','latex','FontSize',10)
subplot(2,3,4)
plot(0:T, x_cdp{1}(1,:),'-b','LineWidth',1.1)
ylabel('$\theta^{CDP}$','Interpreter','latex','FontSize',10)
xlabel('$t$','Interpreter','latex','FontSize',10)
subplot(2,3,5)
plot(0:T, x_cdp{1}(2,:),'-b','LineWidth',1.1)
ylabel('$\dot{\theta}^{CDP}$','Interpreter','latex','FontSize',10)
xlabel('$t$','Interpreter','latex','FontSize',10)
subplot(2,3,6)
plot(0:T-1, u_cdp{1},'-b','LineWidth',1.1)
ylabel('$u^{CDP}$','Interpreter','latex','FontSize',10)
xlabel('$t$','Interpreter','latex','FontSize',10)
hold off
%--------------------------------------------------------------------------
