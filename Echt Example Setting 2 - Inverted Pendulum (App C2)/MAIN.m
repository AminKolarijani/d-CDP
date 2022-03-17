% This file is an example of implementation of the extended d-CDP Algorithm 4 
%   for solving finite-horizon optimal control problems of a noisy inverted 
%   pendulum. For more detail, see Appendix C.2.2 of the following article:
%   
%                KOLARIJANI & MOHAJERIN ESFAHANI (2022), 
%    "FAST APPROXIMATE DYNAMIC PROGRAMMING FOR INPUT-AFFINE DYNAMICS"
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
theta_lim = pi/3; thetadot_lim = pi; u_lim = 3; % state and input variables limits
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
theta_lim = pi/4; Delta_x = [-theta_lim,theta_lim; -thetadot_lim,thetadot_lim]; % i-th row determines the range of i-th state
ProblemData.StateGrid = unif_grid(Delta_x,ProblemData.StateGridSize);
%   Input space discretization (uniform grid)
Delta_u = [-u_lim,u_lim]; % i-th row determines the range of i-th input
ProblemData.InputGrid = unif_grid(Delta_u,ProblemData.InputGridSize);

% Output data
Result = struct;

% DP and CDP configuration
dpData = struct;
dpData.FastStochastic = false;
dpData.ExtensionInterpolationMethod = 'nearest';
dpData.ExtensionExtrapolationMethod = 'nearest';

cdpData = struct;
cdpData.NumericalConjugateCost = false; % the conjugate of the (input) stage cost is avalialble
cdpData.ExpectationInterpolationMethod = 'nearest';
cdpData.ExpectationExtrapolationMethod = 'nearest';
cdpData.CoarseDualGridConstructionAlpha = 2;

%% Implemenatation 

% Feasibility check of discretization scheme 
feasibility_check_2(ProblemData)
%--------------------------------------------------------------------------

% Value iteration (backward iteration) 

% DP implementation 
[Result.J_dp, Result.Pi_dp, Result.run_time_B_dp] = d_DP_Alg_2(ProblemData, dpData); 

% CDP implemetation 
[Result.J_cdp, Result.run_time_B_cdp] = d_CDP_Alg_2(ProblemData, cdpData);
[dummy, Result.Mu_cdp] = d_DP_Operator_2(ProblemData, dpData, Result.J_cdp{2}); % the optimal control law at t=0
%--------------------------------------------------------------------------

% Saving the results 
str = sprintf('Ex_IP_Data');
save(str,'ProblemData','dpData','cdpData','Result')

%% Plotting the outputs 

% Cost-to-go and optimal policy  at t=0 

% scales
J_max = 200;
Mu_max = 3;

% DP
J = Result.J_dp;
Mu = Result.Pi_dp{1};
X = ProblemData.StateGrid;
infRGB = reshape([0 0 0],1,1,3);
infimgdata = repmat(infRGB, size(J{1},1), size(J{1},2));

h1 = figure;
ax1 = subplot(2,1,1);
infimg = image('XData',X{1},'YData',X{2},'CData',infimgdata, 'alphadata', isinf(J{1}));
hold on
test_image = imagesc('XData',X{1},'YData',X{2},'CData',J{1},'alphadata', ~(isnan(J{1})|isinf(J{1})));
caxis([0, J_max]);
colormap(ax1,parula());
colorbar 
xlabel('$\theta$','FontSize',10,'Interpreter','latex')
ylabel('$\dot{\theta}$','FontSize',10,'Interpreter','latex')
ax = gca;
ax.Box = 'on';
xlim([-pi/4 pi/4])
ylim([-pi pi])
hold off
ax2 = subplot(2,1,2);
infimg = image('XData',X{1},'YData',X{2},'CData',infimgdata, 'alphadata', isinf(Mu{1}));
hold on
test_image = imagesc('XData',X{1},'YData',X{2},'CData',Mu{1},'alphadata', ~(isnan(Mu{1})|isinf(Mu{1})));
caxis([-Mu_max, Mu_max]);
colormap(ax2,parula());
colorbar
xlabel('$\theta$','FontSize',10,'Interpreter','latex')
ylabel('$\dot{\theta}$','FontSize',10,'Interpreter','latex')
ax = gca;
ax.Box = 'on';
xlim([-pi/4 pi/4])
ylim([-pi pi])
hold off
%--------------------------------------------------------------------------

% CDP 
J = Result.J_cdp;
Mu = Result.Mu_cdp;
h2 = figure;
ax1 = subplot(2,1,1);
infimg = image('XData',X{1},'YData',X{2},'CData',infimgdata, 'alphadata', isinf(J{1}));
hold on
test_image = imagesc('XData',X{1},'YData',X{2},'CData',J{1},'alphadata', ~(isnan(J{1})|isinf(J{1})));
caxis([0, J_max]);
colormap(ax1,parula())
colorbar 
xlabel('$\theta$','FontSize',10,'Interpreter','latex')
ylabel('$\dot{\theta}$','FontSize',10,'Interpreter','latex')
ax = gca;
ax.Box = 'on';
xlim([-pi/4 pi/4])
ylim([-pi pi])
hold off
ax2 = subplot(2,1,2);
infimg = image('XData',X{1},'YData',X{2},'CData',infimgdata, 'alphadata', isinf(Mu{1}));
hold on
test_image = imagesc('XData',X{1},'YData',X{2},'CData',Mu{1},'alphadata', ~(isnan(Mu{1})|isinf(Mu{1})));
caxis([-Mu_max, Mu_max]);
colormap(ax2,parula());
colorbar
xlabel('$\theta$','FontSize',10,'Interpreter','latex')
ylabel('$\dot{\theta}$','FontSize',10,'Interpreter','latex')
ax = gca;
ax.Box = 'on';
xlim([-pi/4 pi/4])
ylim([-pi pi])
hold off

