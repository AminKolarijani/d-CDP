% This file is an example of implementation of the extended d-CDP Algorithm 3 
%   for solving finite-horizon optimal control problems of SIR system. Please 
%   see Appendix C.2.1 of the article for more details:
%   
%    KOLARIJANI & MOHAJERIN ESFAHANI (2020), 
%      "FAST APPROXIMATE DYNAMIC PROGRAMMING FOR INPUT-AFFINE DYNAMICS"
% 

clc
clear all

%% Problem data

ProblemData = struct;

% Dynamics (SIR model) - we can simply disregard the "R" dynamics
alpha = 0.02*100;
beta = 0.1;
ProblemData.StateDynamics = @(x) ([x(1)*(1-alpha*x(2)); x(2)*(1-beta+alpha*x(1))]); 
ProblemData.InputDynamics = @(x) ([-x(1)*(1-alpha*x(2)); -alpha*x(1)*x(2)]); 
ProblemData.Dynamics = @(x,u) (ProblemData.StateDynamics(x) + ProblemData.InputDynamics(x) * u);

% Horizon
ProblemData.Horizon = 3;

% Constraints
u_max = .8;
ProblemData.StateConstraints = @(x) ([-1 0; -1 0]*x - [0; 0]); 
ProblemData.InputConstraints = @(u) ([1; -1]*u - [u_max; 0]);  
ProblemData.StateInputConstraints = @(x,u) (0);

% Cost functions
gamma = 100;
ProblemData.TerminalCost = @(x) (gamma*x(2)); 
ProblemData.StageCost = @(x,u) (gamma*x(2)+u);

% Stochasticity
ProblemData.Stochastic = false; 

% Output data
Result = struct;

% DP and CDP configurations
dpData = struct;
dpData.ExtensionInterpolationMethod = 'linear';
dpData.ExtensionExtrapolationMethod = 'linear';

cdpData = struct;
cdpData.NumericalConjugateCost = true;
cdpData.CoarseDualGridConstructionAlpha = .5;

%% Implementation: 

% Problem data ------------------------------------------------------------

% State space discretization (uniform grid)
ProblemData.StateGridSize = [21 21]; % size of the grid X
Delta_x = [0 1; 0 .5]; % i-th row determines the range of i-th state
ProblemData.StateGrid = unif_grid(Delta_x,ProblemData.StateGridSize);

% Input space discretization (uniform grid)
ProblemData.InputGridSize = 21; % size of the grid U
Delta_u = [0, u_max]; % i-th row determines the range of i-th input
ProblemData.InputGrid = unif_grid(Delta_u,ProblemData.InputGridSize);
%--------------------------------------------------------------------------

% Feasibility check of discretization scheme ------------------------------
feasibility_check_1(ProblemData)
%--------------------------------------------------------------------------

% Value iteration (backward iteration)-------------------------------------

% DP 
[Result.J_dp, Result.Pi_dp, Result.run_time_B_dp] = d_DP_Alg_1(ProblemData, dpData); % implementation

% CDP - discretization scheme 1
ProblemData.StateDualGridSize = ProblemData.StateGridSize; % size of the grid Y
ProblemData.InputDualGridSize = ProblemData.InputGridSize; % size of the grid V
[Result.J_cdp_1, Result.run_time_B_cdp_1] = d_CDP_Alg_1(ProblemData, cdpData); % implementation
[dummy, Result.Mu_cdp_1] = d_DP_Operator_1(ProblemData, dpData, Result.J_cdp_1{2}); % the optimal control law at t=0

% CDP - discretization scheme 2
ProblemData.StateDualGridSize = [11 11]; % size of the grid Y
ProblemData.InputDualGridSize = 11; % size of the grid V
[Result.J_cdp_2, Result.run_time_B_cdp_2] = d_CDP_Alg_1(ProblemData, cdpData); % implementation
[dummy, Result.Mu_cdp_2] = d_DP_Operator_1(ProblemData, dpData, Result.J_cdp_2{2}); % the optimal control law at t=0
%--------------------------------------------------------------------------

% Saving the results ------------------------------------------------------
str = sprintf('Example_1_SIR_Data');
save(str,'ProblemData','dpData','cdpData','Result') 
%--------------------------------------------------------------------------

%% Plotting the outputs: Cost-to-go and control law at t=0

% scales
J_max = 400;
Mu_max = 0.8;

% DP ----------------------------------------------------------------------
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
xlabel('$s$','FontSize',10,'Interpreter','latex')
ylabel('$i$','FontSize',10,'Interpreter','latex')
ax = gca;
ax.Box = 'on';
xlim([-0.07 1.07])
ylim([-0.03 0.53])
hold off
ax2 = subplot(2,1,2);
infimg = image('XData',X{1},'YData',X{2},'CData',infimgdata, 'alphadata', isinf(Mu{1}));
hold on
test_image = imagesc('XData',X{1},'YData',X{2},'CData',Mu{1},'alphadata', ~(isnan(Mu{1})|isinf(Mu{1})));
caxis([0, Mu_max]);
colormap(ax2,parula());
colorbar
xlabel('$s$','FontSize',10,'Interpreter','latex')
ylabel('$i$','FontSize',10,'Interpreter','latex')
ax = gca;
ax.Box = 'on';
xlim([-0.07 1.07])
ylim([-0.03 0.53])
hold off
%--------------------------------------------------------------------------

% CDP 1 -------------------------------------------------------------------
J = Result.J_cdp_1;
Mu = Result.Mu_cdp_1;
h2 = figure;
ax1 = subplot(2,1,1);
infimg = image('XData',X{1},'YData',X{2},'CData',infimgdata, 'alphadata', isinf(J{1}));
hold on
test_image = imagesc('XData',X{1},'YData',X{2},'CData',J{1},'alphadata', ~(isnan(J{1})|isinf(J{1})));
caxis([0, J_max]);
colormap(ax1,parula())
colorbar 
xlabel('$s$','FontSize',10,'Interpreter','latex')
ylabel('$i$','FontSize',10,'Interpreter','latex')
ax = gca;
ax.Box = 'on';
xlim([-0.07 1.07])
ylim([-0.03 0.53])
hold off
ax2 = subplot(2,1,2);
infimg = image('XData',X{1},'YData',X{2},'CData',infimgdata, 'alphadata', isinf(Mu{1}));
hold on
test_image = imagesc('XData',X{1},'YData',X{2},'CData',Mu{1},'alphadata', ~(isnan(Mu{1})|isinf(Mu{1})));
caxis([0, Mu_max]);
colormap(ax2,parula());
colorbar
xlabel('$s$','FontSize',10,'Interpreter','latex')
ylabel('$i$','FontSize',10,'Interpreter','latex')
ax = gca;
ax.Box = 'on';
xlim([-0.07 1.07])
ylim([-0.03 0.53])
hold off
%--------------------------------------------------------------------------

% CDP 2 -------------------------------------------------------------------
J = Result.J_cdp_2;
Mu = Result.Mu_cdp_2;
h3 = figure;
ax1 = subplot(2,1,1);
infimg = image('XData',X{1},'YData',X{2},'CData',infimgdata, 'alphadata', isinf(J{1}));
hold on
test_image = imagesc('XData',X{1},'YData',X{2},'CData',J{1},'alphadata', ~(isnan(J{1})|isinf(J{1})));
caxis([0, J_max]);
colormap(ax1,parula())
colorbar 
xlabel('$s$','FontSize',10,'Interpreter','latex')
ylabel('$i$','FontSize',10,'Interpreter','latex')
ax = gca;
ax.Box = 'on';
xlim([-0.07 1.07])
ylim([-0.03 0.53])
hold off
ax2 = subplot(2,1,2);
infimg = image('XData',X{1},'YData',X{2},'CData',infimgdata, 'alphadata', isinf(Mu{1}));
hold on
test_image = imagesc('XData',X{1},'YData',X{2},'CData',Mu{1},'alphadata', ~(isnan(Mu{1})|isinf(Mu{1})));
caxis([0, Mu_max]);
colormap(ax2,parula());
colorbar
xlabel('$s$','FontSize',10,'Interpreter','latex')
ylabel('$i$','FontSize',10,'Interpreter','latex')
ax = gca;
ax.Box = 'on';
xlim([-0.07 1.07])
ylim([-0.03 0.53])
hold off
%--------------------------------------------------------------------------


