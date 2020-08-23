function [J_t, run_time] = d_CDP_Alg_2(ProblemData, cdpData)
% This function contains the implementation of the 
%                 "d-CDP ALGORITHM 2 for the SETTING 2"
% and computes the optimal costs for all time steps over the horizon "T".
% Please see the following article for more details:
%   
%    KOLARIJANI & MOHAJERIN ESFAHANI (2020), "FAST DYNAMIC PROGRAMMING: 
%    APPLICATION OF CONJUGATE DUALITY IN APPROXIMATE DYNAMIC PROGRAMMING"
%
% Input: 
%   Data structure containing the problem data and d-CDP configuration (see
%   the block "local variables" below).
%
% Outputs: 
%   J_t: cell of size (T+1,1) for the discrete costs-to-go over the 
%        discrete state space for each step in the horizon
%   run_time: vector of size (2,1) for the run-time (in seconds) of the 
%             algorithm, where the first component is compilation time 
%             (particularly, including the time spend for computing the 
%             conjugte of the stage cost numerically), and the second 
%             component is the time spend for backward value iteration
%             computing the discrete costs-to-go.
%

%==========================================================================

% local variables (begins) ------------------------------------------------
dyn_x = ProblemData.StateDynamics; 
B = ProblemData.InputMatrix;

T = ProblemData.Horizon;
 
cost_x = ProblemData.StateCost;
cost_u = ProblemData.InputCost;
cost_T = ProblemData.TerminalCost;
constr_x = ProblemData.StateConstraints;
constr_u = ProblemData.InputConstraints;

X = ProblemData.StateGrid;
U = ProblemData.InputGrid;
Ny = ProblemData.StateDualGridSize;
Nz = ProblemData.StateDynamicsGridSize;

alpha_coarse = cdpData.CoarseDualGridConstructionAlpha;

num_conj_cost = cdpData.NumericalConjugateCost;
if num_conj_cost
    Nv = ProblemData.InputDualGridSize;
else
    conj_cost_u = ProblemData.ConjugateInputCost;
end
    
stoch = ProblemData.Stochastic;
if stoch
    W = ProblemData.DiscreteDisturbance;
    P = ProblemData.DisturbancePMF;
    exp_interpol = cdpData.ExpectationInterpolationMethod;
    exp_extrapol = cdpData.ExpectationExtrapolationMethod;
end
% local variables (ends) --------------------------------------------------    

% allocation 
J_t = cell(T+1,1); 
run_time = [0; 0];

tic % compilation time (begins)
%==========================================================================

% Computing the conjugate of the input cost numerically
n_x = size(X,1); % dimension of the grid X (and also Y)
n_u = size(U,1); % dimension of the grid U
if num_conj_cost
    
    % construction of the "uniform" grid V --------------------------------
    disc_cost_u = eval_func_constr(cost_u,U,constr_u);
    Delta_v = slope_range(U,disc_cost_u{1}); % assuming C_x is convex
    V = cell(n_u,1);
    for i = 1:n_u
        temp_v = linspace(Delta_v(i,1),Delta_v(i,2),Nv(i)-3);
        temp_v = [2*temp_v(1)-temp_v(2), temp_v, 2*temp_v(end)-temp_v(end-1)];
        V{i} = unique([temp_v';0]);
    end
    %----------------------------------------------------------------------
    
    disc_conj_cost_u = LLT(U,disc_cost_u{1},V);
        
    temp_Ci = disc_cost_u{1}(:);  
    Max_cost_u = max(temp_Ci(~isinf(temp_Ci)));
    Min_cost_u = min(temp_Ci);
            
end

% Computing the discrete state cost
tempCs = eval_func_constr(cost_x,X,constr_x); 
disc_cost_x = tempCs{1};
    
% Computing the maximum and minimum of the input stage cost numerically (for construction of the grid Y)
if (~num_conj_cost)
        disc_cost_u = eval_func_constr(cost_u,U,constr_u);
        temp_Ci = disc_cost_u{1}(:);  
        Max_cost_u = max(temp_Ci(~isinf(temp_Ci)));
        Min_cost_u = min(temp_Ci);
end
    
% construction of the "uniform" grid Z
f_s_X = eval_func(dyn_x,X);
Delta_z = zeros(n_x,2); 
for i = 1:n_x
    Delta_z(i,1) = min(f_s_X{i}(:));
    Delta_z(i,2) = max(f_s_X{i}(:));
end
Z = unif_grid(Delta_z,Nz);

%==========================================================================
run_time(1) = run_time(1) + toc; % compilation time (ends)


tic % value iteration time (begins)
%==========================================================================
% Recursion: value iteration backward in time

% initialization via terminal cost function
disc_cost_T = eval_func_constr(cost_T,X,constr_x);
J = disc_cost_T{1};
J_t{T+1} = J;

% backward iteration
for t = 0:T-1
        
    if stoch
        J = ext_constr_expect(X,J,X,@(x) x,constr_x,W,P,exp_interpol,exp_extrapol);
    end

    % construction of the "uniform" grid Y -------------------------------- 
    temp_J = J(:);
    Delta_Q = Max_cost_u + max(temp_J(~isinf(temp_J))) - Min_cost_u - min(temp_J);
    Delta_y = zeros(n_x,2);
    for i=1:n_x
        Delta_X_i = X{i}(end) - X{i}(1);
        Delta_y(i,2) = alpha_coarse * Delta_Q / Delta_X_i;
        Delta_y(i,1) = -Delta_y(i,2);
    end
    Y = cell(n_x,1); 
    for i = 1:n_x
        Y{i} = unique([(linspace(Delta_y(i,1),Delta_y(i,2),Ny(i)-1))';0]);
    end
    %----------------------------------------------------------------------
        
    J_conj = LLT(X,J,Y);
        
    if num_conj_cost
        phi = ext_constr(V,disc_conj_cost_u,Y,@(y) (-B'*y),@(x) (0),'linear','linear') + J_conj;
    else
        temp = eval_func(@(y) conj_cost_u(-B'*y),Y); phi = temp{1}+ J_conj;
    end
        
    phi_conj = LLT(Y,phi,Z);
        
    J_minus_wo_cost_x = ext_constr(Z,phi_conj,X,dyn_x,@(x) (0),'linear','linear');
        
    J_minus = disc_cost_x + J_minus_wo_cost_x; 
        
    J_t{T-t} = J_minus;
    J = J_minus;

end

%==========================================================================
run_time(2) = run_time(2) + toc; % value iteration time (ends)



