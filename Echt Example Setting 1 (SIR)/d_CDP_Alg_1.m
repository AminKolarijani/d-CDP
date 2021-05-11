function [J_t, run_time] = d_CDP_Alg_1(ProblemData, cdpData)
% This function contains the implementation of the 
%                 "d-CDP ALGORITHM 1 for the SETTING 1"
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
dyn_u = ProblemData.InputDynamics; 
    
T = ProblemData.Horizon;

cost = ProblemData.StageCost;
cost_T = ProblemData.TerminalCost;
constr_x = ProblemData.StateConstraints;
constr_u = ProblemData.InputConstraints;
constr_xu = ProblemData.StateInputConstraints;

X = ProblemData.StateGrid;
U = ProblemData.InputGrid;

Ny = ProblemData.StateDualGridSize; 
alpha_coarse = cdpData.CoarseDualGridConstructionAlpha;

num_conj_cost = cdpData.NumericalConjugateCost; 
if num_conj_cost
    Nv = ProblemData.InputDualGridSize;
else
    conj_cost = ProblemData.CojugateStageCost;
end
 
stoch = ProblemData.Stochastic;
if stoch
    W = ProblemData.DiscreteDisturbance;
    P = ProblemData.DisturbancePMF;
    exp_interpol = cdpData.ExpectationInterpolationMethod;
    exp_extrapol = cdpData.ExpectationExtrapolationMethod;
end
% local variables (ends) --------------------------------------------------

% constraints
feasibility_t = @(x,u)  max( max(max(constr_x(x)) , max(constr_u(u)) ), max(constr_xu(x,u)) );

% allocation 
J_t = cell(T+1,1); 
run_time = [0; 0];
    
tic % compilation time (begins)
%==========================================================================
    
% Computing the conjugate of the stage cost numerically

n_x = size(X,1); % dimension of the grid X (and also Y)
n_u = size(U,1); % dimension of the grid U
if num_conj_cost

    ind_max_x = zeros(1,n_x);
    for i = 1:n_x
        ind_max_x(i) = length(X{i});
    end

    % allocation 
    temp_ind = num2cell(ind_max_x);
    disc_conj_cost = cell(temp_ind{:});
    V = cell(temp_ind{:});
    Max_cost = nan;
    Min_cost = nan;

    ind_x = ones(1,n_x);
    ready = false;
    while ~ready % (loop over x \in X)
        
        temp_ind = num2cell(ind_x);

        x = zeros(n_x,1);
        for i=1:n_x
            x(i) = X{i}(ind_x(i));
        end
            
        % construction of the "uniform" grid V ----------------------------
%        disc_C_x_constrained = eval_func_const(@(u) (cost(x,u)), U, @(u) feasibility_t(x,u));
%        Delta_v = slope_range(U,disc_C_x_constrained{1}); % assuming C_x is convex
         disc_C_x_unconstrained = eval_func(@(u) (cost(x,u)),U);
         Delta_v = slope_range(U,disc_C_x_unconstrained{1}); % assuming C_x is convex
         V{temp_ind{:}} = cell(n_u,1);
         for i = 1:n_u
             temp_v = linspace(Delta_v(i,1),Delta_v(i,2),Nv(i)-3);
             temp_v = [2*temp_v(1)-temp_v(2), temp_v, 2*temp_v(end)-temp_v(end-1)];
             V{temp_ind{:}}{i} = unique([temp_v';0]);
         end
         %-----------------------------------------------------------------
            
         disc_C_x_constrained = eval_func_constr(@(u) (cost(x,u)), U, @(u) feasibility_t(x,u));
         disc_conj_cost{temp_ind{:}} = LLT(U,disc_C_x_constrained{1},V{temp_ind{:}});
            
         temp_Cx = disc_C_x_constrained{1}(:);  temp_max = max(temp_Cx(~isinf(temp_Cx)));
         temp_min = min(disc_C_x_constrained{1}(:));
         Max_cost = max([Max_cost temp_max],[],'omitnan');
         Min_cost = min([Min_cost temp_min],[],'omitnan');
            
         ready = true;
         for k = 1:n_x
             ind_x(k) = ind_x(k)+1;
             if ind_x(k) <= ind_max_x(k)
                 ready = false;
                 break;
             end
             ind_x(k) = 1;
         end

    end

end

%==========================================================================

% Computing the maximum and minimum of the stage cost numerically (for construction of the grid Y)

if (~num_conj_cost) 
    
    ind_max_x = zeros(1,n_x);
    for i = 1:n_x
        ind_max_x(i) = length(X{i});
    end

    % allocation 
    Max_cost = nan;
    Min_cost = nan;

    ind_x = ones(1,n_x);
    ready = false;
    while ~ready % (loop over x \in X)
        
        x = zeros(n_x,1);
        for i=1:n_x
            x(i) = X{i}(ind_x(i));
        end
            
        disc_C_x_constrained = eval_func_constr(@(u) (cost(x,u)), U, @(u) feasibility_t(x,u));

        temp_Cx = disc_C_x_constrained{1}(:);  temp_max = max(temp_Cx(~isinf(temp_Cx)));
        temp_min = min(disc_C_x_constrained{1}(:));
        Max_cost = max([Max_cost temp_max],[],'omitnan');
        Min_cost = min([Min_cost temp_min],[],'omitnan');

        ready = true;
        for k = 1:n_x
            ind_x(k) = ind_x(k)+1;
            if ind_x(k) <= ind_max_x(k)
                ready = false;
                break;
            end
            ind_x(k) = 1;
        end

    end
    
end

%==========================================================================
run_time(1) = run_time(1) + toc; % compilation time (ends)

tic % value iteration time (begins)
%==========================================================================
% Recursion: value iteration backward in time
    
% initialization via terminal cost function
disc_cost_T = eval_func_constr(cost_T,X,constr_x);
J = disc_cost_T{1};
J_t{T+1} = J;
J_minus = zeros(size(J));

% backward iteration
for t = 0:T-1
    
    ind_max_x = zeros(1,n_x);
    for i = 1:n_x
        ind_max_x(i) = length(X{i});
    end

    if stoch
        J = ext_constr_expect(X,J,X,@(x) x,constr_x,W,P,exp_interpol,exp_extrapol);
    end

    % construction of the "uniform" grid Y --------------------------------
    temp_J = J(:); 
    Delta_Q = max(Max_cost, max(temp_J(~isinf(temp_J)))) - min(Min_cost, min(temp_J));
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
    %-------------------------------------------------------------------
        
    J_conj = LLT(X,J,Y);

    ind_x = ones(1,n_x);
    ready = false;
    while ~ready % (loop over x \in X)
        
        temp_ind = num2cell(ind_x);

        x = zeros(n_x,1);
        for i=1:n_x
            x(i) = X{i}(ind_x(i));
        end

        if num_conj_cost
            phi = ext_constr(V{temp_ind{:}},disc_conj_cost{temp_ind{:}},Y,@(y) (- dyn_u(x)'*y),@(x) (0),'linear','linear') + J_conj;
        else
            temp = eval_func(@(y) conj_cost(x,-dyn_u(x)'*y),Y); phi = temp{1}+ J_conj;
        end

        J_minus(temp_ind{:}) = LLT(Y,phi,num2cell(dyn_x(x)));

        ready = true;
        for k = 1:n_x
            ind_x(k) = ind_x(k)+1;
            if ind_x(k) <= ind_max_x(k)
                ready = false;
                break;
            end
            ind_x(k) = 1;
        end

    end

    J_t{T-t} = J_minus;
    J = J_minus;

end

%==========================================================================
run_time(2) = run_time(2) + toc; % value iteration time (ends)


