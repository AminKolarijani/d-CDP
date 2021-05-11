function [J_t, Pi_t, run_time] = d_DP_Alg_2(ProblemData, dpData)
% This function contains the implementation of the 
%                 "d-DP ALGORITHM for the SETTING 2"
% and computes the optimal costs for all time steps over the horizon "T".
% Please see the following article for more details:
%   
%    KOLARIJANI & MOHAJERIN ESFAHANI (2020), "FAST DYNAMIC PROGRAMMING: 
%    APPLICATION OF CONJUGATE DUALITY IN APPROXIMATE DYNAMIC PROGRAMMING"
%
% Inputs: 
%   Data structure containing the problem data and d-DP configuration (see
%   the block "local variables" below).
%
% Outputs: 
%   J_t: cell of size (T+1,1) for the discrete costs-to-go over the 
%        discrete state space for each step in the horizon
%   Pi_t: cell of size (T+1,1) for the discrete optimal ploicy over the 
%         discrete state space for each step in the horizon; each element of
%         the cell is again a cell of size (n_u,1) containing the optimal 
%         control law for the n_u input channels at the corresponding time 
%   run_time: a scalar for the run-time (in seconds) of the algorithm
%

%==========================================================================

% local variables (begins) ------------------------------------------------
dyn = ProblemData.Dynamics;

T = ProblemData.Horizon;

cost_T = ProblemData.TerminalCost;
cost_x = ProblemData.StateCost;
constr_x = ProblemData.StateConstraints;
cost_u = ProblemData.InputCost;
constr_u = ProblemData.InputConstraints;

X = ProblemData.StateGrid;
U = ProblemData.InputGrid;

interpol = dpData.ExtensionInterpolationMethod;
extrapol = dpData.ExtensionExtrapolationMethod;

stoch = ProblemData.Stochastic;
if stoch
    W = ProblemData.DiscreteDisturbance;
    P = ProblemData.DisturbancePMF;
    fast = dpData.FastStochastic;
end
% local variables (ends) --------------------------------------------------

% allocation 
J_t = cell(T+1,1); % approximate cost-to-go functions
Pi_t = cell(T,1); % approximate cost-to-go functions

tic % algorithm run-time (begins)
%==========================================================================

n_x = size(X,1); % dimension of the state space
n_u = size(U,1); % dimension of the input space
ind_max_x = zeros(1,n_x);
for i = 1:n_x
    ind_max_x(i) = length(X{i});
end 

% discrete costs
disc_cost_u = eval_func_constr(cost_u,U,constr_u);
disc_cost_x = eval_func_constr(cost_x,X,constr_x);
disc_cost_T = eval_func_constr(cost_T,X,constr_x);

% initialization via terminal cost function
J = disc_cost_T{1};
J_t{T+1} = J;

% backward iteration
for t = 0:T-1
    
    % fast method
    if stoch && fast
        J = ext_constr_expect(X,J,X,@(x) x,constr_x,W,P,interpol,extrapol);
    end
    
    temp_ind = num2cell(ind_max_x);
    J_minus = zeros(temp_ind{:});

    Mu = cell(n_u,1);
    for j = 1:n_u
        Mu{j} = zeros(temp_ind{:});
    end

    ind_x = ones(1,n_x);
    ready = false;
    while ~ready % loop over x \in X
        
        temp_ind = num2cell(ind_x);

        x = zeros(n_x,1);
        for i=1:n_x
            x(i) = X{i}(ind_x(i));
        end
        
        % minimization over u ---------------------------------------------
        
        % computing the LERP of the cost-to-go J at (x,u) for u \in U
        if stoch && ~fast 
            J_at_U = ext_constr_expect(X,J,U,@(u) (dyn(x,u)),constr_x,W,P,interpol,extrapol);
        else
            J_at_U = ext_constr(X,J,U,@(u) (dyn(x,u)),constr_x,interpol,extrapol);
        end
        
        Q = disc_cost_u{1} + J_at_U;

        [J_opt, temp2] = min(Q(:));
        ind_u_opt = cell(1,n_u);
        [ind_u_opt{:}] = ind2sub(size(Q),temp2);

        u_opt = zeros(n_u,1);
        for i=1:n_u
            u_opt(i) = U{i}(ind_u_opt{i});
        end
        % -----------------------------------------------------------------
        
        J_minus(temp_ind{:}) = J_opt;
        for j = 1:n_u
            Mu{j}(temp_ind{:}) = u_opt(j);
        end

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
    
    J = J_minus + disc_cost_x{1};
    
    J_t{T-t} = J;
    Pi_t{T-t} = Mu;
    
end

%==========================================================================
run_time = toc; % algorithm run-time (ends)
