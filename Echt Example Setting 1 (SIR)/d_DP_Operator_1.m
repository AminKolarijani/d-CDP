function [J, Mu] = d_DP_Operator_1(ProblemData, dpData, J_next)
% This function contains the implementation of the d-DP operation for SETTING 1; 
% i.e., it computes the optimal cost "J_t" and control law "Mu_t" at the current 
% time step, given the cost-to-go of the next step J_{t+1}.
%  
% Please see the following article for more details:
%   
%    KOLARIJANI & MOHAJERIN ESFAHANI (2020), "FAST DYNAMIC PROGRAMMING: 
%    APPLICATION OF CONJUGATE DUALITY IN APPROXIMATE DYNAMIC PROGRAMMING"
%
% Inputs: 
%   Data structure containing the problem data and d-DP configuration (see the block "local variables" below)
%   J_next - the discrete cost-to-go of the next step over the discrete state space
%
% Outputs: 
%   J: the discrete cost-to-go of the current step over the discrete state space
%   Mu: cell of size (n_u,1) for the discrete optimal control law of the current step over the discrete state space
%

%==========================================================================

% local variables (begins) ------------------------------------------------
dyn = ProblemData.Dynamics;

cost = ProblemData.StageCost;
constr_xu = ProblemData.StateInputConstraints;
constr_x = ProblemData.StateConstraints;
constr_u = ProblemData.InputConstraints;

X = ProblemData.StateGrid;
U = ProblemData.InputGrid;

interpol = dpData.ExtensionInterpolationMethod;
extrapol = dpData.ExtensionExtrapolationMethod;

stoch = ProblemData.Stochastic;
if stoch
    fast = dpData.FastStochastic;
    W = ProblemData.DiscreteDisturbance;
    P = ProblemData.DisturbancePMF;
end
%--------------------------------------------------------------------------

% constraints
feasibility_t = @(x,u)  max( max(max(constr_x(x)) , max(constr_u(u)) ), max(constr_xu(x,u)) );


n_x = size(X,1); % dimension of the state space
n_u = size(U,1); % dimension of the input space
ind_max_x = zeros(1,n_x);
for i = 1:n_x
    ind_max_x(i) = length(X{i});
end

temp_ind = num2cell(ind_max_x);

% allocations
J = zeros(temp_ind{:});
Mu = cell(n_u,1);
for j = 1:n_u
    Mu{j} = zeros(temp_ind{:});
end

if stoch && fast
    J_next = ext_constr_expect(X,J_next,X,@(x) x,constr_x,W,P,interpol,extrapol);
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
        J_at_U = ext_constr_expect(X,J_next,U,@(u) (dyn(x,u)),constr_x,W,P,interpol,extrapol);
    else
        J_at_U = ext_constr(X,J_next,U,@(u) (dyn(x,u)),constr_x,interpol,extrapol);
    end
        
    cost_at_U = eval_func_constr(@(u) (cost(x,u)), U, @(u) (feasibility_t(x,u)));
    Q = cost_at_U{1} + J_at_U;

    [J_opt, temp2] = min(Q(:));
    ind_u_opt = cell(1,n_u);
    [ind_u_opt{:}] = ind2sub(size(Q),temp2);

    u_opt = zeros(n_u,1);
    for i=1:n_u
        u_opt(i) = U{i}(ind_u_opt{i});
    end
    % -----------------------------------------------------------------
    
    J(temp_ind{:}) = J_opt;
    
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

