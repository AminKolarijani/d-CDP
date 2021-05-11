function [] = feasibility_check_1(ProblemData)
% This function checks if the discrettization of the state and input spaces
% is feasible corresponding to "SETTING 1" in the sense that 
%   (a) all "x \in X_discrete" satisfy the state constraints
%   (b) all "u \in U_discrete" satisfy the input constraints
%   (c) for each "x \in X_discrete", there exists a "u \in U_discrete" such 
%       that the state-dependent input constraints are satisfied and the 
%       next state is feasible ("f(x,u) \in X").
% Please see the following article for more details:
%   
%    KOLARIJANI & MOHAJERIN ESFAHANI (2020), "FAST DYNAMIC PROGRAMMING: 
%    APPLICATION OF CONJUGATE DUALITY IN APPROXIMATE DYNAMIC PROGRAMMING"
%
% Input: 
%   Data structure containing the problem data (see the block "local
%   variables" below).
%
% Output: 
%   If any of the constriants listed above are not satisfied, the routine 
%   generates a warning with an example of the state or input for which the
%   corresponding contraints are not satisfied. 
%

%==========================================================================

% local variables (begins) ------------------------------------------------
dyn = ProblemData.Dynamics;

constr_xu = ProblemData.StateInputConstraints;
constr_x = ProblemData.StateConstraints;
constr_u = ProblemData.InputConstraints;

X = ProblemData.StateGrid;
U = ProblemData.InputGrid;
% local variables (ends) --------------------------------------------------

n_x = size(X,1); % dimension of the input grid 
n_u = size(U,1); % dimension of the input grid 

ind_max_x = zeros(1,n_x);
for i = 1:n_x
    ind_max_x(i) = length(X{i});
end
ind_max_u = zeros(1,n_u);
for i = 1:n_u
    ind_max_u(i) = length(U{i});
end

%==========================================================================

% Discretization of the state space (state constraints)

state_check = true;
ind_x = ones(1,n_x);
ready_x = false;
while ~ready_x % loop over x \in X
    
    x = zeros(n_x,1);
    for i=1:n_x
        x(i) = X{i}(ind_x(i));
    end
    
    if constr_x(x)> 0
        state_check = false;
        break;
    end
    
    ready_x = true;
    for k = 1:n_x
        ind_x(k) = ind_x(k)+1;
        if ind_x(k) <= ind_max_x(k)
            ready_x = false;
            break;
        end
        ind_x(k) = 1;
    end

end

if ~state_check
    fprintf('WARNING: The state space discretization is not proper.\n')
    fprintf('The following grid point does not satisfy the state constraints:')
    x
end

%==========================================================================

% Discretization of the input space (input constraints)

input_check = true;
ind_u = ones(1,n_u);
ready_u = false;
while ~ready_u % loop over u \in U
    
    u = zeros(n_u,1);
    for i=1:n_u
        u(i) = U{i}(ind_u(i));
    end
    
    if constr_u(u)> 0
        input_check = false;
        break;
    end
    
    ready_u = true;
    for k = 1:n_u
        ind_u(k) = ind_u(k)+1;
        if ind_u(k) <= ind_max_u(k)
            ready_u = false;
            break;
        end
        ind_u(k) = 1;
    end

end

if ~input_check
    fprintf('WARNING: The input space discretization is not proper.\n')
    fprintf('The following grid point does not satisfy the input constraints:')
    u
end

%==========================================================================

% Feasibility check

feas_check = true;
ind_x = ones(1,n_x);
ready_x = false;
while ~ready_x % loop over x \in X
    
    x = zeros(n_x,1);
    for i=1:n_x
        x(i) = X{i}(ind_x(i));
    end
    
    feas_check = false; %temporary
    ind_u = ones(1,n_u);
    ready_u = false;
    while ~ready_u % loop over x \in U
    
        u = zeros(n_u,1);
        for i=1:n_u
            u(i) = U{i}(ind_u(i));
        end

        if (constr_xu(x,u)<=0)
            if (constr_x(dyn(x,u))<=0)
                feas_check = true;
                break;
            end
        end
        
        ready_u = true;
        for k = 1:n_u
            ind_u(k) = ind_u(k)+1;
            if ind_u(k) <= ind_max_u(k)
                ready_u = false;
                break;
            end
            ind_u(k) = 1;
        end

    end
    
    if ~feas_check
        break;
    end

    ready_x = true;
    for k = 1:n_x
        ind_x(k) = ind_x(k)+1;
        if ind_x(k) <= ind_max_x(k)
            ready_x = false;
            break;
        end
        ind_x(k) = 1;
    end

end

if ~feas_check
    fprintf('WARNING: The state and input constriants do not satisfy the feasiblity condition.\n')
    fprintf('For the following state x in X_g, there is no control input u in U_g such that f(x,u) is in X:')
    x
end

