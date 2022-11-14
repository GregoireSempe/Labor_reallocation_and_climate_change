function [beta_uw,beta_w] = fun_regression(x,y,w)

% If no weights specified use ones
if (nargin < 3)
    w       = ones(size(x));
end

X           = [ones(size(x)),x];
W           = diag(w);

% Unweighted
Beta_uw     = (X'*X)\(X'*y);
beta_uw     = Beta_uw(2);

% Weighted
Beta_w      = (X'*W*X)\(X'*W*y);
beta_w      = Beta_w(2);
