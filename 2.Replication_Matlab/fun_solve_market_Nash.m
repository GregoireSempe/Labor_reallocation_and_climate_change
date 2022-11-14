function out = fun_solve_market_Nash(z_i,param,val,glob,options)        
%__________________________________________________________________________
% UNPACK
theta           = param.theta;
eta             = param.eta;
alpha           = param.alpha;                          % This alpha is the alpha_tilde in the text
upsilon         = options.upsilon;
%__________________________________________________________________________
% Starting guess (equal shares)
s_i             = z_i/sum(z_i);
for iter_S = (1:options.iter_S_max)
  % Firm level elasticity of labor supply
  eps_i             =   ((s_i.*(1/theta) + (1-s_i).*(1/eta)).^(-1));
  % Markdown
  mu_i              = eps_i./(eps_i+1);
  %________________________________________________________________________
  % Wage
  a1                = 1/(1+(1-alpha).*theta);
  a2                = -(1-alpha)*(eta-theta)/(eta+1);
  w_i               = (mu_i.*alpha.*z_i.*s_i.^a2).^a1;  % These z_i's are \tilde{z}_{ij} in the text
  %________________________________________________________________________
  % Sectoral Wage (CES index)
  W_j               = sum(w_i.^(1+eta)).^(1/(1+eta));
  % Implied shares
  s_i_new           = (w_i./W_j).^(1+eta);  
  % Distance of shares and new shares
  dist_S            = max(abs(s_i_new-s_i));
  if (dist_S<options.tol_S)
      break;
  end
  % Update S slowly
  s_i               = upsilon*s_i_new + (1-upsilon)*s_i;
  s_i               = s_i/sum(s_i); 
end
%__________________________________________________________________________
% Packup output
out.eps_i           = eps_i;
out.s_i             = s_i;
out.mu_i            = mu_i;
out.w_i             = w_i; 
%__________________________________________________________________________
% Diagnostics
out.non_converge    = (iter_S==options.iter_S_max);
out.iterations      = (iter_S);

end