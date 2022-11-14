%% Code for solving model and computing moments - ALL  INDUSTRIES
%__________________________________________________________________________
% This code is directly taken from Berger et al. and only slightly
% adjusted.
% First, the code runs unchanged to compute all parameters.
% Then, the code is run again with the tax introduced, such that we obtain
% the correct Nash market equilibria.






fprintf('Running: Compute_baseline_no_tax.m\n');
%__________________________________________________________________________
% Note - This is the code for the economy *without* the corporate tax shock
% used in the estimation of the model. However, we include this code anyway and
% set the tax shock to zero. This makes sure the code is consistent with
% baseline_tax.m
%__________________________________________________________________________
clear all;close all;dbstop if error;
%__________________________________________________________________________
X                   = load('Input_parameter_vectors/xmin_B_whole_economy');
x                   = X.xmin;       % Parameters
m_data              = X.m_data;     % Target moments
%__________________________________________________________________________
% OPTIONS
options.iter_S_max  = 100       ;  % Max iterations over wage shares in Nash equilibrium solver
options.tol_S       = 1e-5      ;  % Tolerance for wage shares
options.upsilon     = 0.20      ;  % Adjustment rate of shares in equilibrium solver
%__________________________________________________________________________
% OUTPUT OPTIONS
options.print       = 'Y';
options.tradeable   = 'N'; 

glob.J              = 5000;     % Number of markets (j=1,...,J)
if strcmp(options.tradeable,'Y')
    glob.Ndist      = load('Created_mat_files/Nfit');       % Load results from Figure_F1A_Table_F1_Fit_Mj_Distribution_Tradeable_Ind.m
else
    glob.Ndist      = load('Created_mat_files/Nfit_all');   % Load results from Figure_F1B_Table_F2_Fit_Mj_Distribution_All_Ind.m
end
options.Mj_max      = max(max(glob.Ndist.M_j))      ; % Cap firms if region has more than X firms -- hard coded in fit_firm_dist_gp_II_2014

% Parameters
glob.varphi         = 0.50;             % Frisch
glob.beta           = 1/1.04;           % Discount factor
glob.delta          = 0.1;              % Depreciation rate

% Corpporate tax shock parameters
glob.share_ccorp    = 0.31;         % table_4_market_stats_v1_2014
glob.tauC_shock_size= 0;            % No corporate tax shock
glob.tauC           = 0;            % No corporate tax
glob.lambdaK 		= 0;            % No distinction between Ccorp and non-Ccorp

% Tradeable
if strcmp(options.tradeable,'Y')
    glob.AveFirmSize_data = 34.63;
    glob.AveEarnings_data = (2018*1000)/34.63;
else
    glob.AveFirmSize_data = 22.83;
    glob.AveEarnings_data = (1000*1000)/22.83;
end

%% A. CALIBRATION
options.solve_scale     = 'Y'; 
%__________________________________________________________________________
% FIX SEED
rng('default');
%__________________________________________________________________________
% SETUP / UNPACK PARAMETER VECTOR
param.eta           = x(1); 
param.theta         = x(2);
param.xi            = x(3);
param.alpha         = x(4);
param.lambdaC       = x(5);
param.Delta         = x(6);

share_ccorp         = glob.share_ccorp          ; %set this parameter in the sampling to yield actual_share_ccorp fraction of firms as ccorps
beta                = glob.beta                 ; %Discount factor
delta               = glob.delta                ; %Depreciation rate
tauC_shock_size     = glob.tauC_shock_size      ; %Shock size
tauC                = glob.tauC                 ; %Corporate tax rate
Delta               = param.Delta ;
R                   = (1/beta)-1+delta; % Rental rate=r+delta 

% Distribution of Mj parameters
param.m_Ndist       = glob.Ndist.m_Ndist;
param.sigma_Ndist   = glob.Ndist.sigma_Ndist;
param.theta_Ndist   = glob.Ndist.theta_Ndist;  
param.frac_1        = glob.Ndist.frac_1;        % Mass at 1

% Global parameters
varphi              = glob.varphi;
Mj_max              = options.Mj_max;
J                   = glob.J;                   % Number of markets
val.P               = 1;                        % Normalize to 1, final good is numeraire
%__________________________________________________________________________
% Take parameters out of param structure
eta                 = param.eta;
theta               = param.theta;
xi                  = param.xi; 
alpha               = param.alpha;
lambdaK             = glob.lambdaK;             % Fraction of capital that is debt financed
lambdaC             = param.lambdaC;            % CCorp productivity premium
 
%% 1. DRAW RANDOM VARIABLES
%__________________________________________________________________________
% 1. Draw number of firms Mj in each market j=1,...,J 
u       = rand(J,1);
Mj      = zeros(1,J);
for jj = (1:J)
   ilow = find(glob.Ndist.F_Mj<u(jj),1,'last');
   if isempty(ilow)
       ilow = 1;
   end
   Mj(jj) = glob.Ndist.M_j(ilow);
end
Mj              = Mj'; 
Mj(Mj>=Mj_max)  = Mj_max;          
%__________________________________________________________________________
% 2. Draw Ccorp status
u_ij            = random('uniform',0,1,[options.Mj_max,J])  ; 
c_ij            = zeros(options.Mj_max,J);
for jj=1:J
    c_ij(1:Mj(jj),jj) = (u_ij(1:Mj(jj),jj)<share_ccorp)      ; 
end
%__________________________________________________________________________
% 3.  Draw firm level productivities z_ij
z_ij_notilde  = zeros(options.Mj_max,J); % Productivity draws
for jj=1:J
    % xi is the standard deviation of the lognormal
    z_ij_notilde(1:Mj(jj),jj) = random('Lognormal',1,xi,[1,Mj(jj)]);
end  
% Scale up the productivity of CCorps by (1+lambdaC)
z_ij_notilde(c_ij==1)   = (1+lambdaC)*z_ij_notilde(c_ij==1);
% Recover the "\tilde{z}_ij" which is ALWAYS z_ij in the code
    % We keep the underlying 'no tilde' productivities as z_ij_notilde.
z_ij  = zeros(options.Mj_max,J); %Productivity draws
for jj = (1:J)
    z_ij(1:Mj(jj),jj)=(c_ij(1:Mj(jj),jj)==1).*(1-Delta)*(Delta*(1-tauC)/((1-tauC*lambdaK)*R)).^(Delta/(1-Delta)).*z_ij_notilde(1:Mj(jj),jj).^(1/(1-Delta))...   % Ccorp
                     +(c_ij(1:Mj(jj),jj)==0).*(1-Delta)*(Delta/R).^(Delta/(1-Delta)).*z_ij_notilde(1:Mj(jj),jj).^(1/(1-Delta)) ;                                % Non-Corp
end

%% 2. SOLVE MODEL

%% A. Solve sectoral equilibria
% Storage for outputs
what_ij         = zeros(Mj_max,J);      % Wage
mu_ij           = zeros(Mj_max,J);      % Markdown
s_ij            = zeros(Mj_max,J);      % Share
eps_ij          = zeros(Mj_max,J);      % Labor supply elasticity
nonconverge_j   = zeros(1,J);           % Non convergence dummies
iterations_j    = zeros(1,J);           % Iterations

for jj=(1:J)     % J markets
    z_i                     = z_ij(1:Mj(jj),jj); % Isolate productivity vector in region
    eq0                     = fun_solve_market_Nash(z_i,param,val,glob,options);
    s_ij(1:Mj(jj),jj)       = eq0.s_i;
    mu_ij(1:Mj(jj),jj)      = eq0.mu_i;
    what_ij(1:Mj(jj),jj)    = eq0.w_i;
    eps_ij(1:Mj(jj),jj)     = eq0.eps_i;
    nonconverge_j(:,jj)     = eq0.non_converge;
    iterations_j(:,jj)      = eq0.iterations;
end

%% B. Construct 'hat' terms
what_j                  = sum(what_ij.^(eta+1)).^(1/(eta+1));
What                    = sum(what_j.^(theta+1)).^(1/(theta+1));
nhat_ij                 = bsxfun(@times,bsxfun(@rdivide,what_ij,what_j).^eta,(what_j/What).^theta).*(What/1).^varphi;

%% C. Compute model averages
AveFirmSizehat_model    = sum(sum(nhat_ij))/sum(Mj);
AveEarningshat_model    = sum(sum(what_ij.*nhat_ij))/sum(sum(nhat_ij));
AveFirmSize_data        = glob.AveFirmSize_data; 
AveEarnings_data        = glob.AveEarnings_data; 

%% D. Compute varpibar and zbar
if strcmp(options.solve_scale,'Y')
    % Invert model conditions to obtain parameters
    varphibar           = (AveFirmSize_data/AveFirmSizehat_model)/((AveEarnings_data/AveEarningshat_model).^varphi);
    zbar                = varphibar^(1-alpha) ...
                          * (AveEarnings_data/AveEarningshat_model).^(1+(1-alpha)*varphi) ...
                          * What^(-(1-alpha)*(theta-varphi));
    glob.varphibar      = varphibar;        % Pass out to glob for later counterfactuals
    glob.zbar           = zbar;             % Pass out to glob for later counterfactuals
else
    zbar                = glob.zbar;
    varphibar           = glob.varphibar;
end

%% E. Scale up objects      
omega   = zbar/varphibar^(1-alpha);    
W       = omega^(1/(1+(1-alpha)*varphi)) * What^( (1+(1-alpha)*theta) / (1+(1-alpha)*varphi) ) ;
w_ij    = omega^(1/(1+(1-alpha)*theta)) * W^( ((1-alpha)*(theta-varphi)) / (1+(1-alpha)*theta) ) .* what_ij;
w_j     = sum(w_ij.^(eta+1)).^(1/(eta+1));
n_ij    = varphibar*bsxfun(@times,bsxfun(@rdivide,w_ij,w_j).^eta,(w_j/W).^theta).*(W/1).^varphi;
n_j     = sum(n_ij.^((eta+1)/eta)).^(eta/(eta+1));
N       = sum(n_j.^((theta+1)/theta)).^(theta/(theta+1));


%% SOLVE MODEL UNDER CORPORATE TAX SHOCK

% Taxes
tauCprime   = tauC + tauC_shock_size;
z_ij1       = (c_ij==0).*z_ij+...
              (c_ij==1).*((1-tauCprime)*(1-tauC*lambdaK)/((1-tauC)*(1-tauCprime*lambdaK))).^(Delta/(1-Delta)).*z_ij;

%% A. Solve sectoral equilibria
% Storage for outputs
what_ij1         = zeros(Mj_max,J);      % Wage
mu_ij1           = zeros(Mj_max,J);      % Markdown
s_ij1            = zeros(Mj_max,J);      % Share
eps_ij1          = zeros(Mj_max,J);      % Labor supply elasticity
nonconverge_j1   = zeros(1,J);           % Non convergence dummies
iterations_j1    = zeros(1,J);           % Iterations

for jj=(1:J)     % J regions
    z_i1                     = z_ij1(1:Mj(jj),jj); % Isolate productivity vector in region
    eq1                      = fun_solve_market_Nash(z_i1,param,val,glob,options);
    s_ij1(1:Mj(jj),jj)       = eq1.s_i;
    mu_ij1(1:Mj(jj),jj)      = eq1.mu_i;
    what_ij1(1:Mj(jj),jj)    = eq1.w_i;
    eps_ij1(1:Mj(jj),jj)     = eq1.eps_i;
    nonconverge_j1(:,jj)     = eq1.non_converge;
    iterations_j1(:,jj)      = eq1.iterations;
end

%% B. Compute varpibar and zbar
zbar            = glob.zbar;
varphibar       = glob.varphibar;

%% C. Scale up objects
omega   = zbar/varphibar^(1-alpha);   
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Since the tax shock is market specific, keep aggregate W, and solve for
% partial equilibrium:
W1      = W ;  %[PE]
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
w_ij1   = omega^(1/(1+(1-alpha)*theta)) * W1^( ((1-alpha)*(theta-varphi)) / (1+(1-alpha)*theta) ) .* what_ij1;
w_j1    = sum(w_ij1.^(eta+1)).^(1/(eta+1));
n_ij1   = varphibar*bsxfun(@times,bsxfun(@rdivide,w_ij1,w_j1).^eta,(w_j1/W1).^theta).*(W1/1).^varphi;
y_ij1   = zbar.*z_ij1.*n_ij1.^(alpha);

%% REGRESSIONS
%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
% SAMPLE SELECTION
% Firms that enter the regression sample:
i_lrg           = (z_ij>0) & (z_ij1~=z_ij) & (c_ij==1);
%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
% INPUTS
% Employment
logn_ij_lrg     = log(n_ij(i_lrg)); 
logn_ij1_lrg    = log(n_ij1(i_lrg));
d_logn_lrg      = logn_ij1_lrg-logn_ij_lrg;

% Share
s_ij_lrg        = s_ij(i_lrg) ; 
s_ij1_lrg       = s_ij1(i_lrg) ;
d_s_lrg         = s_ij1_lrg-s_ij_lrg;

% Wages
logw_ij_lrg     = log(w_ij(i_lrg)); 
logw_ij1_lrg    = log(w_ij1(i_lrg));
d_logw_lrg      = logw_ij1_lrg-logw_ij_lrg;
 
% Taxes
tauC_ij_lrg         = tauC*ones(size(z_ij1(i_lrg)))*100;    
tauC_ij1_lrg        = tauCprime*ones(size(z_ij1(i_lrg)))*100;   % Taxes
d_tauC_lrg          = tauC_ij1_lrg-tauC_ij_lrg;

% Differenced interaction term
d_s_ij_x_tauC_lrg   = tauC_ij1_lrg.*s_ij1_lrg - tauC_ij_lrg.*s_ij_lrg; 
%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
% 1. Employment regression
% Replicates the fixed effects/lags consistent with empirical regressions
% (t=0 no shocks, t=1 no shocks, t=2 uniform tax increase)
X           = [ tauC_ij_lrg-.5*tauC_ij_lrg-.5*tauC_ij1_lrg   ,  tauC_ij_lrg.*s_ij_lrg-.5*tauC_ij_lrg.*s_ij_lrg-.5*tauC_ij1_lrg.*s_ij_lrg;
                 tauC_ij1_lrg-.5*tauC_ij_lrg-.5*tauC_ij1_lrg  ,  tauC_ij1_lrg.*s_ij_lrg-.5*tauC_ij_lrg.*s_ij_lrg-.5*tauC_ij1_lrg.*s_ij_lrg];     
Y           = [logn_ij_lrg-.5*logn_ij_lrg-.5*logn_ij1_lrg;logn_ij1_lrg-.5*logn_ij_lrg-.5*logn_ij1_lrg];            
b           = (X'*X)\(X'*Y);
elast_n_t   = b(1);
elast_n_st  = b(2);
%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
% 2. Wage regression
% Replicates the fixed effects/lags consistent with empirical regressions
X           = [ tauC_ij_lrg-.5*tauC_ij_lrg-.5*tauC_ij1_lrg   ,  tauC_ij_lrg.*s_ij_lrg-.5*tauC_ij_lrg.*s_ij_lrg-.5*tauC_ij1_lrg.*s_ij_lrg;
                 tauC_ij1_lrg-.5*tauC_ij_lrg-.5*tauC_ij1_lrg  ,  tauC_ij1_lrg.*s_ij_lrg-.5*tauC_ij_lrg.*s_ij_lrg-.5*tauC_ij1_lrg.*s_ij_lrg];     
Y           = [logw_ij_lrg-.5*logw_ij_lrg-.5*logw_ij1_lrg;logw_ij1_lrg-.5*logw_ij_lrg-.5*logw_ij1_lrg];            
b           = (X'*X)\(X'*Y);
elast_w_t   = b(1);
elast_w_st  = b(2);
%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
% 3. Employment share of CcorpsSHARE OF EMPLOYMENT IN C-CORPS
empl_share_of_ccorps =  mean(sum(n_ij.*(c_ij==1))./sum(n_ij)) ;

%% COMPUTE OTHER MOMENTS
%______________________________________________________________________
% Firm level
rev_ij  = zbar*z_ij.*n_ij.^(alpha); % VA minus payments to capital
wn_ij   = w_ij.*n_ij; 
%______________________________________________________________________
% Sector level quantities
wn_j    = sum(wn_ij);
n_j     = sum(n_ij);                    % Now defined in bodies
wbar_j  = wn_j./n_j;
rev_j   = sum(rev_ij);
ls_j    = wn_j./rev_j;
sn_j    = n_j./sum(n_j);                % Employment weights
swn_j   = wn_j./sum(wn_j);              % Payroll weights
z_j     = sum(zbar*z_ij.*n_ij.^alpha)./sum(n_ij);
%______________________________________________________________________
% Within sector relative quantities
swn_ij  = bsxfun(@rdivide,wn_ij,wn_j);
sn_ij   = bsxfun(@rdivide,n_ij,n_j);
wrel_ij = bsxfun(@rdivide,w_ij,wbar_j);
%______________________________________________________________________
% Herfindahls
hhiwn_j             = sum(swn_ij.^2);
hhin_j              = sum(sn_ij.^2);
%______________________________________________________________________
% 1. Within sector correlations - Not used in paper
X1                  = ones(1,7);
%______________________________________________________________________                        
% 2. Across market correlations of HHI (unweighted)
corr_hhi_hhin       = fun_corr(hhiwn_j',hhin_j');     % corr_hhi_hhin
corr_hhi_n          = fun_corr(hhiwn_j',n_j');        % corr_hhi_tot_emp
X2                  = [1,1,corr_hhi_hhin,corr_hhi_n,1,1,1]; 
%______________________________________________________________________                        
% 3. LABOR AND CAPITAL SHARE
y_ij_scaled         = (1/(1-Delta)).*rev_ij;
LS                  = sum(sum(wn_ij))/sum(sum(y_ij_scaled));
KS                  = (Delta*(1-tauC)/(1-tauC*lambdaK))*(sum(y_ij_scaled(c_ij==1))/sum(sum(y_ij_scaled)))+Delta*(sum(y_ij_scaled(c_ij==0))/sum(sum(y_ij_scaled)));
X3                  = [KS,LS,1,1];
%______________________________________________________________________                        
% 4. HHI distribution statistics 
% 1. Unweighted
hhiwn_avg           = mean(hhiwn_j);
hhin_avg            = mean(hhin_j);
% 2. Payroll weighted
hhiwn_avg_wtd       = sum(swn_j.*hhiwn_j);
hhin_avg_wtd        = sum(swn_j.*hhin_j);
X4                  = [hhiwn_avg,1,1,hhin_avg,1,1,hhiwn_avg_wtd,1,1,hhin_avg_wtd,1,1];
%______________________________________________________________________                        
% 5. Employment, Payroll, Wage statistics 
wn_avg              = mean(wn_ij(n_ij>0)/1000);
n_avg               = mean(n_ij(n_ij>0));
X5                  = [wn_avg,1,1,n_avg,1,1,1,1,1];
%______________________________________________________________________
% 6. MOMENTS FROM THE CROSS-SECTION OF MARKETS - Not used in paper
X6                  = ones(1,19);
%______________________________________________________________________
% 7. CONSTRUCT MOMENTS FROM TAX SHOCK USING REDUCED FORM ESTIMATES
s_grid           			= [0.01:0.01:0.15];                                                 % Grid of shares
labor_supply_elast_fine 	= (elast_n_t+elast_n_st.*s_grid)./(elast_w_t+elast_w_st.*s_grid);   % Reduced form LS elast on grid
labor_supply_elast_0_5per 	= mean(labor_supply_elast_fine(1:5));                               % Average in bin
labor_supply_elast_5_10per 	= mean(labor_supply_elast_fine(6:10));                              % Average in bin
labor_supply_elast_10_15per = mean(labor_supply_elast_fine(11:15));                             % Average in bin
labor_supply_elast_coarse 	= labor_supply_elast_fine(1:2:end);
X7 = [ elast_n_t, 1, elast_n_st, elast_w_t, 1,  elast_w_st, ...
	   empl_share_of_ccorps,...
	   labor_supply_elast_0_5per, labor_supply_elast_5_10per, labor_supply_elast_10_15per,...
	   labor_supply_elast_coarse];
%______________________________________________________________________
% 8. ADDITIONAL TAX REGRESSIONS - Not used in paper
X8      = ones(1,11);

%% MOMENTS OUTPUT
STAT    = [1,X1,X2,X3,X4,X5,ones(1,55),X6,X7,X8];

%% ERROR OUTPUTS 
err1    = 0;
err2    = sum(nonconverge_j)/numel(nonconverge_j);
ERR     = [err1,err2];

%% EQUILIBRIUM OUTCOMES
Y       = sum(sum(zbar*z_ij.*n_ij.^alpha));
EQUI    = [varphibar,zbar,W,Y,N];

clearvars *_direct *_small *_lrg w_k logwbar_k  n_k omega_k *_ij1

save Created_mat_files/baseline_no_tax


























%__________________________________________________________________________
% Note - This is the code for the economy *without* the corporate tax shock
% used in the estimation of the model. However, we include this code anyway and
% set the tax shock to zero. This makes sure the code is consistent with
% baseline_tax.m
%__________________________________________________________________________
clear all;close all;dbstop if error;
%__________________________________________________________________________
X                   = load('Input_parameter_vectors/xmin_B_whole_economy');
x                   = X.xmin;       % Parameters
m_data              = X.m_data;     % Target moments
%__________________________________________________________________________
% OPTIONS
options.iter_S_max  = 100       ;  % Max iterations over wage shares in Nash equilibrium solver
options.tol_S       = 1e-5      ;  % Tolerance for wage shares
options.upsilon     = 0.20      ;  % Adjustment rate of shares in equilibrium solver
%__________________________________________________________________________
% OUTPUT OPTIONS
options.print       = 'Y';
options.tradeable   = 'N'; 

glob.J              = 5000;     % Number of markets (j=1,...,J)
if strcmp(options.tradeable,'Y')
    glob.Ndist      = load('Created_mat_files/Nfit');       % Load results from Figure_F1A_Table_F1_Fit_Mj_Distribution_Tradeable_Ind.m
else
    glob.Ndist      = load('Created_mat_files/Nfit_all');   % Load results from Figure_F1B_Table_F2_Fit_Mj_Distribution_All_Ind.m
end
options.Mj_max      = max(max(glob.Ndist.M_j))      ; % Cap firms if region has more than X firms -- hard coded in fit_firm_dist_gp_II_2014

% Parameters
glob.varphi         = 0.50;             % Frisch
glob.beta           = 1/1.04;           % Discount factor
glob.delta          = 0.1;              % Depreciation rate

% Corpporate tax shock parameters
glob.share_ccorp    = 0.31;         % table_4_market_stats_v1_2014
glob.tauC_shock_size= 0;            % No corporate tax shock
glob.tauC           = 0;            % No corporate tax
glob.lambdaK 		= 0;            % No distinction between Ccorp and non-Ccorp

% Tradeable
if strcmp(options.tradeable,'Y')
    glob.AveFirmSize_data = 34.63;
    glob.AveEarnings_data = (2018*1000)/34.63;
else
    glob.AveFirmSize_data = 22.83;
    glob.AveEarnings_data = (1000*1000)/22.83;
end

%% A. CALIBRATION
options.solve_scale     = 'Y'; 
%__________________________________________________________________________
% FIX SEED
rng('default');
%__________________________________________________________________________
% SETUP / UNPACK PARAMETER VECTOR
param.eta           = x(1); 
param.theta         = x(2);
param.xi            = x(3);
param.alpha         = x(4);
param.lambdaC       = x(5);
param.Delta         = x(6);

share_ccorp         = glob.share_ccorp          ; %set this parameter in the sampling to yield actual_share_ccorp fraction of firms as ccorps
beta                = glob.beta                 ; %Discount factor
delta               = glob.delta                ; %Depreciation rate
tauC_shock_size     = glob.tauC_shock_size      ; %Shock size
tauC                = glob.tauC                 ; %Corporate tax rate
Delta               = param.Delta ;
R                   = (1/beta)-1+delta; % Rental rate=r+delta 

% Distribution of Mj parameters
param.m_Ndist       = glob.Ndist.m_Ndist;
param.sigma_Ndist   = glob.Ndist.sigma_Ndist;
param.theta_Ndist   = glob.Ndist.theta_Ndist;  
param.frac_1        = glob.Ndist.frac_1;        % Mass at 1

% Global parameters
varphi              = glob.varphi;
Mj_max              = options.Mj_max;
J                   = glob.J;                   % Number of markets
val.P               = 1;                        % Normalize to 1, final good is numeraire
%__________________________________________________________________________
% Take parameters out of param structure
eta                 = param.eta;
theta               = param.theta;
xi                  = param.xi; 
alpha               = param.alpha;
lambdaK             = glob.lambdaK;             % Fraction of capital that is debt financed
lambdaC             = param.lambdaC;            % CCorp productivity premium
 
%% 1. DRAW RANDOM VARIABLES
%__________________________________________________________________________
% 1. Draw number of firms Mj in each market j=1,...,J 
u       = rand(J,1);
Mj      = zeros(1,J);
for jj = (1:J)
   ilow = find(glob.Ndist.F_Mj<u(jj),1,'last');
   if isempty(ilow)
       ilow = 1;
   end
   Mj(jj) = glob.Ndist.M_j(ilow);
end
Mj              = Mj'; 
Mj(Mj>=Mj_max)  = Mj_max;          
%__________________________________________________________________________
% 2. Draw Ccorp status
u_ij            = random('uniform',0,1,[options.Mj_max,J])  ; 
c_ij            = zeros(options.Mj_max,J);
for jj=1:J
    c_ij(1:Mj(jj),jj) = (u_ij(1:Mj(jj),jj)<share_ccorp)      ; 
end
%__________________________________________________________________________
% 3.  Draw firm level productivities z_ij
z_ij_notilde  = zeros(options.Mj_max,J); % Productivity draws
for jj=1:J
    % xi is the standard deviation of the lognormal
    z_ij_notilde(1:Mj(jj),jj) = random('Lognormal',1,xi,[1,Mj(jj)]);
end  

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ADDED TO COMPUTE CORRECT NASH Eq.
% the aggregates will be wrong because equations are not adjusted for the
% tax. This does not matter for us here because we are only interested in
% the solution to the labor markets, in particular for mu_ij.
% Since the code does not read in aggregates, we also don't bias future
% codes.

% Deflator 
[data,txt,raw] = xlsread('Input_data_files/prodcom_input_matlab.xlsx');  % data; empty fields will be -9999
    

par = load("cal_raw_mat_distrib.mat");
% carbon emission related parameters
par_tau_CO2 = 100;

% steel_content_intensity	aluminium_content_intensity	paper_content_intensity	cement_content_intensity
mu = [par.mu_i par.mu_a par.mu_p par.mu_c];
sigma = [par.sigma_i par.sigma_a par.sigma_p par.sigma_c];

% 1.2 translate productivities into carbon intensities

% get pdf(z_ij_notilde)
z_ij_pdf = zeros(size(z_ij_notilde));
for jj = 1:J
z_ij_pdf(1:Mj(jj),jj) = logncdf(z_ij_notilde(1:Mj(jj),jj),1,xi); 
end

% assign energy efficiency based on pdf(z_ij_notilde)
carb_ijr = zeros(Mj_max,J,size(mu,2)); % there are 4 categories for raw materials.
for ic = 1:size(mu,2)
for jj = 1:J
    carb_ijr(1:Mj(jj),jj,ic) = logninv(z_ij_pdf(1:Mj(jj),jj),mu(ic),sigma(ic));
end
end

% draw a random sector for each firm:
assign_sector = randi([1 size(data,1)],Mj_max,J);


% 1.3  Effective tax rate

% compute tau_ij based on a) energy efficiency and b) assigned sector
tau_ij = zeros(Mj_max,J); 
for jj = 1:J
    sec = assign_sector(1:Mj(jj),jj); % assigned sectors
    carb_int = sum(squeeze(carb_ijr(1:Mj(jj),jj,:)).*data(sec,2:end),2);
    if Mj(jj)==1
       carb_int = sum(squeeze(carb_ijr(1:Mj(jj),jj,:))'.*data(sec,2:end));
    end

    tau_ij(1:Mj(jj),jj) = par_tau_CO2.*carb_int./data(sec,1) ;

end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   
    

% Scale up the productivity of CCorps by (1+lambdaC)
z_ij_notilde(c_ij==1)   = (1+lambdaC)*z_ij_notilde(c_ij==1);
% Recover the "\tilde{z}_ij" which is ALWAYS z_ij in the code
    % We keep the underlying 'no tilde' productivities as z_ij_notilde.
z_ij  = zeros(options.Mj_max,J); %Productivity draws
for jj = (1:J)
    z_ij(1:Mj(jj),jj)=(c_ij(1:Mj(jj),jj)==1).*(1-Delta)*(Delta*(1-tauC)/((1-tauC*lambdaK)*R)).^(Delta/(1-Delta)).*z_ij_notilde(1:Mj(jj),jj).^(1/(1-Delta))...   % Ccorp
                     +(c_ij(1:Mj(jj),jj)==0).*(1-Delta)*(Delta/R).^(Delta/(1-Delta)).*z_ij_notilde(1:Mj(jj),jj).^(1/(1-Delta)) ;                                % Non-Corp
end

%% 2. SOLVE MODEL

%% A. Solve sectoral equilibria
% Storage for outputs
what_ij         = zeros(Mj_max,J);      % Wage
mu_ij           = zeros(Mj_max,J);      % Markdown
s_ij            = zeros(Mj_max,J);      % Share
eps_ij          = zeros(Mj_max,J);      % Labor supply elasticity
nonconverge_j   = zeros(1,J);           % Non convergence dummies
iterations_j    = zeros(1,J);           % Iterations

for jj=(1:J)     % J markets
    z_i                     = z_ij(1:Mj(jj),jj); % Isolate productivity vector in region
    eq0                     = fun_solve_market_Nash(z_i,param,val,glob,options);
    s_ij(1:Mj(jj),jj)       = eq0.s_i;
    mu_ij(1:Mj(jj),jj)      = eq0.mu_i;
    what_ij(1:Mj(jj),jj)    = eq0.w_i;
    eps_ij(1:Mj(jj),jj)     = eq0.eps_i;
    nonconverge_j(:,jj)     = eq0.non_converge;
    iterations_j(:,jj)      = eq0.iterations;
end

%% B. Construct 'hat' terms
what_j                  = sum(what_ij.^(eta+1)).^(1/(eta+1));
What                    = sum(what_j.^(theta+1)).^(1/(theta+1));
nhat_ij                 = bsxfun(@times,bsxfun(@rdivide,what_ij,what_j).^eta,(what_j/What).^theta).*(What/1).^varphi;

%% C. Compute model averages
AveFirmSizehat_model    = sum(sum(nhat_ij))/sum(Mj);
AveEarningshat_model    = sum(sum(what_ij.*nhat_ij))/sum(sum(nhat_ij));
AveFirmSize_data        = glob.AveFirmSize_data; 
AveEarnings_data        = glob.AveEarnings_data; 

%% D. Compute varpibar and zbar
if strcmp(options.solve_scale,'Y')
    % Invert model conditions to obtain parameters
    varphibar           = (AveFirmSize_data/AveFirmSizehat_model)/((AveEarnings_data/AveEarningshat_model).^varphi);
    zbar                = varphibar^(1-alpha) ...
                          * (AveEarnings_data/AveEarningshat_model).^(1+(1-alpha)*varphi) ...
                          * What^(-(1-alpha)*(theta-varphi));
    glob.varphibar      = varphibar;        % Pass out to glob for later counterfactuals
    glob.zbar           = zbar;             % Pass out to glob for later counterfactuals
else
    zbar                = glob.zbar;
    varphibar           = glob.varphibar;
end

%% E. Scale up objects      
omega   = zbar/varphibar^(1-alpha);    
W       = omega^(1/(1+(1-alpha)*varphi)) * What^( (1+(1-alpha)*theta) / (1+(1-alpha)*varphi) ) ;
w_ij    = omega^(1/(1+(1-alpha)*theta)) * W^( ((1-alpha)*(theta-varphi)) / (1+(1-alpha)*theta) ) .* what_ij;
w_j     = sum(w_ij.^(eta+1)).^(1/(eta+1));
n_ij    = varphibar*bsxfun(@times,bsxfun(@rdivide,w_ij,w_j).^eta,(w_j/W).^theta).*(W/1).^varphi;
n_j     = sum(n_ij.^((eta+1)/eta)).^(eta/(eta+1));
N       = sum(n_j.^((theta+1)/theta)).^(theta/(theta+1));


%% SOLVE MODEL UNDER CORPORATE TAX SHOCK

% Taxes
tauCprime   = tauC + tauC_shock_size;
z_ij1       = (c_ij==0).*z_ij+...
              (c_ij==1).*((1-tauCprime)*(1-tauC*lambdaK)/((1-tauC)*(1-tauCprime*lambdaK))).^(Delta/(1-Delta)).*z_ij;

%% A. Solve sectoral equilibria
% Storage for outputs
what_ij1         = zeros(Mj_max,J);      % Wage
mu_ij1           = zeros(Mj_max,J);      % Markdown
s_ij1            = zeros(Mj_max,J);      % Share
eps_ij1          = zeros(Mj_max,J);      % Labor supply elasticity
nonconverge_j1   = zeros(1,J);           % Non convergence dummies
iterations_j1    = zeros(1,J);           % Iterations

for jj=(1:J)     % J regions
    z_i1                     = z_ij1(1:Mj(jj),jj); % Isolate productivity vector in region
    eq1                      = fun_solve_market_Nash(z_i1,param,val,glob,options);
    s_ij1(1:Mj(jj),jj)       = eq1.s_i;
    mu_ij1(1:Mj(jj),jj)      = eq1.mu_i;
    what_ij1(1:Mj(jj),jj)    = eq1.w_i;
    eps_ij1(1:Mj(jj),jj)     = eq1.eps_i;
    nonconverge_j1(:,jj)     = eq1.non_converge;
    iterations_j1(:,jj)      = eq1.iterations;
end

%% B. Compute varpibar and zbar
zbar            = glob.zbar;
varphibar       = glob.varphibar;

%% C. Scale up objects
omega   = zbar/varphibar^(1-alpha);   
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Since the tax shock is market specific, keep aggregate W, and solve for
% partial equilibrium:
W1      = W ;  %[PE]
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
w_ij1   = omega^(1/(1+(1-alpha)*theta)) * W1^( ((1-alpha)*(theta-varphi)) / (1+(1-alpha)*theta) ) .* what_ij1;
w_j1    = sum(w_ij1.^(eta+1)).^(1/(eta+1));
n_ij1   = varphibar*bsxfun(@times,bsxfun(@rdivide,w_ij1,w_j1).^eta,(w_j1/W1).^theta).*(W1/1).^varphi;
y_ij1   = zbar.*z_ij1.*n_ij1.^(alpha);

%% REGRESSIONS
%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
% SAMPLE SELECTION
% Firms that enter the regression sample:
i_lrg           = (z_ij>0) & (z_ij1~=z_ij) & (c_ij==1);
%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
% INPUTS
% Employment
logn_ij_lrg     = log(n_ij(i_lrg)); 
logn_ij1_lrg    = log(n_ij1(i_lrg));
d_logn_lrg      = logn_ij1_lrg-logn_ij_lrg;

% Share
s_ij_lrg        = s_ij(i_lrg) ; 
s_ij1_lrg       = s_ij1(i_lrg) ;
d_s_lrg         = s_ij1_lrg-s_ij_lrg;

% Wages
logw_ij_lrg     = log(w_ij(i_lrg)); 
logw_ij1_lrg    = log(w_ij1(i_lrg));
d_logw_lrg      = logw_ij1_lrg-logw_ij_lrg;
 
% Taxes
tauC_ij_lrg         = tauC*ones(size(z_ij1(i_lrg)))*100;    
tauC_ij1_lrg        = tauCprime*ones(size(z_ij1(i_lrg)))*100;   % Taxes
d_tauC_lrg          = tauC_ij1_lrg-tauC_ij_lrg;

% Differenced interaction term
d_s_ij_x_tauC_lrg   = tauC_ij1_lrg.*s_ij1_lrg - tauC_ij_lrg.*s_ij_lrg; 
%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
% 1. Employment regression
% Replicates the fixed effects/lags consistent with empirical regressions
% (t=0 no shocks, t=1 no shocks, t=2 uniform tax increase)
X           = [ tauC_ij_lrg-.5*tauC_ij_lrg-.5*tauC_ij1_lrg   ,  tauC_ij_lrg.*s_ij_lrg-.5*tauC_ij_lrg.*s_ij_lrg-.5*tauC_ij1_lrg.*s_ij_lrg;
                 tauC_ij1_lrg-.5*tauC_ij_lrg-.5*tauC_ij1_lrg  ,  tauC_ij1_lrg.*s_ij_lrg-.5*tauC_ij_lrg.*s_ij_lrg-.5*tauC_ij1_lrg.*s_ij_lrg];     
Y           = [logn_ij_lrg-.5*logn_ij_lrg-.5*logn_ij1_lrg;logn_ij1_lrg-.5*logn_ij_lrg-.5*logn_ij1_lrg];            
b           = (X'*X)\(X'*Y);
elast_n_t   = b(1);
elast_n_st  = b(2);
%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
% 2. Wage regression
% Replicates the fixed effects/lags consistent with empirical regressions
X           = [ tauC_ij_lrg-.5*tauC_ij_lrg-.5*tauC_ij1_lrg   ,  tauC_ij_lrg.*s_ij_lrg-.5*tauC_ij_lrg.*s_ij_lrg-.5*tauC_ij1_lrg.*s_ij_lrg;
                 tauC_ij1_lrg-.5*tauC_ij_lrg-.5*tauC_ij1_lrg  ,  tauC_ij1_lrg.*s_ij_lrg-.5*tauC_ij_lrg.*s_ij_lrg-.5*tauC_ij1_lrg.*s_ij_lrg];     
Y           = [logw_ij_lrg-.5*logw_ij_lrg-.5*logw_ij1_lrg;logw_ij1_lrg-.5*logw_ij_lrg-.5*logw_ij1_lrg];            
b           = (X'*X)\(X'*Y);
elast_w_t   = b(1);
elast_w_st  = b(2);
%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
% 3. Employment share of CcorpsSHARE OF EMPLOYMENT IN C-CORPS
empl_share_of_ccorps =  mean(sum(n_ij.*(c_ij==1))./sum(n_ij)) ;

%% COMPUTE OTHER MOMENTS
%______________________________________________________________________
% Firm level
rev_ij  = zbar*z_ij.*n_ij.^(alpha); % VA minus payments to capital
wn_ij   = w_ij.*n_ij; 
%______________________________________________________________________
% Sector level quantities
wn_j    = sum(wn_ij);
n_j     = sum(n_ij);                    % Now defined in bodies
wbar_j  = wn_j./n_j;
rev_j   = sum(rev_ij);
ls_j    = wn_j./rev_j;
sn_j    = n_j./sum(n_j);                % Employment weights
swn_j   = wn_j./sum(wn_j);              % Payroll weights
z_j     = sum(zbar*z_ij.*n_ij.^alpha)./sum(n_ij);
%______________________________________________________________________
% Within sector relative quantities
swn_ij  = bsxfun(@rdivide,wn_ij,wn_j);
sn_ij   = bsxfun(@rdivide,n_ij,n_j);
wrel_ij = bsxfun(@rdivide,w_ij,wbar_j);
%______________________________________________________________________
% Herfindahls
hhiwn_j             = sum(swn_ij.^2);
hhin_j              = sum(sn_ij.^2);
%______________________________________________________________________
% 1. Within sector correlations - Not used in paper
X1                  = ones(1,7);
%______________________________________________________________________                        
% 2. Across market correlations of HHI (unweighted)
corr_hhi_hhin       = fun_corr(hhiwn_j',hhin_j');     % corr_hhi_hhin
corr_hhi_n          = fun_corr(hhiwn_j',n_j');        % corr_hhi_tot_emp
X2                  = [1,1,corr_hhi_hhin,corr_hhi_n,1,1,1]; 
%______________________________________________________________________                        
% 3. LABOR AND CAPITAL SHARE
y_ij_scaled         = (1/(1-Delta)).*rev_ij;
LS                  = sum(sum(wn_ij))/sum(sum(y_ij_scaled));
KS                  = (Delta*(1-tauC)/(1-tauC*lambdaK))*(sum(y_ij_scaled(c_ij==1))/sum(sum(y_ij_scaled)))+Delta*(sum(y_ij_scaled(c_ij==0))/sum(sum(y_ij_scaled)));
X3                  = [KS,LS,1,1];
%______________________________________________________________________                        
% 4. HHI distribution statistics 
% 1. Unweighted
hhiwn_avg           = mean(hhiwn_j);
hhin_avg            = mean(hhin_j);
% 2. Payroll weighted
hhiwn_avg_wtd       = sum(swn_j.*hhiwn_j);
hhin_avg_wtd        = sum(swn_j.*hhin_j);
X4                  = [hhiwn_avg,1,1,hhin_avg,1,1,hhiwn_avg_wtd,1,1,hhin_avg_wtd,1,1];
%______________________________________________________________________                        
% 5. Employment, Payroll, Wage statistics 
wn_avg              = mean(wn_ij(n_ij>0)/1000);
n_avg               = mean(n_ij(n_ij>0));
X5                  = [wn_avg,1,1,n_avg,1,1,1,1,1];
%______________________________________________________________________
% 6. MOMENTS FROM THE CROSS-SECTION OF MARKETS - Not used in paper
X6                  = ones(1,19);
%______________________________________________________________________
% 7. CONSTRUCT MOMENTS FROM TAX SHOCK USING REDUCED FORM ESTIMATES
s_grid           			= [0.01:0.01:0.15];                                                 % Grid of shares
labor_supply_elast_fine 	= (elast_n_t+elast_n_st.*s_grid)./(elast_w_t+elast_w_st.*s_grid);   % Reduced form LS elast on grid
labor_supply_elast_0_5per 	= mean(labor_supply_elast_fine(1:5));                               % Average in bin
labor_supply_elast_5_10per 	= mean(labor_supply_elast_fine(6:10));                              % Average in bin
labor_supply_elast_10_15per = mean(labor_supply_elast_fine(11:15));                             % Average in bin
labor_supply_elast_coarse 	= labor_supply_elast_fine(1:2:end);
X7 = [ elast_n_t, 1, elast_n_st, elast_w_t, 1,  elast_w_st, ...
	   empl_share_of_ccorps,...
	   labor_supply_elast_0_5per, labor_supply_elast_5_10per, labor_supply_elast_10_15per,...
	   labor_supply_elast_coarse];
%______________________________________________________________________
% 8. ADDITIONAL TAX REGRESSIONS - Not used in paper
X8      = ones(1,11);

%% MOMENTS OUTPUT
STAT    = [1,X1,X2,X3,X4,X5,ones(1,55),X6,X7,X8];

%% ERROR OUTPUTS 
err1    = 0;
err2    = sum(nonconverge_j)/numel(nonconverge_j);
ERR     = [err1,err2];

%% EQUILIBRIUM OUTCOMES
Y       = sum(sum(zbar*z_ij.*n_ij.^alpha));
EQUI    = [varphibar,zbar,W,Y,N];

clearvars *_direct *_small *_lrg w_k logwbar_k  n_k omega_k *_ij1

save('Created_mat_files/baseline_no_tax_correct_mu','mu_ij')

