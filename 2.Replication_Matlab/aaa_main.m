clear;clc;
tic

%% Options (speed)
calibrate = 0;

%% A. RUN FILES
%__________________________________________________________________________
% CALIBRATE DISTRIBUTION OF NUMBER OF FIRMS ACROSS MARKETS
if calibrate == 1
calibration;
Figure_F1B_Table_F2_Fit_Mj_Distribution_All_Ind;       
    % (Table*) - Distribution of firms across markets, M j ~ G ( M j ) , all industries
    % (Figure*) - Distribution of the number of firms across sectors, all industries
    % Outputs: Nfit, Nfit_all, used in Compute_* code

Compute_baseline_no_tax;                % Uses xmin_B_whole_economy.mat 
    % Outputs: baseline_no_tax
end
%% solves the model:

% 
par_tau_CO2 = 0;
% load parameters from estimation of Berger et al (2022)
load('Created_mat_files/baseline_no_tax','param','glob','mu_ij','z_ij','Mj','theta','eta','Mj_max','J','xi','Delta','R');
% load('Created_mat_files/baseline_no_tax_correct_mu','mu_ij'); % mu_ij corrected for tau_ij
model;

%
clear all;
par_tau_CO2 = 100;
% load parameters from estimation of Berger et al (2022)
load('Created_mat_files/baseline_no_tax','param','glob','z_ij','Mj','theta','eta','Mj_max','J','xi','Delta','R');
load('Created_mat_files/baseline_no_tax_correct_mu','mu_ij'); % mu_ij corrected for tau_ij
model; % with







%% preliminary analysis
clear all
baseline = load('results/results_Co2_0','out');
tax = load('results/results_Co2_100.mat');


% fall in sectoral employment in %
fall_nj = 100*(tax.out.nj-baseline.out.nj)./baseline.out.nj;

% fall in sectoral wage in %
fall_wj = 100*(tax.out.wj-baseline.out.wj)./baseline.out.wj;

% sort local labor markets by carbon intensity
[sortj_c,sortj_c_id] = sort(sum(tax.c_int_ij));

% average change of the markets with the highest impact
perc_1 = ceil(0.01*5000);
perc_5 = ceil(0.05*5000);
perc_10 = ceil(0.1*5000);

% labor
aux = fall_nj(sortj_c_id);
avg_fall_nj_p1 = sum(aux(perc_1:end))/perc_1;
avg_fall_nj_p5 = sum(aux(perc_5:end))/perc_5;
avg_fall_nj_p10 = sum(aux(perc_10:end))/perc_10;

% wages
aux = fall_wj(sortj_c_id);
avg_fall_wj_p1 = sum(aux(perc_1:end))/perc_1;
avg_fall_wj_p5 = sum(aux(perc_5:end))/perc_5;
avg_fall_wj_p10 = sum(aux(perc_10:end))/perc_10;





% aggregate intensity




toc
