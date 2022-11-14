%% Figure F1B and Table F1 - And determination of G(Mj) parameters - ALL INDUSTRIES
clc
clear all
close all
fprintf('Running: Figure F1B and Table F1 - And determination of G(Mj) parameters - ALL INDUSTRIES\n');
%__________________________________________________________________________
% DATA ALL
avg_target          =   113.10	 ;
std_target          =   619.00	 ;
skew_target         =   26.14    ;
no_markets          =   52000    ; 
frac_1              =   0.09419  ;       % Fraction of markets with 1 firm

no_markets_geq2     = floor(no_markets*(1-frac_1));
no_markets_eq1      = floor(no_markets*frac_1);
%__________________________________________________________________________
% Set up grid of parameters to optimize over
m_big               = [0.69 : 0.0025 : 0.74]; % Corresponds to k tail index (shape) parameter
sigma_big           = [38   :  0.005  : 39];    % Scale parameter
%__________________________________________________________________________
% Loop over grid of possible parameter values
for ii=1:length(m_big)
    ii
    for jj=1:length(sigma_big)
        %______________________________________________________________
        m           = m_big(ii);
        sigma       = sigma_big(jj);
        theta       = 2;  % Lower bound must be 2 as know mass at 1
        %______________________________________________________________
        % DRAW FROM DISTRIBUTIONS
        rand('seed',1010);
        dist_geq2_a     = random('gp',m,sigma,theta,[floor(no_markets_geq2),1]);        % Draw for non-degenerate markets
        dist_geq2       = [dist_geq2_a];
        tot_dist        = [dist_geq2;ones(no_markets_eq1,1)];                           % Add back in degerenate markets
        tot_dist        = floor(tot_dist);
        %______________________________________________________________
        % COMPUTE MOMENTS
        avg_store(ii,jj)     = mean(tot_dist);
        std_store(ii,jj)     = std(tot_dist);
        skew_store(ii,jj)    = skewness(tot_dist);
        %______________________________________________________________
        % STORE
        m_store(ii,jj)       = m;
        sigma_store(ii,jj)   = sigma;
        theta_store(ii,jj)   = theta;
        %______________________________________________________________
    end
end
%__________________________________________________________________________
% Distance metric
dist            = abs(avg_store-avg_target)/avg_target+abs(std_store-std_target)/std_target;%+abs(skew_store-skew_target)/skew_target;
dist_min        = min(min(min(dist)));
I               = find(dist==dist_min);
%__________________________________________________________________________
% Best fit
m_Ndist         = m_store(I);
sigma_Ndist     = sigma_store(I);
theta_Ndist     = theta_store(I); 

diary('Created_text_files/firm_dist_stats_all.txt')
display(['mean fit:' num2str(avg_store(I)) ' : Target :' num2str(avg_target)]) ;
display(['std fit:' num2str(std_store(I)) ' : Target :' num2str(std_target)]) ;
display(['skew fit:' num2str(skew_store(I)) ' : Target :' num2str(skew_target)]) ;

display(['m*:'     num2str(m_Ndist)])
display(['sigma*:' num2str(sigma_Ndist)])
display(['theta*:' num2str(theta_store(I))]) 
diary off
%__________________________________________________________________________
no_markets          =   250000;
no_markets_geq2     = floor(no_markets*(1-frac_1));
no_markets_eq1      = floor(no_markets*frac_1);

dist_geq2_a         = random('gp',m_store(I),sigma_store(I),theta_store(I),[no_markets_geq2,1]);          % Draw for non-degenerate markets
dist_geq2           = [dist_geq2_a];
tot_dist            = [dist_geq2;ones(no_markets_eq1,1)]; % Add back in degerenate markets
tot_dist            = floor(tot_dist);
dist_geq2_a         = floor(dist_geq2_a);
                       
n_trunc                             = 200;
tot_dist(tot_dist>n_trunc)          = n_trunc;
dist_geq2_a(dist_geq2_a>n_trunc)    = n_trunc;

% Bins for plot
histbin     = [0.5:1:100.5];
hist1       = hist(tot_dist,histbin); 
hist1       = hist1/sum(hist1); 
histbin     = histbin+0.5;

cut = 101;
xx  = find(histbin<cut,1,'last');
hist1(xx) = sum(hist1(xx:end));

xi      = m_store(I);
mu      = theta_store(I);
sigma   = sigma_store(I); 

%% FIGURE
F1      = figure(1);
set(F1,'Pos',[1.9877e+03 -59 806 636.6667]);
font    = 28;
set(0,'defaultlinelinewidth',6);
set(0,'defaultlinemarkersize',6);

plot(histbin(1:xx),hist1(1:xx),'ko-');hold on;
l = legend('$G\left(M_j\right)$');
grid on;
xlabel('Number of firms $M_j$','fontsize',font','interpreter','latex');
ylabel('Fraction of markets','fontsize',font','interpreter','latex');
set(l,'fontsize',font-6,'interpreter','latex','location','NorthWest','orientation','vertical');
set(gca,'fontsize',font-6,'TickLabelInterpreter','latex');
set(gca,'XTick',[0,1,2,10:10:100]);
xlim([1,100]);
ylim([0,.3]);
set(gcf,'PaperPositionMode','auto');
clear print
print -depsc Created_figure_files/Figure_F1B_Fit_Mj_Distribution_All_Ind.eps

%% TABLE
fprintf('../OUTPUT/Table_F2_Fit_Mj_Distribution_All_Ind\n');
eval(['fid = fopen(''Created_table_files/Table_F2_Fit_Mj_Distribution_All_Ind.tex'',''w'')']);
fprintf(fid,'\\begin{tabular}{lccc}\n');
fprintf(fid,'\\toprule[2pt]\n');
fprintf(fid,'A. Moments &   & &    \\\\ \n');  
fprintf(fid,'Distribution of firms $M_j$ & Mean  & Std. Dev & Skewnewss   \\\\ \n');   
fprintf(fid,'\\midrule   \n');
fprintf(fid,'Data (LBD 2014) &   %2.2f    &  %2.2f  & %2.2f    \\\\ \n', avg_target , std_target ,skew_target )
fprintf(fid,'Model           &   %2.2f    &  %2.2f  & %2.2f    \\\\ \n', avg_store(I) , std_store(I),skew_store(I));
fprintf(fid,'\\toprule[2pt]  \n');
fprintf(fid,'B. Parameters &   & &    \\\\ \n');  
fprintf(fid,'Mass at $M_j=1$ & Pareto Tail  & Pareto Scale & Pareto Location   \\\\ \n'); 
fprintf(fid,'\\midrule \n');
fprintf(fid,'%2.2f  &   %2.2f    &  %2.2f  & %2.2f    \\\\ \n', frac_1, m_Ndist, sigma_Ndist, theta_Ndist );
fprintf(fid,'\\bottomrule \n');
fprintf(fid,'\\end{tabular}');
fclose(fid);

%% SAVE OUTPUT FOR USE IN MAIN CODE

M_j     = tot_dist;
M_j     = sort(M_j);
f_Mj    = ones(numel(M_j),1)*1/numel(M_j);
F_Mj    = cumsum(f_Mj);
 
% Keep only unique values of M_j
[~,i]   = unique(M_j);
M_j     = M_j(i);
F_Mj    = F_Mj(i);

% SAVE OUTPUT FOR USE IN MAIN CODE
save('Created_mat_files/Nfit_all','m_Ndist','sigma_Ndist','theta_Ndist','frac_1','M_j','F_Mj');
 