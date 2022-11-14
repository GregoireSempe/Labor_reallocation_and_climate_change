%% Calibration of sectoral carbon intensities

clear all
% parameters that determine the distribution of carbon intensities

% iron
avg_i   = 1.52;
med_i   = 1.443;
ten_i   = 1.33;

mu_i    = log(med_i);
sigma_i = 0.7*2*(log(avg_i)-mu_i);

% aluminimum
avg_a   = 1.681;
med_a   = 1.604;
ten_a   = 1.484;

mu_a    = log(med_a);
sigma_a = 0.2*2*(log(avg_a)-mu_i);


% cement
avg_c   = 0.9575;
med_c   = 0.9555;
ten_c   = 0.8475;

mu_c    = log(med_c);
sigma_c = 22*2*(log(avg_c)-mu_c);


% % glass
% avg_g = 0.4695;
% med_g = 0.44375;
% ten_g = 0.32475;
% 
% mu_g    = log(med_g);
% sigma_g = 2.1*2*(log(avg_g)-mu_g);

% paper
avg_p   = 0.5405;
med_p   = 0.215;
ten_p   = 0.009;

mu_p    = log(med_p);
sigma_p = 2*(log(avg_p)-mu_p);


save cal_raw_mat_distrib.mat mu_a sigma_a mu_p sigma_p mu_c sigma_c mu_i sigma_i


% construct grids to visualise calibration
grid_i = linspace(0.6,avg_i+2*(avg_i-ten_i),100);
grid_a = linspace(0.6,avg_a+2*(avg_a-ten_a),100);
grid_c = linspace(0.3,avg_c+2*(avg_c-ten_c),100);
% grid_g = linspace(0.1,avg_g+2*(avg_g-ten_g),100);
grid_p = linspace(0.01,avg_p+2*(avg_p-ten_p),100);


plot_i = logncdf(grid_i,mu_i,sigma_i);
plot_a = logncdf(grid_a,mu_a,sigma_a);
plot_c = logncdf(grid_c,mu_c,sigma_c);
% plot_g = logncdf(grid_g,mu_g,sigma_g);
plot_p = logncdf(grid_p,mu_p,sigma_p);


figure_style_template;


figure(1)
f1 = tiledlayout(4,2);


% iron
nexttile(f1)
plot(grid_i,plot_i);
hold on
% plot([ten_i ten_i],[0 0.2],'--','Color',red)
% plot([0 avg_i+(avg_i-ten_i)],[0.1 0.1],'--','Color',red)
% plot([med_i med_i],[0 0.6],'--','Color',green)
% plot([0 avg_i+(avg_i-ten_i)],[0.5 0.5],'--','Color',green)
plot([ten_i],[0.1],'x','Color',red)
plot([med_i],[0.5],'x','Color',green)
hold off
subtitle('CDF: Iron')
legend('CDF', '10^{th} Percentile','Median','Location','southeast')

nexttile(f1)
plot(grid_i,lognpdf(grid_i,mu_i,sigma_i));
subtitle('Distribution: Iron')


% aluminium
nexttile(f1)
plot(grid_a,plot_a);
hold on
% plot([ten_a ten_a],[0 0.2],'--','Color',red)
plot([ten_a],[0.1],'x','Color',red)
% plot([0 avg_a+(avg_a-ten_a)],[0.1 0.1],'--','Color',red)
% plot([med_a med_a],[0 0.6],'--','Color',green)
plot([med_a],[0.5],'x','Color',green)
% plot([0 avg_a+(avg_a-ten_a)],[0.5 0.5],'--','Color',green)
hold off
subtitle('CDF: Aluminium')
legend('CDF', '10^{th} Percentile','Median','Location','southeast')

nexttile(f1)
plot(grid_a,lognpdf(grid_a,mu_a,sigma_a));
subtitle('Distribution: Aluminium')

% cement
nexttile(f1)
plot(grid_c,plot_c);
hold on
% plot([ten_c ten_c],[0 0.2],'--','Color',red)
% plot([0 avg_c+(avg_c-ten_c)],[0.1 0.1],'--','Color',red)
% plot([med_c med_c],[0 0.6],'--','Color',green)
% plot([0 avg_c+(avg_c-ten_c)],[0.5 0.5],'--','Color',green)
plot([ten_c],[0.1],'x','Color',red)
plot([med_c],[0.5],'x','Color',green)
hold off
subtitle('CDF: Cement')
legend('CDF', '10^{th} Percentile','Median','Location','southeast')

nexttile(f1)
plot(grid_c,lognpdf(grid_c,mu_c,sigma_c));
subtitle('Distribution: Cement')

% % glass
% nexttile(f1)
% % f4 = tiledlayout(1,2);
% % nexttile(f4)
% plot(grid_g,plot_g);
% hold on
% plot([ten_g ten_g],[0 0.2],'--','Color',red)
% plot([0 avg_g+(avg_g-ten_g)],[0.1 0.1],'--','Color',red)
% plot([med_g med_g],[0 0.6],'--','Color',green)
% plot([0 avg_g+(avg_g-ten_g)],[0.5 0.5],'--','Color',green)
% hold off
% 
% % nexttile(f4)
% nexttile(f1)
% plot(grid_g,lognpdf(grid_g,mu_g,sigma_g));

% paper
% f5 = tiledlayout(1,2);
% nexttile(f5)
nexttile(f1)
plot(grid_p,plot_p);
hold on
% plot([ten_p ten_p],[0 0.2],'--','Color',red)
% plot([0 avg_p+(avg_p-ten_p)],[0.1 0.1],'--','Color',red)
% plot([med_p med_p],[0 0.6],'--','Color',green)
% plot([0 avg_p+(avg_p-ten_p)],[0.5 0.5],'--','Color',green)
plot([ten_p],[0.1],'x','Color',red)
plot([med_p],[0.5],'x','Color',green)
hold off
subtitle('CDF: Paper')
legend('CDF', '10^{th} Percentile','Median','Location','southeast')

% nexttile(f5)
nexttile(f1)
plot(grid_p,lognpdf(grid_p,mu_p,sigma_p));
subtitle('Distribution: Paper')

