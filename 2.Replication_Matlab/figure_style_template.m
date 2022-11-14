%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%       Change Matlab default figure style for current session        %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% The new defaults will not take effect if there are any open figures. To
% use them, we close all figures, and then repeat the first example.
close all;

% adjust values:
% ===============

width = 6;     % Width in inches
height = 5;    % Height in inches
alw = 0.75;    % AxesLineWidth
fsz = 11;      % Fontsize
lw = 1.5;      % LineWidth
msz = 8;       % MarkerSize


% adjust standard colors:
% Blau: [0 0.4470 0.7410]
% Rot:  [0.8500 0.3250 0.0980]
% Gelb: [0.9290 0.6940 0.1250]
% Gruen: [0.4667    0.6745    0.1882]

newcolors = [0 0.4470 0.7410
             0.8500 0.3250 0.0980
             0.9290 0.6940 0.1250
             0.4667    0.6745    0.1882];
         
colororder(newcolors)

red = newcolors(2,:)
green = newcolors(4,:)


% The properties we've been using in the figures
set(0,'defaultLineLineWidth',lw);   % set the default line width to lw
set(0,'defaultLineMarkerSize',msz); % set the default line marker size to msz
% set(0,'defaultLineLineWidth',lw);   % set the default line width to lw
% set(0,'defaultLineMarkerSize',msz); % set the default line marker size to msz

% set display area
set(0, 'DefaultFigurePosition', [1 1  600 500]);


% Set the default Size for display
defpos = get(0,'defaultFigurePosition');
set(0,'defaultFigurePosition', [defpos(1) defpos(2) width*100, height*100]);

% Set the defaults for saving/printing to a file
set(0,'defaultFigureInvertHardcopy','on'); % This is the default anyway
set(0,'defaultFigurePaperUnits','inches'); % This is the default anyway
defsize = get(gcf, 'PaperSize');
left = (defsize(1)- width)/2;
bottom = (defsize(2)- height)/2;
defsize = [left, bottom, width, height];
set(0, 'defaultFigurePaperPosition', defsize);
