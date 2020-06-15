%% ===================================================================== %%
% The code is (C) 2014-2015 Filipe I. Fazanaro, Diogo C. Soriano, Ricardo
% Suyama, Marconi K. Madrid, Jose Raimundo de Oliveira, Ignacio B. Munoz,
% Romis Attux and is released under the GPL v3 license. 
%
% Sample code for the "Numerical Characterization of Nonlinear Dynamical
% Systems Using Parallel Computing: The Role of GPUs Approach" paper, which
% was submitted to the Communications of Nonlinear Science and Numerical
% Computation. 
%
%
% Initial version: October 22, 2014
% Last update: June 15, 2020
%
%% ===================================================================== %%
%% DESCRIPTION
%
%   - Time response of the system defined by the Duffing forced oscillator
%   dynamical model [1,2].
%
%   - Employs the 'ode45' integrator.
%
%
%% ===================================================================== %%
%% REFERENCES
% [1] Parker and Chua (1989) Practical Numerical algorithms for Chaotic
% Systems. Springer-Verlag.
%
% [2] Guckenheimer, Holmes (2002) Nonlinear Oscillations, Dynamical
% systems, and Bifurcations of Vector Fields. Springer. 2nd edition.
%
% [3] http://mmae.iit.edu/shadden/LCS-tutorial/overview.html
%
% [4] Soriano, Fazanaro, Suyama, JRO, Attux and Madrid (2012) A method for
% Lyapunov spectrum estimation using cloned dynamics and its application
% to the discontinuously-excited FitzHugh-Nagumo model. Nonlinear Dynamics,
% vol 67, no 1, pp 413-424.
% DOI: 10.1007/s11071-011-9989-2.
%% ===================================================================== %%

clc
close all
clear all


%% ===================================================================== %%
%% DYNAMICAL SYSTEM DIMENSION

dim = 3;


dim_total = dim;

%% ===================================================================== %%
%% PARAMETERS

epsilon = 0.22;

gamma = 0.3;

omega = 1;

%% ===================================================================== %%
%% INITIAL CONDITIONS OF THE DYNAMICAL SYSTEM

vX0 = 1.0;
vY0 = 1.0;
vZ0 = 0.0;

% Complete system
y_init = [ vX0, vY0, vZ0 ];

%% ===================================================================== %%
%% INTEGRATION PARAMETERS

% Initial time
t_init = 0;

% Final time
t_final = 2500;

% Transient
t_trans = 1000;

% Time step
t_step = 0.01;


% Time span
tspan = t_init:t_step:t_final;


% Transient index
index_trans = round(t_trans/t_step);

% 'ode45' options
Options = odeset('RelTol',1e-8,'AbsTol',1e-8);


%% ===================================================================== %%

fprintf ( 1, '\n  Computation begun at %s.\n\n', datestr ( now ) );

tbase = tic();

%% ===================================================================== %%
%% DYNAMICAL SYSTEM INTEGRATION

[T,Y] = ode45(@(t,y) Duffing1989 ( t, y, gamma, epsilon, omega ), tspan, y_init, Options);

%% ===================================================================== %%
%% SCREEN - DEBUG

% for ii = 1:size(Y,1)
%     
%     for jj = 1:dim_total
%         
%         fprintf( '[%.6f]', Y(ii,jj) );
%         
%     end
%     
%     fprintf( '\n' );
%     
% end

%% ===================================================================== %%

fprintf ( 1, '\n  Computation completed at %s.\n', datestr ( now ) );

cpuTime = toc( tbase );

fprintf('Total execution time [s] = %.4f\n\n',cpuTime);

%% ===================================================================== %%
%% SOME USEFULL STRINGS

% Uncomment as you wish

% strEpsilon = ['Epsilon_' num2str( epsilon ) ];
% strEpsilon( ismember(strEpsilon, '-') ) = 'n';
% strEpsilon( ismember(strEpsilon, '.') ) = 'p';
% 
% 
% strGamma = ['Gamma_' num2str( gamma ) ];
% strGamma( ismember(strGamma, '-') ) = 'n';
% strGamma( ismember(strGamma, '.') ) = 'p';
% 
% 
% strOmega = ['Omega_' num2str( omega ) ];
% strOmega( ismember(strOmega, '-') ) = 'n';
% strOmega( ismember(strOmega, '.') ) = 'p';

%% ===================================================================== %%
%% SAVE THE DATA

% Uncomment as you wish

% sSave = ['save <put_the_name_here>.mat'];
% eval(sSave);

%% ===================================================================== %%
%% PLOT

close all

% ----------------------------------------------------------------------- %

set(0,'DefaultTextFontName','Times', ...
    'DefaultTextFontSize', 18, ...
    'DefaultAxesFontName', 'Times', ...
    'DefaultAxesFontSize', 18, ...
    'DefaultLineLineWidth', 1, ...
    'DefaultLineMarkerSize', 7.75);

% ----------------------------------------------------------------------- %
% http://dgleich.wordpress.com/2013/06/04/creating-high-quality-graphics-in-matlab-for-papers-and-presentations/
%
% Defaults for this blog post

width = 3;                  % Width in inches

height = 3;                 % Height in inches

AxesLineWidth = 0.75;       % AxesLineWidth

FontSize = 18;              % Fontsize

LFontSize = 18;             % Legend Fontsize

AnnotationFontSize = 20;    % Annotation Fontsize

LineWidth = 1;              % LineWidth

MarkerSize = 2;             % MarkerSize

% ----------------------------------------------------------------------- %

hFig1 = figure( 'visible', 'on' );
hax1 = axes( 'Parent', hFig1, 'FontSize', FontSize );

box( hax1, 'on' );
hold(hax1, 'all' );
grid(hax1, 'off' );



plot( T(:,1), Y(:,1), ...
    'LineStyle', '-', ...
    'Color', [0 0 1] );



% Adjusting the x- and y-labels positioning
set( hax1, 'Units', 'Normalized' );

pos = get( hax1, 'Position' );

offset = 0.01;

set( hax1, ...
    'XGrid'       , 'off'                        , ...
    'YGrid'       , 'off'                        , ...
    'Position'    , pos + [0, offset, 0, -offset] );

hx = xlabel( 'Time [u.a.]', 'Interpreter', 'latex' );
hy = ylabel( '$x$', 'Interpreter', 'latex' );

set(hx, 'Units', 'Normalized');
set(hy, 'Units', 'Normalized');

pos_xlabel = get(hx, 'Position');
pos_ylabel = get(hy, 'Position');

set(hx, 'Position', pos_xlabel + [0, (-0)*offset, 0]);
set(hy, 'Position', pos_ylabel + [-offset, 0, 0]);


% ----------------------------------------------------------------------- %

% Print

% sGraficoEPS = ['print -depsc2 -r600 test_name.eps'];
% eval(sGraficoEPS);
%
% sGraficoJPG = ['print -djpeg -r600 test_name.jpg'];
% eval(sGraficoJPG);
%
% sGraficoPNG = ['print -dpng -r600 test_name.png'];
% eval(sGraficoPNG);

% ----------------------------------------------------------------------- %

hFig2 = figure( 'visible', 'on' );
hax2 = axes( 'Parent', hFig2, 'FontSize', FontSize );

box( hax2, 'on' );
hold(hax2, 'all' );
grid(hax2, 'off' );


plot( T(index_trans:end), Y(index_trans:end,1), ...
    'LineStyle', '-', ...
    'Color', [0 0 1] );


% axis([-2 3 -2 3]);


% Adjusting the x- and y-labels positioning
set( hax2, 'Units', 'Normalized' );

pos = get( hax2, 'Position' );

offset = 0.01;

set( hax2, ...
    'XGrid'       , 'off'                        , ...
    'YGrid'       , 'off'                        , ...
    'Position'    , pos + [0, offset, 0, -offset] );

hx = xlabel( 'Time [u.a.]', 'Interpreter', 'latex' );
hy = ylabel( '$x$', 'Interpreter', 'latex' );

set(hx, 'Units', 'Normalized');
set(hy, 'Units', 'Normalized');

pos_xlabel = get(hx, 'Position');
pos_ylabel = get(hy, 'Position');

set(hx, 'Position', pos_xlabel + [0, (-0)*offset, 0]);
set(hy, 'Position', pos_ylabel + [-offset, 0, 0]);

% ----------------------------------------------------------------------- %

% Print

% sGraficoEPS = ['print -depsc2 -r600 <put_the_name_here>.eps'];
% eval(sGraficoEPS);

% sGraficoJPG = ['print -djpeg -r600 <put_the_name_here>.jpg'];
% eval(sGraficoJPG);

% sGraficoPNG = ['print -dpng -r600 <put_the_name_here>.png'];
% eval(sGraficoPNG);

% ----------------------------------------------------------------------- %

hFig3 = figure( 'visible', 'on' );
hax3 = axes( 'Parent', hFig3, 'FontSize', FontSize );

box( hax3, 'on' );
hold(hax3, 'all' );
grid(hax3, 'off' );


plot( Y(index_trans:end,1), Y(index_trans:end,2), ...
    'LineStyle', '-', ...
    'Color', [0 0 1] );


axis([-2 2 -1.5 1.5]);


% Adjusting the x- and y-labels positioning
set( hax3, 'Units', 'Normalized' );

pos = get( hax3, 'Position' );

offset = 0.01;

set( hax3, ...
    'XGrid'       , 'off'                        , ...
    'YGrid'       , 'off'                        , ...
    'Position'    , pos + [0, offset, 0, -offset] );

hx = xlabel( '$x$', 'Interpreter', 'latex' );
hy = ylabel( '$y$', 'Interpreter', 'latex' );

set(hx, 'Units', 'Normalized');
set(hy, 'Units', 'Normalized');

pos_xlabel = get(hx, 'Position');
pos_ylabel = get(hy, 'Position');

set(hx, 'Position', pos_xlabel + [0, (-0)*offset, 0]);
set(hy, 'Position', pos_ylabel + [-offset, 0, 0]);

% ----------------------------------------------------------------------- %

% Print

% sGraficoEPS = ['print -depsc2 -r600 <put_the_name_here>.eps'];
% eval(sGraficoEPS);

% sGraficoJPG = ['print -djpeg -r600 <put_the_name_here>.jpg'];
% eval(sGraficoJPG);

% sGraficoPNG = ['print -dpng -r600 <put_the_name_here>.png'];
% eval(sGraficoPNG);

% ======================================================================= %





%% ===================================================================== %%
%% EOF: End Of File
%% ===================================================================== %%