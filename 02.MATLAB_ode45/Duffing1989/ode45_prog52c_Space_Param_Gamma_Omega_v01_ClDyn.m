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
%   - Calculation of the Lyapunov global exponents - using the Cloned
%   Dynamics approach - for the forced Duffing oscillator dynamical system
%   [1,2].
%
%   - Control parameters: 'gamma' and 'omega'.
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
% [4] Wolf, Swift, Swinney and Vastano (1985) Determining Lyapunov
% exponents from time series, Physica 16D, pp 285-317.
% DOI:10.1016/0167-2789(85)90011-9
%
% [5] Soriano, Fazanaro, Suyama, JRO, Attux and Madrid (2012) A method for
% Lyapunov spectrum estimation using cloned dynamics and its application
% to the discontinuously-excited FitzHugh-Nagumo model, Nonlinear Dynamics,
% vol 67, no 1, pp 413-424.
% DOI: 10.1007/s11071-011-9989-2.
%% ===================================================================== %%

clc
close all
clear all


%% ===================================================================== %%
%% DYNAMICAL SYSTEM DIMENSION

dim = 3;

dim_total = dim*(dim+1);

%% ===================================================================== %%
%% PARAMETERS

epsilon = 0.25;

%% ===================================================================== %%
%% CONTROL PARAMETERS

% Variation step of the control parameter
nIncControlParameter1 = 0.1;
nIncControlParameter2 = 0.1;


vGamma = [0.2, 0.3, 0.5];%0.1:nIncControlParameter1:1.0;

vOmega = [0.8, 1.0, 1.2, 1.4, 1.6];%0.8:nIncControlParameter2:1.6;


% Defines the meshgrid
[mGridGamma, mGridOmega] = meshgrid(vGamma, vOmega);


%% ===================================================================== %%
%% INTEGRATION DEFINITIONS AND INITIAL CONDITIONS

% Initial time
t_init = 0;

% Aux
t_aux = 0;

% Final time
t_final = 1000;

% Gram-Schimidt reorthonormalization time interval
t_gsr = 0.5;

% Time step
t_step = 0.01;

% Maximum iteration number
nMaxItera = round( (t_final-t_init)/t_gsr );


% Cloned Dynamics approach aux
deltax = zeros( dim );

% Gram-Schimidt Reorthonormalization
vk = zeros( dim );
uk = zeros( dim );

% 'ode45' options
Options = odeset('RelTol',1e-8,'AbsTol',1e-8);


% Memory allocation: these vectors will store the maximum (minimum)
% Lyapunov exponents
Mz_Lyap1 = NaN( size(mGridGamma) );
Mz_Lyap2 = NaN( size(mGridGamma) );
Mz_Lyap3 = NaN( size(mGridGamma) );

% Used during the construction of the space parameter
mStatesLocalLyapClDyn_Lyap1 = [];
mStatesLocalLyapClDyn_Lyap2 = [];
mStatesLocalLyapClDyn_Lyap3 = [];


% Aux
Yfinal = [];

%% ===================================================================== %%


fprintf ( 1, '\n  Computation begun at %s.\n\n', datestr ( now ) );

tbase = tic();


%% ===================================================================== %%
%% MAIN

for ij = 1:numel(mGridGamma)
    
    % ------------------------------------------------------------------- %
    % SCREEN
    % ------------------------------------------------------------------- %
    
    fprintf( '%d / %d\n', ij, numel(mGridGamma) );
    
    % ------------------------------------------------------------------- %
    
    t_aux = 0;
    
    % ------------------------------------------------------------------- %
    % PARAMETER SPACE
    % ------------------------------------------------------------------- %
    
    gamma = mGridGamma(ij);
    
    omega = mGridOmega(ij);
    
    % ------------------------------------------------------------------- %
    % INITIAL CONDITIONS
    % ------------------------------------------------------------------- %
    
    % Cloned dynamical systems perturbation - see [5]
    delta = 1e-4;
    
    
    % Initial conditions
    vX0 = 1;
    vY0 = 1;
    vZ0 = 0;
    
    y_init_orig  = [ vX0, vY0, vZ0 ];
    
    
    % Cloned dynamical systems initial conditions
    y_init_clon = (ones(dim,1)*y_init_orig)' + delta*eye(dim);
    
    
    % Fiducial and clones initial conditions
    y_init = [ y_init_orig, reshape(y_init_clon',1,[])];
    
    % ------------------------------------------------------------------- %
    % MEMORY ALLOCATION
    % ------------------------------------------------------------------- %
    
    LyapSoma  = zeros( dim, nMaxItera );
    LyapLocal = zeros( dim, nMaxItera );
    Lyap      = zeros( dim, nMaxItera );
    
    % ------------------------------------------------------------------- %
    
    for ii = 1:nMaxItera
        
        % --------------------------------------------------------------- %
        
        [T,Y] = ode45(@(t,y) Duffing1989_ClDyn ( t, y, gamma, epsilon, omega ), (t_aux:t_step:(t_aux+t_gsr)), y_init, Options);
        
        % --------------------------------------------------------------- %
        
        t_aux = t_aux + t_gsr;
        
        % --------------------------------------------------------------- %
        % You only need the conditions at the end of the 't_gsr' interval
        
        y_orig = Y( end, 1:dim )';
        y_clon = Y( end, (dim+1):(dim*(dim+1)));
        
        y_clon = reshape( y_clon, dim, [] )';
        
        % --------------------------------------------------------------- %
        % Difference state vectors
        
        deltax = y_orig*ones( 1, dim ) - y_clon;
        
        % --------------------------------------------------------------- %
        % Gram-Schimidt reorthonormalization
        
        [vk, uk, Normk] = GSR2( deltax, dim );
        
        % --------------------------------------------------------------- %
        % Lyapunov exponents calculation
        
        for jj = 1:dim
            if Normk(jj) ~= 0
                % Local exponents
                LyapLocal(jj,ii+1) = (1/t_gsr)*log(Normk(jj)/delta);
                
                % Local exponents summation
                LyapSoma(jj,ii+1) = LyapSoma(jj,ii) + log(Normk(jj)/delta);
                
                % Gloabl exponents
                Lyap(jj,ii+1) = (1/(t_aux-t_init))*LyapSoma(jj,ii+1);
            end
        end
        
        % --------------------------------------------------------------- %
        % Prepare for the next iteration
        
        % Fiducial and cloned dynamical systems initial conditions for the
        % next iteration
        
        y_init_orig = y_orig';
        y_init_clon = (ones(dim,1)*y_init_orig)' + delta*uk;
        
        y_init = [ y_init_orig, reshape(y_init_clon',1,[]) ];
        
        % --------------------------------------------------------------- %
        
    end
    
    % ------------------------------------------------------------------- %
    
    tempStatesLocalLyapClDyn_Lyap1 = [mGridGamma(ij), mGridOmega(ij), ...
        Lyap(:,end)', Lyap(1,end) ];%max(Lyap(:,end)) ];
    
    tempStatesLocalLyapClDyn_Lyap2 = [mGridGamma(ij), mGridOmega(ij), ...
        Lyap(:,end)', Lyap(2,end) ];
    
    tempStatesLocalLyapClDyn_Lyap3 = [mGridGamma(ij), mGridOmega(ij), ...
        Lyap(:,end)', Lyap(3,end) ];%min(Lyap(:,end)) ];
    
    % ------------------------------------------------------------------- %
    
    Mz_Lyap1(ij) = tempStatesLocalLyapClDyn_Lyap1(end,end);
    Mz_Lyap2(ij) = tempStatesLocalLyapClDyn_Lyap2(end,end);
    Mz_Lyap3(ij) = tempStatesLocalLyapClDyn_Lyap3(end,end);
    
    mStatesLocalLyapClDyn_Lyap1(ij,:) = tempStatesLocalLyapClDyn_Lyap1;
    mStatesLocalLyapClDyn_Lyap2(ij,:) = tempStatesLocalLyapClDyn_Lyap2;
    mStatesLocalLyapClDyn_Lyap3(ij,:) = tempStatesLocalLyapClDyn_Lyap3;
    
    % ------------------------------------------------------------------- %
    
end

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

% sSave = ['save <put_the_name_of_the_file_here>.mat'];
% evEpsilonl(sSave);

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


surf(mGridGamma, mGridOmega, Mz_Lyap1);


colormap jet;
shading interp;
view([0 90]);


axis([ min(vGamma) max(vGamma) min(vOmega) max(vOmega) ]);


hcb = colorbar;
colorTitleHandle = get( hcb, 'Title' );
titleString = '\lambda_{1}';
set( colorTitleHandle , 'String', titleString, 'FontSize', 14 );


xlabel( '$\epsilon$', 'Interpreter', 'latex' );
ylabel( '$\gamma$', 'Interpreter', 'latex' );

% ----------------------------------------------------------------------- %

% Print
% sGraficoEPS = ['print -depsc2 fig_bifurc_Lyap_vAlpha_' num2str(vAlpha(1)) 'a' num2str(vAlpha(end)) '.eps'];
% eval(sGraficoEPS);

% ----------------------------------------------------------------------- %

hFig2 = figure( 'visible', 'on' );
hax2 = axes( 'Parent', hFig2, 'FontSize', FontSize );

box( hax2, 'on' );
hold(hax2, 'all' );
grid(hax2, 'off' );


surf(mGridGamma, mGridOmega, Mz_Lyap2);


colormap jet;
shading interp;
view([0 90]);


axis([ min(vGamma) max(vGamma) min(vOmega) max(vOmega) ]);


hcb = colorbar;
colorTitleHandle = get( hcb, 'Title' );
titleString = '\lambda_{2}';
set( colorTitleHandle , 'String', titleString, 'FontSize', 14 );


xlabel( '$\epsilon$', 'Interpreter', 'latex' );
ylabel( '$\gamma$', 'Interpreter', 'latex' );

% ----------------------------------------------------------------------- %

% Print
% sGraficoEPS = ['print -depsc2 fig_bifurc_Lyap_vAlpha_' num2str(vAlpha(1)) 'a' num2str(vAlpha(end)) '.eps'];
% eval(sGraficoEPS);

% ----------------------------------------------------------------------- %

hFig3 = figure( 'visible', 'on' );
hax3 = axes( 'Parent', hFig3, 'FontSize', FontSize );

box( hax3, 'on' );
hold(hax3, 'all' );
grid(hax3, 'off' );


surf(mGridGamma, mGridOmega, Mz_Lyap3);


colormap jet;
shading interp;
view([0 90]);


axis([ min(vGamma) max(vGamma) min(vOmega) max(vOmega) ]);


hcb = colorbar;
colorTitleHandle = get( hcb, 'Title' );
titleString = '\lambda_{3}';
set( colorTitleHandle , 'String', titleString, 'FontSize', 14 );


xlabel( '$\epsilon$', 'Interpreter', 'latex' );
ylabel( '$\gamma$', 'Interpreter', 'latex' );

% ----------------------------------------------------------------------- %

% Print
% sGraficoEPS = ['print -depsc2 fig_bifurc_Lyap_vAlpha_' num2str(vAlpha(1)) 'a' num2str(vAlpha(end)) '.eps'];
% eval(sGraficoEPS);

% ======================================================================= %




%% ===================================================================== %%
%% EOF: End Of File
%% ===================================================================== %%