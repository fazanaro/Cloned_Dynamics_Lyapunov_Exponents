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
% Initial version: October 09, 2013
% Last update: June 15, 2020
%
%% ===================================================================== %%
%% DESCRIPTION
%
%   - Calculation of the Lyapunov global exponents - using the Tangent Map
%   methodology - for the forced Duffing oscillator dynamical system
%   [1,2].
%
%   - Control parameters: 'epsilon' and 'omega'.
%
%   - Employs the fourth-order Runge Kutta integrator.
%
%
%% ===================================================================== %%
%% REFERENCES
% [1] Parker and Chua (1989) Practical Numerical algorithms for Chaotic
% Systems. Springer-Verlag.
%
% [2] Guckenheimer, Holmes (2002) Nonlinear Oscillations, Dynamical
% Systems, and Bifurcations of Vector Fields. Springer. 2nd edition.
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

gamma = 0.3;

%% ===================================================================== %%
%% CONTROL PARAMETERS

% Variation step of the control parameter
nIncControlParameter1 = 0.1;
nIncControlParameter2 = 0.1;


vEpsilon = [0.2, 0.22, 0.25];%0.0:nIncControlParameter1:0.5;

vOmega   = [0.5, 1.0];%0.9:nIncControlParameter2:1.4;


% Defines the meshgrid
[mGridEpsilon, mGridOmega] = meshgrid(vEpsilon, vOmega);


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
h = 0.01;

% Maximum iteration number
nMaxItera = round( (t_final-t_init)/t_gsr );


% Calculate the tangent map
deltax = zeros( dim );


% Memory allocation: these vectors will store the maximum (minimum)
% Lyapunov exponents
Mz_Lyap1 = NaN( size(mGridEpsilon) );
Mz_Lyap2 = NaN( size(mGridEpsilon) );
Mz_Lyap3 = NaN( size(mGridEpsilon) );

% Used during the construction of the space parameter
mStatesLocalLyapClDyn_Lyap1 = [];
mStatesLocalLyapClDyn_Lyap2 = [];
mStatesLocalLyapClDyn_Lyap3 = [];

%% ===================================================================== %%


fprintf ( 1, '\n  Computation begun at %s.\n\n', datestr ( now ) );

tbase = tic();


%% ===================================================================== %%
%% MAIN

for ij = 1:numel(mGridEpsilon)
    
    % ------------------------------------------------------------------- %
    % SCREEN
    % ------------------------------------------------------------------- %
    
    fprintf( '%d / %d\n', ij, numel(mGridEpsilon) );
    
    % ------------------------------------------------------------------- %
    
    t_aux = 0;
    
    % ------------------------------------------------------------------- %
    % PARAMETER SPACE
    % ------------------------------------------------------------------- %
    
    epsilon = mGridEpsilon(ij);
    
    omega = mGridOmega(ij);
    
    % ------------------------------------------------------------------- %
    % VECTOR PARAMETER
    % ------------------------------------------------------------------- %

    vParameters = [ epsilon, gamma, omega ];
    
    % ------------------------------------------------------------------- %
    % INITIAL CONDITIONS
    % ------------------------------------------------------------------- %
    
    % Initial conditions
    vX0 = 1;
    vY0 = 1;
    vZ0 = 0;
    
    y_init_orig  = [ vX0, vY0, vZ0 ];
    
    y_init = [ y_init_orig, reshape(eye(dim),1,[]) ];
    
    % ------------------------------------------------------------------- %
    % GRAM-SCHIMIDT REORTHONORMALIZATION INITIALIZATION
    % ------------------------------------------------------------------- %
    
    vk = zeros( dim );
    uk = eye( dim );
    
    % ------------------------------------------------------------------- %
    % MEMORY ALLOCATION
    % ------------------------------------------------------------------- %
    
    LyapSoma  = zeros( dim, nMaxItera );
    LyapLocal = zeros( dim, nMaxItera );
    Lyap      = zeros( dim, nMaxItera );
    
    % ------------------------------------------------------------------- %
    
    for ii = 1:nMaxItera
        
        % --------------------------------------------------------------- %
        
        [T, Y] = odeRK4v3( @odefun_Duffing1989_TanMap, (t_aux:h:(t_aux+t_gsr)), h, y_init, vParameters, dim_total );
        
        % --------------------------------------------------------------- %
        
        t_aux = t_aux + t_gsr;
        
        % --------------------------------------------------------------- %
        % You only need the conditions at the end of the 't_gsr' interval
        
        y = Y( end, : );
        
        Fi_T = reshape( y((dim+1):end), dim, [] )';
        
        % --------------------------------------------------------------- %
        % Tangent map
        
        deltax = Fi_T*uk;
        
        % --------------------------------------------------------------- %
        % Gram-Schimidt reorthonormalization
        
        [vk, uk, Normk] = GSR2( deltax, dim );
        
        % --------------------------------------------------------------- %
        % Lyapunov exponents calculation
        
        for jj = 1:dim
            if Normk(jj) ~= 0
                % Local exponents
                LyapLocal(jj,ii+1) = (1/t_gsr)*log(Normk(jj));
                
                % Local exponents summation
                LyapSoma(jj,ii+1) = LyapSoma(jj,ii) + log(Normk(jj));
                
                % Gloabl exponents
                Lyap(jj,ii+1) = (1/(t_aux-t_init))*LyapSoma(jj,ii+1);
            end
        end
        
        % --------------------------------------------------------------- %
        % Prepare for the next iteration
        
        % Resets the vector with initial conditions making the identity
        % matrix for the variational equation
        
        y_init = [ y(1:dim), reshape(eye(dim),1,[]) ];
        
        % --------------------------------------------------------------- %
        
    end
    
    % ------------------------------------------------------------------- %
    
    tempStatesLocalLyapClDyn_Lyap1 = [mGridEpsilon(ij), mGridOmega(ij), ...
        Lyap(:,end)', Lyap(1,end) ];%max(Lyap(:,end)) ];
    
    tempStatesLocalLyapClDyn_Lyap2 = [mGridEpsilon(ij), mGridOmega(ij), ...
        Lyap(:,end)', Lyap(2,end) ];
    
    tempStatesLocalLyapClDyn_Lyap3 = [mGridEpsilon(ij), mGridOmega(ij), ...
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


fprintf ( 1, '\n\n  Computation completed at %s.\n', datestr ( now ) );

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


surf(mGridEpsilon, mGridOmega, Mz_Lyap1);


colormap jet;
shading interp;
view([0 90]);


axis([ min(vEpsilon) max(vEpsilon) min(vOmega) max(vOmega) ]);


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


surf(mGridEpsilon, mGridOmega, Mz_Lyap2);


colormap jet;
shading interp;
view([0 90]);


axis([ min(vEpsilon) max(vEpsilon) min(vOmega) max(vOmega) ]);


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


surf(mGridEpsilon, mGridOmega, Mz_Lyap3);


colormap jet;
shading interp;
view([0 90]);


axis([ min(vEpsilon) max(vEpsilon) min(vOmega) max(vOmega) ]);


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