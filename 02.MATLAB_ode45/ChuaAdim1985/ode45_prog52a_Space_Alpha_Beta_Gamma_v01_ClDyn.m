%%% ===================================================================== %%
% The code is (C) 2014-2020 Filipe I. Fazanaro, Diogo C. Soriano, Ricardo
% Suyama, Marconi K. Madrid, Jose Raimundo de Oliveira, Ignacio B. Munoz,
% Romis Attux and is released under the GPL v3 license. 
%
% Sample code based on the supplementary material associated to the work
% "Numerical Characterization of Nonlinear Dynamical Systems Using 
% Parallel Computing: The Role of GPUs Approach", which was published
% by the Communications of Nonlinear Science and Numerical Computation. 
% DOI: 10.1016/j.cnsns.2015.12.021
%
% git: https://github.com/fazanaro/Cloned_Dynamics_Lyapunov_Exponents.git
%
% ------------------------------------------------------
% Author:           Filipe Ieda Fazanaro
% Contact:          filipe.fazanaro AT ufabc.edu.br
% Initial version:  v05.02.2013.01
% Last update:      v19.06.2020.01
% ------------------------------------------------------
%
%% ===================================================================== %%
%% DESCRIPTION
%
%   - Calculation of the Lyapunov global exponents - using the Clonded
%   Dynamics approach - for the Chua's circuit oscillator dynamical model 
%   [3,4].
%
%   - Control parameters: 'alpha' and 'beta'
%
%   - Employs the 'ode45' integrator.
%
%% ===================================================================== %%
%% OBSERVATIONS
%
%   - Core i7 2600K 3.4GHz, MATLAB R2020a
%       - alpha variation step: 0.4
%       - alpha = [7.0; 9.0]
%
%       - beta variation step: 4.0
%       - beta = [70.0; 200.0]/7
%
%       - Total combinations = 
%       - Total execution time [s] = 
%
%
%% ===================================================================== %%
%% REFERENCES
%
% [1] Parker and Chua (1989) Practical Numerical algorithms for Chaotic
% Systems. Springer-Verlag.
%
% [2] Soriano, Fazanaro, Suyama, JRO, Attux and Madrid (2012) A method for
% Lyapunov spectrum estimation using cloned dynamics and its application
% to the discontinuously-excited FitzHugh-Nagumo model. Nonlinear Dynamics,
% vol 67, no 1, pp 413-424.
% DOI: 10.1007/s11071-011-9989-2.
%
% [3] Matsumoto, Chua, Komuro (1985) The Double Scroll. IEEE Transactions
% on Circuits and Systems CAS-32(8), pp 798-818.
%
% [4] Arena, Baglio, Fortuna, Manganaro (1995) Chua's Circuit Can be
% Generated by CNN Cells. IEEE Transactions on circuits ans Systems - I:
% fundamental, Theory and Applications, vol 42, no 2, February, pp 123-125.
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
%
%   - See [3] for aditional details

% -------------------------------------------------------------------------
% C1 = 1/7;
% C2 = 1/1;
% L  = 1/7;
% RL = 0;
% G  = 0.7;
% m0 = -0.5;
% m1 = -0.8;
% Bp = 1;
% 
% alpha = C2/C1;
% beta  = C2/(L*G^2);
% gamma = (RL*C2)/(G*L);
% 
% a = m1/G;
% b = m0/G;
% -------------------------------------------------------------------------

% -------------------------------------------------------------------------
% alpha = 9;
% beta  = 100/7;
gamma = 0;

a = -8/7;
b = -5/7;
% -------------------------------------------------------------------------

%% ===================================================================== %%
%% CONTROL PARAMETERS

% Variation step of the control parameter
nIncControlParameter1 = 0.4;
nIncControlParameter2 = 4.0;


vAlpha = 7.0:nIncControlParameter1:9.0;

vBeta = 70.0:nIncControlParameter2:200;
vBeta = vBeta/7.0;


% Defines the meshgrid
[mGridAlpha, mGridBeta] = meshgrid(vAlpha, vBeta);


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
Mz_Lyap1 = NaN( size(mGridAlpha) );
Mz_Lyap2 = NaN( size(mGridAlpha) );
Mz_Lyap3 = NaN( size(mGridAlpha) );

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

for ij = 1:numel(mGridAlpha)
    
    % ------------------------------------------------------------------- %
    % SCREEN
    % ------------------------------------------------------------------- %
    
    fprintf( '%d / %d\n', ij, numel(mGridAlpha) );
    
    % ------------------------------------------------------------------- %
    
    t_aux = 0;
    
    % ------------------------------------------------------------------- %
    % PARAMETER SPACE
    % ------------------------------------------------------------------- %
    
    alpha = mGridAlpha(ij);
    
    beta = mGridBeta(ij);
    
    % ------------------------------------------------------------------- %
    % INITIAL CONDITIONS
    % ------------------------------------------------------------------- %
    
    % Cloned dynamical systems perturbation - see [2]
    delta = 1e-4;
    
    
    % Initial conditions
    vX0 = 0.15264;
    vY0 = -0.02281;
    vZ0 = 0.38127;
    
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
        
        [T,Y] = ode45(@(t,y) ChuaAdim1985_ClDyn( t, y, alpha, beta, gamma, a, b ), (t_aux:t_step:(t_aux+t_gsr)), y_init, Options);
        
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
    
    tempStatesLocalLyapClDyn_Lyap1 = [mGridAlpha(ij), mGridBeta(ij), ...
        Lyap(:,end)', Lyap(1,end) ];%max(Lyap(:,end)) ];
    
    tempStatesLocalLyapClDyn_Lyap2 = [mGridAlpha(ij), mGridBeta(ij), ...
        Lyap(:,end)', Lyap(2,end) ];
    
    tempStatesLocalLyapClDyn_Lyap3 = [mGridAlpha(ij), mGridBeta(ij), ...
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

% stralpha = ['alpha_' num2str( alpha ) ];
% stralpha( ismember(stralpha, '-') ) = 'n';
% stralpha( ismember(stralpha, '.') ) = 'p';
%
%
% strbeta = ['beta_' num2str( beta ) ];
% strbeta( ismember(strbeta, '-') ) = 'n';
% strbeta( ismember(strbeta, '.') ) = 'p';
%
%
% strOmega = ['Omega_' num2str( omega ) ];
% strOmega( ismember(strOmega, '-') ) = 'n';
% strOmega( ismember(strOmega, '.') ) = 'p';

%% ===================================================================== %%
%% SAVE THE DATA

% Uncomment as you wish

% sSave = ['save <put_the_name_of_the_file_here>.mat'];
% evAlphal(sSave);

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


surf(mGridAlpha, mGridBeta, Mz_Lyap1);


colormap jet;
shading interp;
view([0 90]);


axis([ min(vAlpha) max(vAlpha) min(vBeta) max(vBeta) ]);


hcb = colorbar;
colorTitleHandle = get( hcb, 'Title' );
titleString = '\lambda_{1}';
set( colorTitleHandle , 'String', titleString, 'FontSize', 14 );


xlabel( '$\alpha$', 'Interpreter', 'latex' );
ylabel( '$\beta$', 'Interpreter', 'latex' );

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


surf(mGridAlpha, mGridBeta, Mz_Lyap2);


colormap jet;
shading interp;
view([0 90]);


axis([ min(vAlpha) max(vAlpha) min(vBeta) max(vBeta) ]);


hcb = colorbar;
colorTitleHandle = get( hcb, 'Title' );
titleString = '\lambda_{2}';
set( colorTitleHandle , 'String', titleString, 'FontSize', 14 );


xlabel( '$\alpha$', 'Interpreter', 'latex' );
ylabel( '$\beta$', 'Interpreter', 'latex' );

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


surf(mGridAlpha, mGridBeta, Mz_Lyap3);


colormap jet;
shading interp;
view([0 90]);


axis([ min(vAlpha) max(vAlpha) min(vBeta) max(vBeta) ]);


hcb = colorbar;
colorTitleHandle = get( hcb, 'Title' );
titleString = '\lambda_{3}';
set( colorTitleHandle , 'String', titleString, 'FontSize', 14 );


xlabel( '$\alpha$', 'Interpreter', 'latex' );
ylabel( '$\beta$', 'Interpreter', 'latex' );

% ----------------------------------------------------------------------- %

% Print
% sGraficoEPS = ['print -depsc2 fig_bifurc_Lyap_vAlpha_' num2str(vAlpha(1)) 'a' num2str(vAlpha(end)) '.eps'];
% eval(sGraficoEPS);

% ======================================================================= %




%% ===================================================================== %%
%% EOF: End Of File
%% ===================================================================== %%