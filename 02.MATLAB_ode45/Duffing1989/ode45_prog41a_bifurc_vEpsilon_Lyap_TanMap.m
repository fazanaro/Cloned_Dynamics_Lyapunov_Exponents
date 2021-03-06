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
% Initial version: October 21, 2014
% Last update: June 15, 2020
%
%% ===================================================================== %%
%% DESCRIPTION
%
%   - Calculation of the Lyapunov global exponents - using the Tangent Map
%   methodology - for the forced Duffing oscillator dynamical system
%   [1,2].
%
%   - Defines 'epsilon' as the control parameter.
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

gamma = 0.3;

omega = 1.0;

%% ===================================================================== %%
%% CONTROL PARAMETERS

% Variation step of the control parameter
nIncControlParameter = 0.02;

% Control parameter
vEpsilon = 0.20:nIncControlParameter:0.25;


%% ===================================================================== %%
%% INTEGRATION PARAMETERS

% Initial time
t_init = 0;


% Final time
t_final = 1000;


% Time step
t_step = 0.01;


% Gram-Schimidt reorthonormalization time interval
%   - See [5] for further details related to how to change this value.
t_gsr = 0.5;


% Maximum iteration number
nMaxItera = round( (t_final-t_init)/t_gsr );


% Store the Lyapunov spectrum
vBifurcLyap = 255*ones( length(vEpsilon), (dim+1) );


% 'ode45' options
Options = odeset('RelTol',1e-8,'AbsTol',1e-8);

%% ===================================================================== %%

fprintf ( 1, '\n  Computation begun at %s.\n', datestr ( now ) );

tbase = tic();

% ===================================================================== %%
%% DYNAMICAL SYSTEM INTEGRATION

for iEpsilon = 1:length(vEpsilon)
    
    % ------------------------------------------------------------------- %
    
    epsilon = vEpsilon(iEpsilon);
    
    % ------------------------------------------------------------------- %
    
    t_aux = 0;
    
    % ------------------------------------------------------------------- %
    % INITIAL CONDITIONS OF THE DYNAMICAL SYSTEM AND THE VARIATIONAL
    % EQUATION
    % ------------------------------------------------------------------- %
    
    % Initial conditions
    vX0 = 1;
    vY0 = 1;
    vZ0 = 0;
    
    y_init_orig  = [ vX0, vY0, vZ0 ];
    
    
    % Complete system initial conditions
    y_init = [ y_init_orig, reshape(eye(dim),1,[]) ];
    
    % ------------------------------------------------------------------- %
    % USED DURING THE TANGENT MAP CALCULATION
    % ------------------------------------------------------------------- %
    
    deltax = zeros( dim );
    
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
        
        [T,Y] = ode45(@(t,y) Duffing1989_TanMap ( t, y, gamma, epsilon, omega ), (t_aux:t_step:(t_aux+t_gsr)), y_init, Options);
        
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
        % Gram-Schimidt Reorthonormalization
        
        [vk, uk, Normk] = GSR2(deltax, dim);
        
        % --------------------------------------------------------------- %
        % Lyapunov exponents calculation
        
        for jj = 1:dim
            if norm(vk(:,jj)) ~= 0
                % Local exponent
                LyapLocal(jj,ii+1) = (1/t_gsr)*log(norm(vk(:,jj)));
                
                % Local exponents summation
                LyapSoma(jj,ii+1) = LyapSoma(jj,ii) + log(norm(vk(:,jj)));
                
                % Global Lyapunov exponents
                Lyap(jj,ii+1) = (1/(t_aux-t_init))*LyapSoma(jj,ii+1);
            end
        end
        
        % --------------------------------------------------------------- %
        % PREPARE FOR THE NEXT ITERATION
        % --------------------------------------------------------------- %
        
        % Resets the vector with initial conditions making the identity
        % matrix for the variational equation
        
        y_init = [ y(1:dim), reshape(eye(dim),1,[]) ];
        
        % --------------------------------------------------------------- %
        
    end
    
    % ------------------------------------------------------------------- %
    
    vBifurcLyap(iEpsilon,:) = [epsilon, Lyap(:,end)']
    
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

figure

% plot(vBifurcLyap(:,1),0.0,...
%     'Marker', '.', ...
%     'LineStyle','-', ...
%     'Color',[0 0 0]);

line([min(vEpsilon) max(vEpsilon)], [0 0], ...
    'Color', [0 0 0]);

hold on;

plot(vBifurcLyap(:,1),vBifurcLyap(:,2:end),...
    'Marker','.', 'MarkerSize',2, ...
    'LineStyle','-');

hold on;


% axis([ min(vGamma) max(vGamma) -0.6 0.3 ]);


xlabel( '$\epsilon$', 'Interpreter', 'latex' );
ylabel( '$\lambda$', 'Interpreter', 'latex' );

% ----------------------------------------------------------------------- %

figure

% plot(vBifurcLyap(:,1),0.0,...
%     'Marker', '.', ...
%     'LineStyle','-', ...
%     'Color',[0 0 0]);

line([min(vEpsilon) max(vEpsilon)], [0 0], ...
    'Color', [0 0 0]);

hold on;

plot( vBifurcLyap(:,1), vBifurcLyap(:,2), ...
    'Marker','.', ...
    'LineStyle', '-' );

hold on;


% axis([ min(vGamma) max(vGamma) -0.6 0.3 ]);


xlabel( '$\epsilon$', 'Interpreter', 'latex' );
ylabel( '$\lambda_1$', 'Interpreter', 'latex' );

% ----------------------------------------------------------------------- %

% Print the figure
% sGraficoEPS = ['print -depsc2 fig_Duffing1989_bifurc_Lyap_vEpsilon_' num2str(vEpsilon(1)) 'a' num2str(vEpsilon(end)) '.eps'];
% evEpsilonl(sGraficoEPS);

% ======================================================================= %



%% ===================================================================== %%
%% EOF: End Of File
%% ===================================================================== %%