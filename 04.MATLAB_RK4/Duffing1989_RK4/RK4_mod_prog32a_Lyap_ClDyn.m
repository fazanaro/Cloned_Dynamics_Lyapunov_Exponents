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

epsilon = 0.25;

gamma = 0.3;

omega = 1.0;

%% ===================================================================== %%
%% VECTOR PARAMETER

vParameters = [ epsilon, gamma, omega ];

%% ===================================================================== %%
%% MORE IMPORTANT DEFINITIONS

% Cloned dynamical systems perturbation - see [5]

delta = 1e-4;

%% ===================================================================== %%
%% INITIAL CONDITIONS OF THE DYNAMICAL SYSTEM

% Initial conditions
vX0 = 1;
vY0 = 1;
vZ0 = 0;

y_init_orig  = [ vX0, vY0, vZ0 ];


% Cloned dynamical systems initial conditions
y_init_clon = (ones(dim,1)*y_init_orig)' + delta*eye(dim);

% Fiducial and clones initial conditions
y_init = [ y_init_orig, reshape(y_init_clon',1,[])];


%% ===================================================================== %%
%% INTEGRATION DEFINITIONS AND INITIAL CONDITIONS

% Initial time
t_init = 0;


% Aux
t_aux = 0;


% Final time
t_final = 1000;


% Gram-Schimidt reorthonormalization time interval
%   - See [5] for further details related to how to change this value.
t_gsr = 0.50;


% Time step
h = 0.01;


% Maximum iteration number
nMaxItera = round( (t_final-t_init)/t_gsr );


% Cloned Dynamics approach aux
deltax = zeros( dim );


% Gram-Schimidt Reorthonormalization
vk = zeros( dim );
uk = zeros( dim );


% Memory alocation
LyapSoma  = zeros( dim, nMaxItera );
LyapLocal = zeros( dim, nMaxItera );
Lyap      = zeros( dim, nMaxItera );

Time      = zeros( 1, nMaxItera );


% Aux
Yfinal = [];

%% ===================================================================== %%

fprintf ( 1, '\n  Computation begun at %s.\n', datestr ( now ) );

tbase = tic();

%% ===================================================================== %%
%% DYNAMICAL SYSTEM INTEGRATION

for ii = 1:nMaxItera
    
    % ------------------------------------------------------------------- %
    % DYNAMICAL SYSTEM INTEGRATION
    % ------------------------------------------------------------------- %
    
    [T, Y] = odeRK4v3( @odefun_Duffing1989_ClDyn, (t_aux:h:(t_aux+t_gsr)), h, y_init, vParameters, dim_total );
    
    % ------------------------------------------------------------------- %
    % Store the (complete) system evolution
    
    Yfinal = [Yfinal; Y];
    
%     for idxRows = 1:size( Yfinal,1 )
%         
%         for idxCols = 1:size( Yfinal, 2 )
%             
%             fprintf( '[%.6f]', Yfinal(idxRows, idxCols) );
%             
%         end
%         
%         fprintf( '\n' );
%         
%     end
    
    % ------------------------------------------------------------------- %
    
    t_aux = t_aux + t_gsr;
    
    % ------------------------------------------------------------------- %
    % You only need the conditions at the end of the 't_gsr' interval
    
    y_orig = Y( end, 1:dim )';
    y_clon = Y( end, (dim+1):(dim*(dim+1)));
    
    y_clon = reshape( y_clon, dim, [] )';
    
    % ------------------------------------------------------------------- %
    % DIFERRENCE STATE VECTORS 
    % ------------------------------------------------------------------- %
    
    deltax = y_orig*ones( 1, dim ) - y_clon;
    
    
%     % SCREEN
%     
%     fprintf( '\n' );
%     
%     for idxRows = 1:size( deltax,1 )
%         
%         for idxCols = 1:size( deltax, 2 )
%             
%             fprintf( '[%.6f]', deltax(idxRows, idxCols) );
%             
%         end
%         
%         fprintf( '\n' );
%         
%     end
    
    % ------------------------------------------------------------------- %
    % GRAM-SCHIMIDT REORTHONORMALIZATION
    % ------------------------------------------------------------------- %
    
    for jj = 1:dim
        if jj == 1
            vk(:,jj) = deltax(:,jj);
            uk(:,jj) = vk(:,jj)/norm(vk(:,jj));
        else
            % Aux
            pdi = 0;
            
            for kk = 1:(jj-1)
                pdi = pdi + ...
                    (( uk(:,kk)'*deltax(:,jj) )/( uk(:,kk)'*uk(:,kk) ))*uk(:,kk);
            end
            
            vk(:,jj) = deltax(:,jj) - pdi;
            uk(:,jj) = vk(:,jj)/norm(vk(:,jj));
        end
    end
    
    
%     % SCREEN
%     
%     fprintf( '\n' );
%     
%     for idxRows = 1:size( vk,1 )
%         
%         for idxCols = 1:size( vk, 2 )
%             
%             fprintf( '[%.12f]', vk(idxRows, idxCols) );
%             
%         end
%         
%         fprintf( '\n' );
%         
%     end
%     
%     
%     
%     % SCREEN
%     
%     fprintf( '\n' );
%     
%     for idxRows = 1:size( uk,1 )
%         
%         for idxCols = 1:size( uk, 2 )
%             
%             fprintf( '[%.12f]', uk(idxRows, idxCols) );
%             
%         end
%         
%         fprintf( '\n' );
%         
%     end
    
    % ------------------------------------------------------------------- %
    % LYAPUNOV EXPONENTS CALCULATION
    % ------------------------------------------------------------------- %
    
    for jj = 1:dim
        if norm(vk(:,jj)) ~= 0
            % Local exponent
            LyapLocal(jj,ii+1) = (1/t_gsr)*log(norm(vk(:,jj))/delta);
            
            % Local exponents summation
            LyapSoma(jj,ii+1) = LyapSoma(jj,ii) + log(norm(vk(:,jj))/delta);
            
            % Local exponents summation
            Lyap(jj,ii+1) = (1/(t_aux-t_init))*LyapSoma(jj,ii+1);
        end
    end
    
    
%     % SCREEN
%     
%     fprintf( '\n' );
%     
%     for idxRows = 1:size( LyapSoma,1 )
%         
%         for idxCols = 1:size( LyapSoma, 2 )
%             
%             fprintf( '[%.12f]', LyapSoma(idxRows, idxCols) );
%             
%         end
%         
%         fprintf( '\n' );
%         
%     end
%     
%     
%     % SCREEN
%     
%     fprintf( '\n' );
%     
%     for idxRows = 1:size( Lyap,1 )
%         
%         for idxCols = 1:size( Lyap, 2 )
%             
%             fprintf( '[%.12f]', Lyap(idxRows, idxCols) );
%             
%         end
%         
%         fprintf( '\n' );
%         
%     end
    
    % ------------------------------------------------------------------- %
    % Prepare for the next iteration
    
    Time(ii+1) = t_aux;
    
    % Fiducial and cloned dynamical systems initial conditions for the next
    % iteration
    
    y_init_orig = y_orig';
    y_init_clon = (ones(dim,1)*y_init_orig)' + delta*uk;
    
    y_init = [ y_init_orig, reshape(y_init_clon',1,[]) ];
    
    % ------------------------------------------------------------------- %
    
end

%% ===================================================================== %%

fprintf ( 1, '\n  Computation completed at %s.\n', datestr ( now ) );

cpuTime = toc( tbase );

fprintf( 'Total execution time [s] = %.4f\n\n', cpuTime );

%% ===================================================================== %%
%% GLOBAL LYAPUNOV EXPONENTS

% Only the final Lyapunov exponents

for ii = 1:dim
    
    fprintf( 'Lyap[%d] = %.6f\n', ii, Lyap(ii,end) );
    
end

fprintf( '\n' );


% % The complete evolution
% 
% for ii = 1:dim
%     
%     for jj = 1:size( Lyap, 2 )
%     
%         fprintf( '[%.6f]', Lyap(ii,jj) );
%     
%     end
%     
%     fprintf( '\n' );
%     
% end
% 
% fprintf( '\n' );

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
grid(hax1, 'on' );


plot(Time, LyapLocal);

% Adjusting the x- and y-labels positioning
set( hax1, 'Units', 'Normalized' );

pos = get( hax1, 'Position' );

offset = 0.01;

set( hax1, ...
    'XGrid'       , 'off'                        , ...
    'YGrid'       , 'off'                        , ...
    'Position'    , pos + [0, offset, 0, -offset] );

hx = xlabel( 'Time [u.a.]', 'Interpreter', 'latex' );
hy = ylabel( '$\lambda$', 'Interpreter', 'latex' );

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
grid(hax2, 'on' );


line([min(Time) max(Time)],[0 0], ...
    'Color', [0 0 0]);

plot(Time, Lyap);



% axis( [min(Time) max(Time) -10 5] );


% Adjusting the x- and y-labels positioning
set( hax2, 'Units', 'Normalized' );

pos = get( hax2, 'Position' );

offset = 0.01;

set( hax2, ...
    'XGrid'       , 'off'                        , ...
    'YGrid'       , 'off'                        , ...
    'Position'    , pos + [0, offset, 0, -offset] );

hx = xlabel( 'Time [u.a.]', 'Interpreter', 'latex' );
hy = ylabel( '$\lambda$', 'Interpreter', 'latex' );

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

hFig3 = figure( 'visible', 'on' );
hax3 = axes( 'Parent', hFig3, 'FontSize', FontSize );

box( hax3, 'on' );
hold(hax3, 'all' );
grid(hax3, 'off' );


plot( Yfinal(:,1), Yfinal(:,2) );


% axis( [min(Time)/t_step max(Time)/t_step -20 120] );


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

% sGraficoEPS = ['print -depsc2 -r600 test_name.eps'];
% eval(sGraficoEPS);
%
% sGraficoJPG = ['print -djpeg -r600 test_name.jpg'];
% eval(sGraficoJPG);
%
% sGraficoPNG = ['print -dpng -r600 test_name.png'];
% eval(sGraficoPNG);

% ----------------------------------------------------------------------- %

% ======================================================================= %




%% ===================================================================== %%
%% EOF: End Of File
%% ===================================================================== %%