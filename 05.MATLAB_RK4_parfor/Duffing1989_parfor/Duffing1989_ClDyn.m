function dydt = Duffing1989_ClDyn( t, y, gamma, epsilon, omega )
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
% Initial version: September 15, 2013
% Last update: June 15, 2020
%
%% ===================================================================== %%
%%                 DYNAMIC MODEL - ORIGINAL (or FIDUCIAL)                %%
%% ===================================================================== %%

dydt(1,1) = y(2);
dydt(2,1) = y(1) - gpow(y(1),3) - epsilon*y(2) + gamma*cos( omega*y(3) );
dydt(3,1) = 1.0;


%% ===================================================================== %%
%%                       DYNAMIC MODEL - CLONES                          %%
%% ===================================================================== %%

dydt(4,1) = y(7);
dydt(5,1) = y(8);
dydt(6,1) = y(9);

dydt(7,1) = y(4) - gpow(y(4),3) - epsilon*y(7) + gamma*cos( omega*y(10) );
dydt(8,1) = y(5) - gpow(y(5),3) - epsilon*y(8) + gamma*cos( omega*y(11) );
dydt(9,1) = y(6) - gpow(y(6),3) - epsilon*y(9) + gamma*cos( omega*y(12) );

dydt(10,1) = 1.0;
dydt(11,1) = 1.0;
dydt(12,1) = 1.0;

% ======================================================================= %

end