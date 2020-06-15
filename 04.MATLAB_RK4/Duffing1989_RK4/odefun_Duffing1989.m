function dydt = odefun_Duffing1989 ( t0, Y0, vParameters )
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
%%                              PARAMETERS                               %%
%% ===================================================================== %%

epsilon  = vParameters(1);
gamma    = vParameters(2);
omega    = vParameters(3);


%% ===================================================================== %%
%%                          INITIAL CONDITIONS                           %%
%% ===================================================================== %%

% DYNAMIC MODEL - FIDUCIAL

y10 = Y0(1);
y20 = Y0(2);
y30 = Y0(3);


%% ===================================================================== %%
%%                            DYNAMIC MODEL                              %%
%% ===================================================================== %%

dydt(1,1) = y20;
dydt(2,1) = y10 - gpow(y10,3) - epsilon*y20 + gamma*cos( omega*y30 );
dydt(3,1) = 1.0;

% ======================================================================= %

end