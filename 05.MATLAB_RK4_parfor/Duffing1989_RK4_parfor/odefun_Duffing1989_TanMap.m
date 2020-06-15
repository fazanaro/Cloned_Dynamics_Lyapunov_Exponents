function dydt = odefun_Duffing1989_TanMap ( t0, Y0, vParameters )
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

% DYNAMIC MODEL - VARIATIONAL

y40 = Y0(4);
y50 = Y0(5);
y60 = Y0(6);

y70 = Y0(7);
y80 = Y0(8);
y90 = Y0(9);

y100 = Y0(10);
y110 = Y0(11);
y120 = Y0(12);


%% ===================================================================== %%
%%                 DYNAMIC MODEL - ORIGINAL (or FIDUCIAL)                %%
%% ===================================================================== %%

dydt(1,1) = y20;
dydt(2,1) = y10 - gpow(y10,3) - epsilon*y20 + gamma*cos( omega*y30 );
dydt(3,1) = 1;


%% ===================================================================== %%
%%                      DYNAMIC MODEL - TANGENT MAP                      %%
%% ===================================================================== %%

dydt(4,1) = y70;
dydt(5,1) = y80;
dydt(6,1) = y90;

dydt(7,1) = (1.0 - 3.0*gpow(y10,2) )*y40 - epsilon*y70 - gamma*sin( omega*y30 )*y100;
dydt(8,1) = (1.0 - 3.0*gpow(y10,2) )*y50 - epsilon*y80 - gamma*sin( omega*y30 )*y110;
dydt(9,1) = (1.0 - 3.0*gpow(y10,2) )*y60 - epsilon*y90 - gamma*sin( omega*y30 )*y120;

dydt(10,1) = 0.0;
dydt(11,1) = 0.0;
dydt(12,1) = 0.0;

% ======================================================================= %

end