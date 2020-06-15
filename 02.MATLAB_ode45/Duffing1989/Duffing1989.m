function dydt = Duffing1989 ( t, y, gamma,epsilon,omega )
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
%% DYNAMIC MODEL

dydt(1,1) = y(2);
dydt(2,1) = y(1) - gpow(y(1),3) - epsilon*y(2) + gamma*cos( omega*y(3) );
dydt(3,1) = 1;

% ======================================================================= %

end