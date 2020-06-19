function dydt = ChuaAdim1985 ( t, y, alpha, beta, gamma, a, b )
%% ===================================================================== %%
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
% -------------------------------------------------------------------------
% DESCRIPTION
%   Defines the adimensional Chua's circuit model;
%
%
% MODEL
%   dxdt = alpha*(y(2) - y(1) - h(y(1)))
%   dydt = y(1) - y(2) + y(3)
%   dzdt = (-beta)*y(2) - gamma*y(3)
%
%
% where:
%   - alpha = 9;
%   - beta  = 14+2/7 = 100/7 = 14.2857;
%   - gamma = 0;
%   - a = -8/7
%   - b = -5/7
%   - (x0, y0, z0) = (0.15264, -0.02281, 0.38127);
%
%
% ------------------------------------------------------
% Author:           Filipe Ieda Fazanaro
% Contact:          filipe.fazanaro AT ufabc.edu.br
% Initial version:  v05.02.2013.01
% Last update:      v18.06.2020.01
% ------------------------------------------------------
%
% SEE ALSO:
%   some other functions....
%
%% ===================================================================== %%
%% DYNAMIC MODEL

% Chua's diode equation
hx = b*y(1) + 0.5*(a-b)*( abs(y(1)+1)-abs(y(1)-1) );

dydt(1,1) = alpha*( y(2) - y(1) - hx );
dydt(2,1) = y(1) - y(2) + y(3);
dydt(3,1) = (-beta)*y(2) - gamma*y(3);

% ======================================================================= %

end

%% ===================================================================== %%
%% EOF: End Of File
%% ===================================================================== %%