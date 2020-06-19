function dydt = ChuaAdim1985_ClDyn( t, y, alpha, beta, gamma, a, b )
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
%   - Defines the adimensional Chua's circuit model;
%   - Computes the Lyapunov exponents using the Cloned Dynamics approach.
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
% STANDARD
%   clones  =   | y(4)      y(5)    y(6)    |
%               | y(7)      y(8)    y(9)    |
%               | y(10)     y(11)   y(12)   |
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
%%                   NON-LINEARITIES - ORIGINAL (or FIDUCIAL)            %%
%% ===================================================================== %%

hx_y1 = b*y(1) + 0.5*(a-b)*(abs(y(1)+1)-abs(y(1)-1));


%% ===================================================================== %%
%%                        NON-LINEARITIES - CLONES                       %%
%% ===================================================================== %%

hx_y4 = b*y(4) + 0.5*(a-b)*(abs(y(4)+1)-abs(y(4)-1));
hx_y5 = b*y(5) + 0.5*(a-b)*(abs(y(5)+1)-abs(y(5)-1));
hx_y6 = b*y(6) + 0.5*(a-b)*(abs(y(6)+1)-abs(y(6)-1));


%% ===================================================================== %%
%%                 DYNAMIC MODEL - ORIGINAL (or FIDUCIAL)                %%
%% ===================================================================== %%

dydt(1,1) = alpha*(y(2)-y(1)-hx_y1);
dydt(2,1) = y(1) - y(2) + y(3);
dydt(3,1) = (-beta)*y(2) - gamma*y(3);


%% ===================================================================== %%
%%                       DYNAMIC MODEL - CLONES                          %%
%% ===================================================================== %%

dydt(4,1) = alpha*(y(7)-y(4)-hx_y4);
dydt(5,1) = alpha*(y(8)-y(5)-hx_y5);
dydt(6,1) = alpha*(y(9)-y(6)-hx_y6);

dydt(7,1) = y(4) - y(7) + y(10);
dydt(8,1) = y(5) - y(8) + y(11);
dydt(9,1) = y(6) - y(9) + y(12);

dydt(10,1) = (-beta)*y(7) - gamma*y(10);
dydt(11,1) = (-beta)*y(8) - gamma*y(11);
dydt(12,1) = (-beta)*y(9) - gamma*y(12);

% ======================================================================= %

end

%% ===================================================================== %%
%% EOF: End Of File
%% ===================================================================== %%