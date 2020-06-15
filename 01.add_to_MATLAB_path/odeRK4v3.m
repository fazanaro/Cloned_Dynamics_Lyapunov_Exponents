function [T,Y] = odeRK4v3(odefun, tspan, tstep, y_init, vParameters, dim)
% DESCRIPTION
%   The code is (C) 2014-2016 Filipe I. Fazanaro, Diogo C. Soriano, Ricardo
%   Suyama, Marconi K. Madrid, Jose Raimundo de Oliveira, Ignacio B. Munoz,
%   Romis Attux and is released under the GPL v3 license. 
%
%   Sample code for the work intitled "Numerical Characterization of
%   Nonlinear Dynamical Systems Using Parallel Computing: The Role of GPUs
%   Approach", which was accepted to be published in Communications of
%   Nonlinear Science and Numerical Computation.  
%   DOI: 10.1016/j.cnsns.2015.12.021
%
%   Implementation of the fourth order Runge-Kutta integration method.
%
%
% SYNTAX
%
%   [T,Y] = odeRK4v3(odefun, tspan, tstep, y_init);
%
%
% ARGUMENTS
%   - 'odefun'  = ODE function
%   - tspan     = time vector
%   - tstep     = Runge Kutta fourth order step
%   - y_init    = initial conditions
%   - dim       = dynamical system dimension (driver + response + error)
%
%
% LOGs
%   - [07/01/2015]:
%       - Initial version;
%
%
% -----------------------------------------------
% Author:       Filipe Ieda Fazanaro
% Contact:      filipe.fazanaro AT gmail.com
% Version:      v07.01.2015.02
% Last update:  23.01.2015
% -----------------------------------------------


%% ===================================================================== %%
%% INITIALIZATION

h = tstep;

numElements = length(tspan);


T = 255*ones(numElements, 1);
T(1) = tspan(1);


Y = 255*ones(numElements, dim);
Y(1,:) = y_init;


k1 = 255*ones(1,dim);
k2 = 255*ones(1,dim);
k3 = 255*ones(1,dim);
k4 = 255*ones(1,dim);

%% ===================================================================== %%
for ii = 1:(numElements-1)
    
    yi = Y(ii,:);
    ti = tspan(ii);
    
    k1(1,:) = h*odefun( ti, yi, vParameters );
    k2(1,:) = h*odefun( ti, yi+k1(1,:)/2, vParameters );
    k3(1,:) = h*odefun( ti, yi+k2(1,:)/2, vParameters );
    k4(1,:) = h*odefun( ti, yi+k3(1,:), vParameters );
    
    T(ii+1,1) = ti + h;
    Y(ii+1,:) = yi + (1/6)*(k1(1,:)+2*k2(1,:)+2*k3(1,:)+k4(1,:));
    
end
% ======================================================================= %


%% ===================================================================== %%
%% EOF: End Of File
%% ===================================================================== %%
end