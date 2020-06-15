function out = gpow( in, x )
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
%   Implement a potentiation. It was used to improve the ANSI C
%   implementation.
%
% -----------------------------------------------
% Author:       Filipe Ieda Fazanaro
% Contact:      filipe.fazanaro AT gmail.com
% Version:      v07.01.2015.02
% Last update:  23.01.2015
% -----------------------------------------------

out = in^x;

end