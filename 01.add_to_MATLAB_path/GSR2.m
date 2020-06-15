function [vk, uk, Normk] = GSR2(Vector, dim)
% DESCRIPTION AND OBJECTIVES:
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
%   This function implemets the Gram-Schimidt Reorthonormalization used
%   during the procedure of calculation of the Lyapunov exponents.
%
% TEMPLATE:
%       [vk, uk, Normk] = GSR2(Vector, dim)
%
%       where:
%           - Vector = it must be a column vector or a square matrix;
%           - dim    = dimension of the embbeded space;
%           - vk     = ortonormal base;
%           - uk     = normalized ortonormal base;
%           - Normk  = norm of each colum of 'uk';
%
% EXAMPLE:
%   - working on it....
%
%
% REFERENCES:
% [1] http://www.scholarpedia.org/article/Attractor_reconstruction
%
% [2] Abarbanel (1996) Analysis of Observed Chaotic Data, 1996, Springer,
% 1st edition.
%
% [3] Nichols, Nichols (2001) Attractor reconstruction for non-linear
% systems: a methodological note. Mathematical Biosciences, May 2001, vol
% 171, issue 1, pp 21-32.
%
% [4] Soriano, Fazanaro, Suyama, JRO, Attux and Madrid (2011) A method for
% Lyapunov spectrum estimation using cloned dynamics and its application to
% the discontinuously-excited FitzHugh-Nagumo model, Nonlinear Dynamics, pp
% 1-12, DOI 10.1007/s11071-011-9989-2.
%
%
% LOG:
%   - v29.03.2011.01:
%       - Initial version;
%
%   - v06.03.2012.01:
%       - Modifications were made in the description;
%
%
% --------------------------------------------
% Author:       Diogo Coutinho Soriano
% Modified by:  Filipe Ieda Fazanaro
% Contact:      filipe.fazanaro AT gmail.com
% Version:      v06.03.2012.01
% --------------------------------------------

% ----------------------------------------------------------------------- %
% MEMORY ALOCATION
vk = zeros(dim);
uk = zeros(dim);
Normk = zeros(1,dim);
% ----------------------------------------------------------------------- %
for jj = 1:dim
    if jj == 1
        vk(:,jj) = Vector(:,jj);
        Normk(jj) = norm(vk(:,jj));
        uk(:,jj) = vk(:,jj)/Normk(jj);
    else
        % Aux
        pdi = 0;
        
        for kk = 1:(jj-1)
            pdi = pdi + ...
                (( uk(:,kk)'*Vector(:,jj) )/( uk(:,kk)'*uk(:,kk) ))*uk(:,kk);
        end
        
        vk(:,jj) = Vector(:,jj) - pdi;
        Normk(jj) = norm(vk(:,jj));
        uk(:,jj) = vk(:,jj)/Normk(jj);
    end
end
% ----------------------------------------------------------------------- %

% End function
end