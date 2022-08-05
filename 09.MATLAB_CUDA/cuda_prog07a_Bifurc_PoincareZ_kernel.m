% Ultima atualizacao: 19/02/2014



%% ===================================================================== %%
%% NOME DO ARQUIVO:
%   - cuda_prog07a_Bifurc_PoincareZ_kernel.m
%
%
%% DESCRICAO E OBJETIVOS:
%   - Esse script tem como objetivo principal executar o kernel em CUDA C
%   que implementa a integracao numerica RK4 para o modelo do oscilador
%   forcado de Duffing [1,2] para a construcao do diagrama de bifurcacao
%   composto por secoes de Poincare justapostas;
%
%   - O kernel integra o modelo dinamico empregando o algoritmo RK4 e
%   retorna o estado do sistema ao final do intervalo de tempo considerado;
%
%
%% PROCEDIMENTOS:
%   - Os parametros sao passados na forma de um VETOR COLUNA;
%
%
%% OBSERVACOES:
%   - [13/02/2014]:
%       - Script inicial;
%
%       - EXISTE A LIMITACAO DE QUE EU NAO SEI COMO RETORNAR UM VETOR
%       MULDIMENSIONAL, I.E., EU ENTRO COM UM VETOR CONTENDO A VARIACAO DO
%       PARAMETRO DE CONTROLE E EU GOSTARIA DE RETORNAR UMA MATRIZ ONDE
%       CADA LINHA REPRESENTA A EVOLUCAO DO SISTEMA DINAMICO PARA AQUELE
%       DETERMINADO VALOR DO PARAMETRO DE CONTROLE;
%
%
%% REFERENCIAS
% [1] Parker and Chua (1989) Practical Numerical algorithms for Chaotic
% Systems. Springer-Verlag.
%
% [2] Guckenheimer, Holmes (2002) Nonlinear Oscillations, Dynamical
% systems, and Bifurcations of Vector Fields. Springer. 2nd edition.
%
% [3] Soriano, Fazanaro, Suyama, JRO, Attux and Madrid (2012), A method for
% Lyapunov spectrum estimation using cloned dynamics and its application
% to the discontinuously-excited FitzHugh-Nagumo model, Nonlinear Dynamics,
% vol 67, no 1, pp 413-424. DOI: 10.1007/s11071-011-9989-2.
%
% [4] http://www.mathworks.com/matlabcentral/newsreader/view_thread/316327
%% ===================================================================== %%
%% MODELO DO SISTEMA: Duffing (1989)
%
%   dydt(1) = y(2);
%   dydt(2) = y(1) - y(1)^3 - epsilon*y(2) + gamma*cos(omega*y(3));
%   dydt(3) = 1;
%
% ======================================================================= %

clc
close all
clear all



%% ===================================================================== %%
%% CARACTERISTICAS DA GPU

gpuCard = gpuDevice;

stringnameGPU = gpuCard.Name;

% remove pontuacoes
stringnameGPU(ismember(stringnameGPU,' ,.:;!')) = [];

%% ===================================================================== %%
%% DEFINE A DIMENSAO DO SISTEMA DINAMICO A SER ANALISADO
%   - Em se tratando da metodologia TanMap, o numero total de equacoes a
%   ser integrado eh, obrigatoriamente, igual ah "dim*(dim+1)";

dim = 3;

%% ===================================================================== %%
%% DEFINE AS CONSTANTES DO MODELO
% Constantes do modelo (apresentados em [1]):
%   - Periodic-1 solution:
%       - epsilon = 0.15
%       - gamma = 0.3
%       - omega = 1
%
%   - Periodic-3 solution:
%       - epsilon = 0.22
%       - gamma = 0.3
%       - omega = 1
%
%   - Chaotic solution:
%       - epsilon = 0.25
%       - gamma = 0.3
%       - omega = 1


% Define o passo de variacao do parametro de controle
nIncControlParameter = 0.02;

% Definicao do vetor de variacao do parametro de controle
% vEpsilon = 0.25;
vEpsilon = [0.18, 0.25];
% vEpsilon = 0:nIncControlParameter:0.8;
%   - Definido como um vetor COLUNA;
vEpsilon = vEpsilon';


% Numero total de elementos
numElementos = numel( vEpsilon );

nComprimento = length( vEpsilon );   % redundante; eh usada no script das LCS


% Define os vetores de parametros passados para o kernel:
%   - A ideia eh a mesma quando se emprega 'arrayfun';
%   - Sao passados ao kernel como vetores COLUNA;
gamma = 0.3;
omega = 1.0;

vGamma = gamma*ones( nComprimento, 1 );
vOmega = omega*ones( nComprimento, 1 );


%% ===================================================================== %%
%% DEFINICAO DAS CONDICOES INICIAIS DO SISTEMA ORIGINAL E DOS CLONES

% Define a variacao das condicoes iniciais para cada variavei de estado
vX0 = 1.0;
vY0 = 1.0;
vZ0 = 0.0;

% % Transforma em vetor COLUNA
% vCols_X0 = vX0*ones( nComprimento, 1 );
% vCols_Y0 = vY0*ones( nComprimento, 1 );
% vCols_Z0 = vZ0*ones( nComprimento, 1 );
% 
% 
% % Define as condicoes iniciais a serem passadas para o sistema ORIGINAL
% y_init = [ vCols_X0, vCols_Y0, vCols_Z0 ];

y_init = [ vX0, vY0, vZ0 ];


%% ===================================================================== %%
%% DEFINICAO DAS OPCOES DE INTEGRACAO E DEMAIS VARIAVEIS

% Define o instante de tempo inicial
t_init = 0;

% Define o instante de tempo final de simulacao
t_final = 30.0;

% Tempo de transitorio
t_trans = 10.0;


% Define o passo de incremento do tempo
%   - Em outros scripts, esse parametro eh definido como 'passo';
h = 0.01;
%   - Expande  para todas as possiveis combinacoes das condicoes iniciais;
H = h;%h*ones( nComprimento, 1 );


% % Define o indice do transitorio, ou seja, os pontos que deverao ser
% % desconsiderados
% %   - Esse valor eh estimado a partir da construcao da secao de Poincare
% index_trans = round((t_trans-t_init)/h);
% 
% 
% % Define quantas iteracoes serao consideradas ateh que o sistema dinamico
% % supere o transitorio
% nSuperaTrans = round( (t_trans-t_init)/h );
% 
% % Define quantas iteracoes do algoritmo RK4 serao executadas a cada
% % iteracao
% nIntegra = round( (t_final-t_init)/h );


%% ===================================================================== %%
%% ALOCACOES E DEFINICOES PARA A SECAO DE POINCARE

% DEFINE A SECAO DE POINCARE
%
%   - Eh interessante que a resposta temporal seja analisada para que esse
%   valor possa ser ajustado;

Threshold = 0.0;


% NUMERO MAXIMO DE PONTOS CONSIDERADOS EM CADA SECAO DE POINCARE

numMaxElementsPoincare = 5;


% ALOCA MEMORIA PARA OS VETORES QUE CONTEM OS PONTOS QUE FORMAM A SECAO DE
% POINCARE PARA CADA VARIACAO DO PARAMETRO DE CONTROLE
%
%   - Cada LINHA representa os pontos da secao de Poincare para um
%   determinado valor do parametro de controle

% vPoincareX = [];
% vPoincareY = [];
vPoincareX = 255*ones( length(vEpsilon)*numMaxElementsPoincare, 1 );
vPoincareY = 255*ones( length(vEpsilon)*numMaxElementsPoincare, 1 );


%% ===================================================================== %%
%% ALOCACAO DE MEMORIA NO DISPOSITIVO

% ALOCA MEMORIA PARA OS VETORES DE SAIDA
Y1 = gpuArray.zeros( length(vEpsilon)*numMaxElementsPoincare, 1 );
Y2 = gpuArray.zeros( length(vEpsilon)*numMaxElementsPoincare, 1 );


% ALOCA OS DADOS NA MEMORIA DA GPU A SEREM PASSADOS COMO ARGUMENTO DE
% ENTRADA DO KERNEL
y010 = gpuArray( y_init( 1, 1 ) );
y020 = gpuArray( y_init( 1, 2 ) );
y030 = gpuArray( y_init( 1, 3 ) );


%% ===================================================================== %%
%% CONFIGURACAO DO OBJETO RESPONSAVEL PELA EXECUCAO DO KERNEL EM CUDA


kernel = parallel.gpu.CUDAKernel( 'cudaDuffing1989RK4_PoincareZ.ptx', 'cudaDuffing1989RK4_PoincareZ.cu' );


% Make sure we have sufficient blocks to cover all of the locations
kernel.ThreadBlockSize = [ 2, 1, 1 ];%[ kernel.MaxThreadsPerBlock, 1, 1 ];
kernel.GridSize        = [ ceil(numElementos/kernel.MaxThreadsPerBlock), 1 ];


%% ===================================================================== %%


fprintf ( 1, '  Computation begun at %s.\n\n', datestr ( now ) );
tbase = tic();


%% ===================================================================== %%

tGPU_Transitorio_RK4 = tic;


[ Y1, Y2 ] = feval( kernel, ...
    Y1, Y2, ...
    y010, y020, y030, ...
    H, ...
    vGamma, vEpsilon, vOmega, ...
    Threshold, ...
    numElementos )


tGPU_Transitorio_RK4 = toc(tGPU_Transitorio_RK4);
fprintf('[tGPU_Transitorio_RK4 = %.4f]\n\n', tGPU_Transitorio_RK4);


Y1 = gather( Y1 );
Y2 = gather( Y2 );


for ii = 1:length(Y1)
    
    fprintf('Y1[%d] = %.6f\n', ii, Y1(ii));
    
end


%% ===================================================================== %%


fprintf ( 1, '\n  Computation completed at %s.\n', datestr ( now ) );
cpuTime = toc( tbase );
fprintf('tempo_execucao_total = %.4f\n\n',cpuTime);


%% ===================================================================== %%
%% LIMPA A MEMORIA DA GPU


gpuDevice([]);


% ======================================================================= %




%% ===================================================================== %%
%% EOF: End Of File
%% ===================================================================== %%
