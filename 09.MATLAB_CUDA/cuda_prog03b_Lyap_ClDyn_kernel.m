% Ultima atualizacao: 21/03/2014


%% ===================================================================== %%
%% NOME DO ARQUIVO:
%	- cuda_prog03b_Lyap_ClDyn_kernel.m
%
%
%% DESCRICAO E OBJETIVOS:
%   - Esse script tem como objetivo principal executar o kernel em CUDA C
%   que implementa a integracao numerica RK4 para o modelo do oscilador
%   forcado de Duffing [1] onde se calculam os expoentes de Lyapunov
%   empregando a abordagem via Dinamicas Clonadas [2];
%
%
%% PROCEDIMENTOS:
%   - O kernel integra o modelo dinamico e os clones empregando o algoritmo
%   RK4 e retorna a evolucao dos expoentes globais;
%
%   - Inicialmente, os testes foram realizados considerando-se somente uma
%   condicao inicial. A extensao consiste em passar duas ou mais condicoes
%   iniciais e fazer a analise em paralela;
%       - A implementacao visa extender para a identificacao das LCS e
%       construcao do espaco de parametros;
%
%
%% OBSERVACOES:
%   - [24/07/2013]:
%       - Script modificado em relacao ah versao 'prog03a';
%       - Nesse script, o kernel retorna os valores dos expoentes para o
%       instante 't_final' o qual eh definido no proprio kernel;
%       - A ideia eh extender o kernel de modo que 't_final' possa ser
%       passado como argumento evitando que o kernel tenha que ser
%       compilado sempre que esse parametro eh alterado;
%
%   - [17/03/2014]:
%       - Modificacao nos comentarios;
%       - O kernel foi alterado de modo a retornar somente a evolucao dos
%       expoentes globais;
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
%% ===================================================================== %%
%% MODELO DO SISTEMA: Duffing (1989)
%
%   dydt(1) = y(2);
%   dydt(2) = y(1) - y(1)^3 - epsilon*y(2) + gamma*cos(y(3));
%   dydt(3) = omega;
%
% ======================================================================= %

% clc
close all
clear all



%% ===================================================================== %%
%% CARREGA AS CARACTERISTICAS DA GPU


gpuCard = gpuDevice;


%% ===================================================================== %%
%% DEFINE A DIMENSAO DO SISTEMA DINAMICO A SER ANALISADO

dim = 3;

%% ===================================================================== %%
%% DEFINICAO DAS CONDICOES INICIAIS DO SISTEMA ORIGINAL E DOS CLONES

% Define a variacao das condicoes iniciais para cada variavei de estado
% vX0 = [-0.7,0.1,0.5,0.7];
% vY0 = [-0.5,0.5];
% vZ0 = 0;

% vX0 = [-0.7,0.1,0.5];
% vY0 = [-0.5,0.5];
% vZ0 = 0;

vX0 = 1.0;
vY0 = 1.0;
vZ0 = 0.0;

% vX0 = -1.5:0.05:1.5;
% vY0 = -1.0:0.05:1.0;
% vZ0 = 0;


% Constroi as matrizes das possiveis combinacoes das condicoes iniciais
[mGridX0, mGridY0] = meshgrid( vX0, vY0 );

% Numero total de elementos
numElementos = numel( mGridX0 );

% Transforma em vetor COLUNA
vCols_mGridX0 = reshape( mGridX0, [], 1 );
vCols_mGridY0 = reshape( mGridY0, [], 1 );

nComprimento = length( vCols_mGridX0 );   % auxiliar


% Define o vetor COLUNA referente ah terceira variavel de estado
vCols_mGridZ0 = vZ0*ones( nComprimento, 1 );


% Define as condicoes iniciais a serem passadas para o sistema ORIGINAL
y_init_orig = [ vCols_mGridX0, vCols_mGridY0, vCols_mGridZ0 ];


% Construcao do vetor de condicoes iniciais (sistema ORIGINAL + CLONES) que
% eh passado ao integrador
%   - Cada LINHA representa as condicoes iniciais para as variaveis de
%   estado do sistema original e dos sistemas clonados
y_init = [ y_init_orig ];

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

epsilon = 0.68;
gamma = 0.8;
omega = 1;

% Parametro de controle
%   - Definido como um vetor COLUNA;
vEpsilon = epsilon*ones( nComprimento, 1 );

% Define os vetores de parametros passados para o kernel:
%   - A ideia eh a mesma quando se emprega 'arrayfun';
%   - Sao passados ao kernel como vetores COLUNA;
vGamma = gamma*ones( nComprimento, 1 );
vOmega = omega*ones( nComprimento, 1 );

%% ===================================================================== %%
%% DEFINICAO DAS OPCOES DE INTEGRACAO E DEMAIS VARIAVEIS

% Define o instante de tempo inicial
t_init = 0;
tempo = 0;

% Define o instante de tempo final de simulacao
t_final = 5000;

% Define o intervalo de tempo para o qual o sistema (ORIGINAL + CLONES) eh
% integrado
t_gsr = 0.5;


% Define o passo de incremento do tempo
%   - Em outros scripts, esse parametro eh definido como 'passo';
h = 0.01;


% Define o numero maximo de iteracoes
nMaxIteracoes = round( (t_final-t_init)/t_gsr );

% Define quantas iteracoes do algoritmo RK4 serao executadas a cada
% iteracao
nIntegra = t_gsr/h;

%% ===================================================================== %%
%% COMPILA OS ARQUIVOS CUDA


eval(['!/usr/local/cuda-5.0/bin/nvcc -ptx cudaDuffing1989RK4_ClDyn_v03.cu']);


%% ===================================================================== %%
%% CONFIGURACAO DO OBJETO RESPONSAVEL PELA EXECUCAO DO KERNEL EM CUDA


kernel = parallel.gpu.CUDAKernel( 'cudaDuffing1989RK4_ClDyn_v03.ptx', 'cudaDuffing1989RK4_ClDyn_v03.cu' );


% Make sure we have sufficient blocks to cover all of the locations
kernel.ThreadBlockSize = [ kernel.MaxThreadsPerBlock, 1, 1 ];
kernel.GridSize        = [ ceil(numElementos/kernel.MaxThreadsPerBlock), 1 ];


%% ===================================================================== %%
%% ALOCA MEMORIA PARA OS VETORES DE SAIDA


Lyap1 = gpuArray.zeros( nComprimento, nMaxIteracoes+1 );
Lyap2 = gpuArray.zeros( nComprimento, nMaxIteracoes+1 );
Lyap3 = gpuArray.zeros( nComprimento, nMaxIteracoes+1 );


%% ===================================================================== %%
%% ALOCA OS DADOS NA MEMORIA DA GPU A SEREM PASSADOS COMO ARGUMENTO DE
%% ENTRADA DO KERNEL


y010 = gpuArray( y_init( 1:nComprimento, 1 ) );
y020 = gpuArray( y_init( 1:nComprimento, 2 ) );
y030 = gpuArray( y_init( 1:nComprimento, 3 ) );


%% ===================================================================== %%


fprintf ( 1, '  Computation begun at %s.\n', datestr ( now ) );
tbase = tic();


%% ===================================================================== %%
%% EXECUCAO DO MODULO PRINCIPAL

% for idxIteracao = 1:nMaxIteracoes
%
%     % ------------------------------------------------------------------- %
%     % CHAMADA DO KERNEL PARA A INTEGRACAO DO MODELO DINAMICO
%     %   - Retorna o estado final do sistema apos o intervalo 't_gsr';
%
%     %tGPU_RK4 = tic;

% [   Lyap1, Lyap2, Lyap3 ] = feval( kernel, ...
%     Lyap1, Lyap2, Lyap3, ...
%     y010, y020, y030, ...
%     h, ...
%     vGamma, vEpsilon, vOmega, ...
%     numElementos);


[   Lyap1, Lyap2, Lyap3 ] = feval( kernel, ...
    Lyap1, Lyap2, Lyap3, ...
    vGamma, vEpsilon, vOmega, ...
    numElementos);


%tGPU_RK4 = toc(tGPU_RK4);
%fprintf('[tGPU_RK4 = %.4f]', tGPU_RK4);
% ------------------------------------------------------------------- %

% end

%% ===================================================================== %%
%% TRANSFERE OS DADOS DA MEMORIA

% tGATHER = tic;


Lyap1 = gather( Lyap1 );
Lyap2 = gather( Lyap2 );
Lyap3 = gather( Lyap3 );


% tGATHER = toc(tGATHER);
% fprintf('[tGATHER = %.4f]\n', tGATHER);

%% ===================================================================== %%


fprintf ( 1, '  Computation completed at %s.\n', datestr ( now ) );
cpuTime = toc( tbase );
fprintf('tempo_execucao_total = %.4f\n',cpuTime);


%% ===================================================================== %%
%% LIMPA A GPU

gpuDevice([]);


%% ===================================================================== %%


Lyap = [Lyap1; Lyap2; Lyap3];


%% ===================================================================== %%
%% DESENHA A SAIDA

% figure

% plot(Tempo, Lyap);

% ======================================================================= %





%% ===================================================================== %%
%% EOF: End Of File
%% ===================================================================== %%