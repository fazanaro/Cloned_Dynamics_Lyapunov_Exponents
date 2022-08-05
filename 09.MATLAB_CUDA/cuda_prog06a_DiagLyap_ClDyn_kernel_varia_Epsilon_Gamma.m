% Ultima atualizacao: 27/03/2014



%% ===================================================================== %%
%% NOME DO ARQUIVO:
%   - cuda_prog06a_DiagLyap_ClDyn_kernel_varia_Epsilon_Gamma.m
%
%
%% DESCRICAO E OBJETIVOS:
%   - Esse script tem como objetivo principal executar o kernel em CUDA C
%   que implementa a integracao numerica RK4 para o modelo do oscilador
%   forcado de Duffing [1,2] empregando a abordagem das Dinamicas Clonadas
%   [3];  
%
%   - O kernel integra o modelo dinamico e os clones empregando o algoritmo
%   RK4 e retorna os vetores contendo os expoentes globais;
%
%   - Esse script calcula o espaco de parametros;
%
%
%% PROCEDIMENTOS:
%   - Os parametros sao passados na forma de um VETOR COLUNA;
%
%   - Vairacao dos parametros: vEpsilon e vGamma;
%
%
%% OBSERVACOES:
%   - [14/10/2013]:
%       - Script inicial;
%
%       - Existe a limitacao de que "nMaxItera" eh definida internamente no
%       kernel. O ideal eh que fosse passado como argumento de entrada do
%       kernel. Deve-se ter especial atencao de que modificacoes nas
%       variaveis relacionadas ah "nMaxItera" devem ser feitas tanto no
%       kernel (obviamente, devendo-se, entao, recompila-lo) quanto nesse
%       script;
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


% Variacao das condicoes iniciais
nIncControlParameter = 0.001;

% Parametros de controle
% vEpsilon = [0.15, 0.18, 0.22, 0.25];
% vGamma   = [0.1, 0.3];

vEpsilon = 0.3:nIncControlParameter:0.8;%0:nIncControlParameter:0.8;
vGamma   = 0.3:nIncControlParameter:0.8;%0:nIncControlParameter:0.8;

% vEpsilon = [ 0.15, 0.25, 0.54, 0.68 ];
% vGamma   = [ 0.30, 0.45, 0.76, 0.80 ];


% Demais constantes do modelo
omega = 1;


% Constroi as matrizes das possiveis combinacoes entre condicoes iniciais
[mGridEpsilon, mGridGamma] = meshgrid(vEpsilon, vGamma);

% Numero total de elementos
numElementos = numel( mGridEpsilon );

% Transforma em vetor COLUNA
vCols_mGridEpsilon = reshape( mGridEpsilon, [], 1 );
vCols_mGridGamma   = reshape( mGridGamma, [], 1 );

% Variavel auxiliar
nComprimento = length(vCols_mGridEpsilon);

% Define os vetores dos demais parametros a serem passados para a funcao
% 'arrayfun':
%   - Sao passados ah 'arrayfun' como vetores COLUNA;
vOmega = omega*ones( nComprimento, 1 );

%% ===================================================================== %%
%% DEFINICAO DAS CONDICOES INICIAIS DO SISTEMA ORIGINAL E DOS CLONES

% Define a variacao das condicoes iniciais para cada variavei de estado
vX0 = 1.0;
vY0 = 1.0;
vZ0 = 0.0;

% Define o vetor COLUNA referente ah terceira variavel de estado
vCols_X0 = vX0*ones( nComprimento, 1 );
vCols_Y0 = vY0*ones( nComprimento, 1 );
vCols_Z0 = vZ0*ones( nComprimento, 1 );

% Define o valor da perturbacao inicial aplicada aos CLONES
delta = 1e-4;

% Define as condicoes iniciais a serem passadas para o sistema ORIGINAL
y_init_orig = [ vCols_X0, vCols_Y0, vCols_Z0 ];

% Prepara as condicoes iniciais a serem perturbadas
%   - A funcao 'repmat' eh empregada para manter o padrao de ordem dos
%   clones (padrao definido no artigo do Wolf);
%   - Para obter esse padrao, cada linha deve ser disposta da seguinte
%   maneira:
%
%       - Considere, por exemplo, iLinha = 1;
%
%           | dado(iLinha,1), dado(iLinha,2) |
%           | dado(iLinha,3), dado(iLinha,4) |
%
mAux1 = [ repmat(vCols_X0,1,dim), repmat(vCols_Y0,1,dim), repmat(vCols_Z0,1,dim) ];

% Prepara os vetores das perturbacoes iniciais
%   - Vetor auxiliar
vPerturbClonesAux = reshape( delta*eye(dim), 1, [] );
%   - Repete esse vetor linha para todas as possiveis combinacoes de
%   condicoes iniciais
vPerturbClones = repmat( vPerturbClonesAux, nComprimento, 1 );

% Perturbacao das condicoes iniciais que serao passadas aos clones
y_init_clon = mAux1 + vPerturbClones;

% Construcao do vetor de condicoes iniciais (sistema ORIGINAL + CLONES) que
% eh passado ao integrador
%   - Cada LINHA representa as condicoes iniciais para as variaveis de
%   estado do sistema original e dos sistemas clonados
y_init = [ y_init_orig, y_init_clon ];

%% ===================================================================== %%
%% DEFINICAO DAS OPCOES DE INTEGRACAO E DEMAIS VARIAVEIS

% Define o instante de tempo inicial
t_init = 0;

% Define o instante de tempo final de simulacao
t_final = 10000;

% Define o intervalo de tempo para o qual o sistema (ORIGINAL + CLONES) eh
% integrado
t_gsr = 0.5;

% Define o passo de incremento do tempo
%   - Em outros scripts, esse parametro eh definido como 'passo';
h = 0.01;
%   - Expande  para todas as possiveis combinacoes das condicoes iniciais;
H = h*ones( nComprimento, 1 );

% Define o numero maximo de iteracoes
nMaxIteracoes = round( (t_final-t_init)/t_gsr );

% Define quantas iteracoes do algoritmo RK4 serao executadas a cada
% iteracao
nIntegra = t_gsr/h;

%% ===================================================================== %%
%% ALOCA MEMORIA PARA OS VETORES DE SAIDA

% Aloca memoria para os vetores que armaenam os valores dos expoentes
Lyap1 = gpuArray.zeros( nComprimento, 1 );
Lyap2 = gpuArray.zeros( nComprimento, 1 );
Lyap3 = gpuArray.zeros( nComprimento, 1 );

%% ===================================================================== %%
%% ALOCA OS DADOS NA MEMORIA DA GPU A SEREM PASSADOS COMO ARGUMENTO DE
%% ENTRADA DO KERNEL

y010 = gpuArray( y_init( 1:nComprimento, 1 ) );
y020 = gpuArray( y_init( 1:nComprimento, 2 ) );
y030 = gpuArray( y_init( 1:nComprimento, 3 ) );
y040 = gpuArray( y_init( 1:nComprimento, 4 ) );
y050 = gpuArray( y_init( 1:nComprimento, 5 ) );
y060 = gpuArray( y_init( 1:nComprimento, 6 ) );
y070 = gpuArray( y_init( 1:nComprimento, 7 ) );
y080 = gpuArray( y_init( 1:nComprimento, 8 ) );
y090 = gpuArray( y_init( 1:nComprimento, 9 ) );
y100 = gpuArray( y_init( 1:nComprimento, 10 ) );
y110 = gpuArray( y_init( 1:nComprimento, 11 ) );
y120 = gpuArray( y_init( 1:nComprimento, 12 ) );


%% ===================================================================== %%
%% COMPILA OS ARQUIVOS CUDA


eval(['!/usr/local/cuda-6.0/bin/nvcc -ptx cudaDuffing1989RK4_DiagLyap_v01.cu']);


%% ===================================================================== %%
%% CONFIGURACAO DO OBJETO RESPONSAVEL PELA EXECUCAO DO KERNEL EM CUDA

kernel = parallel.gpu.CUDAKernel( 'cudaDuffing1989RK4_DiagLyap_v01.ptx', 'cudaDuffing1989RK4_DiagLyap_v01.cu' );

% Make sure we have sufficient blocks to cover all of the locations
kernel.ThreadBlockSize = [ kernel.MaxThreadsPerBlock, 1, 1 ];
kernel.GridSize        = [ ceil(numElementos/kernel.MaxThreadsPerBlock), 1 ];

%% ===================================================================== %%

fprintf ( 1, '  Computation begun at %s.\n\n', datestr ( now ) );
tbase = tic();

%% ===================================================================== %%
%% EXECUCAO DO MODULO PRINCIPAL
%% CHAMADA DO KERNEL PARA A INTEGRACAO DO MODELO DINAMICO
%   - Retorna o estado final do sistema apos o intervalo 't_gsr';

tGPU_RK4 = tic;

[ Lyap1, Lyap2, Lyap3 ] = feval( kernel, ...
    Lyap1, Lyap2, Lyap3, ...
    y010, y020, y030, y040, y050, y060, y070, y080, y090, y100, y110, y120, ...
    H, vCols_mGridGamma, vCols_mGridEpsilon, vOmega, ...
    numElementos);

tGPU_RK4 = toc(tGPU_RK4);
% fprintf('[tGPU_RK4 = %.4f]', tGPU_RK4);

%% ===================================================================== %%
%% TRANSFERE OS DADOS DA MEMORIA

tGATHER = tic;


Lyap1 = gather( Lyap1 );
Lyap2 = gather( Lyap2 );
Lyap3 = gather( Lyap3 );


tGATHER = toc(tGATHER);
% fprintf('[tGATHER = %.4f]', tGATHER);

%% ===================================================================== %%
%% MANIPULACAO DOS DADOS

tMANIPULA = tic;


% Expoentes globais
Lyap = [ Lyap1, Lyap2, Lyap3 ];

% Vetor de dados
mStatesLocalLyapClDyn    = [ vCols_mGridEpsilon, vCols_mGridGamma, Lyap, max( Lyap, [], 2 ) ];
mNegStatesLocalLyapClDyn = [ vCols_mGridEpsilon, vCols_mGridGamma, Lyap, min( Lyap, [], 2 ) ];

% Vetor para construcao do espaco de estados
Mz    = reshape( mStatesLocalLyapClDyn(:,end),    size(mGridEpsilon,1), [] );
MzNeg = reshape( mNegStatesLocalLyapClDyn(:,end), size(mGridEpsilon,1), [] );



tMANIPULA = toc(tMANIPULA);
% fprintf('[tMANIPULA = %.4f]\n\n', tMANIPULA);

%% ===================================================================== %%

fprintf ( 1, '  Computation completed at %s.\n', datestr ( now ) );
cpuTime = toc( tbase );
fprintf('tempo_execucao_total = %.4f\n\n',cpuTime);

%% ===================================================================== %%
%% SALVA OS DADOS EM DISCO

fprintf ( 1, '  Saving the data ....');
sSave = ['save dados_Duffing1989_DiagLyap_vEpsilon_' num2str(min(vEpsilon)) '_a_' num2str(max(vEpsilon)) '_vGamma_' num2str(min(vGamma)) '_a_' num2str(max(vGamma)) '_vOmega_' num2str(min(vOmega)) '_a_' num2str(max(vOmega)) '_tf_' num2str(t_final) '_nIncParam_' num2str(nIncControlParameter) '_GPU_CUDA_Kernel_' stringnameGPU '.mat'];
eval(sSave);
fprintf ( 1, '  done!\n\n');

%% ===================================================================== %%
%% LIMPA A MEMORIA DA GPU

gpuDevice([]);

% ======================================================================= %



% %% ===================================================================== %%
% %% DESENHA A SAIDA
% 
% close all
% 
% figure(1);
% surf(mGridEpsilon, mGridGamma, Mz);
% colormap jet;
% shading interp;
% view([0 90]);
% 
% xlabel('epsilon');
% ylabel('gamma');
% 
% % title( ['[epsilon = ', num2str(epsilon), '][gamma = ', num2str(gamma), ']'] );
% % 
% % sGraficoEPS = ['print -depsc2 fig_Duffing1989_vX0_' num2str(vX0(1)) 'a' num2str(vX0(end)) '_vY0_' num2str(vY0(1)) 'a' num2str(vY0(end)) '_Z0_' num2str(vZ0(1)) '_epsilon_' num2str(epsilon) '_gamma_' num2str(gamma) '_Passo_' num2str(nIncControlParameter) '_GPU.eps'];
% % eval(sGraficoEPS);
% 
% 
% figure(2);
% surf(mGridEpsilon, mGridGamma, MzNeg);
% colormap jet;
% shading interp;
% view([0 90]);
% % ======================================================================= %





%% ===================================================================== %%
%% EOF: End Of File
%% ===================================================================== %%
