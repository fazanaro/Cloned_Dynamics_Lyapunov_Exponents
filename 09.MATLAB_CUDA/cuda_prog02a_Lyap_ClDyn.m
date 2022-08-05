% Ultima atualizacao: 10/06/2013


%% ===================================================================== %%
%% NOME DO ARQUIVO:
%   - cuda_prog02a_Lyap_ClDyn.m
%
%
%% DESCRICAO E OBJETIVOS:
%  - Esse script tem como objetivo principal executar o kernel em CUDA C
%  que implementa a integracao numerica RK4 para o modelo do oscilador
%  forcado de Duffing [1] para a resolucao da abordagem via Dinamicas
%  Clonadas [2];
%
%
%% PROCEDIMENTOS:
%   - O kernel integra o modelo dinamico e os clones empregando o algoritmo
%   RK4 e retorna o estado final do sistema apos 'nIntegra' iteracoes;
%
%   - Esse estado final eh, entao, utilizado para completar o
%   desenvolvimento da abordagem ClDyn;
%
%   - Inicialmente, os testes serao realizados considerando-se somente uma
%   condicao inicial. A extensao consiste em passar duas ou mais condicoes
%   iniciais e fazer a analise em paralela;
%       - A implementacao visa extender para a identificacao das LCS e
%       construcao do espaco de parametros;
%
%
%% OBSERVACOES:
%   - [27/02/2013]:
%       - Script inicial;
%
%
%% REFERENCIAS
% [1] Parker and Chua (1989) Practical Numerical algorithms for Chaotic
% Systems. Springer-Verlag.
%
% [2] Soriano, Fazanaro, Suyama, JRO, Attux and Madrid (2012), A method for
% Lyapunov spectrum estimation using cloned dynamics and its application
% to the discontinuously-excited FitzHugh-Nagumo model, Nonlinear Dynamics,
% vol 67, no 1, pp 413-424. DOI: 10.1007/s11071-011-9989-2.
%% ===================================================================== %%
%% MODELO DO SISTEMA: Duffing (1989)
%
%   dydt(1) = y(2);
%   dydt(2) = y(1) - y(1)^3 - epsilon*y(2) + gamma*cos(y(3));
%   dydt(3) = omega;
% ======================================================================= %
clc
close all
clear all



%% Reinicia a GPU
gpuCard = gpuDevice;
reset(gpuCard);



%% ===================================================================== %%
%% CARREGA DADOS IMPORTANTES PARA DEPURACAO
%   - Geracao dos dados eh feita utilizando o seguinte script:
%
%                   RK4_prog32a_Lyap_ClDyn.m
%
% % % load('saida_integrador_Y.mat');
%% ===================================================================== %%
%% DEFINE A DIMENSAO DO SISTEMA DINAMICO A SER ANALISADO
%   - Em se tratando da metodologia TanMap, o numero total de equacoes a
%   ser integrado eh, obrigatoriamente, igual ah "dim*(dim+1)";
dim = 3;
%% ===================================================================== %%
%% DEFINICAO DAS CONDICOES INICIAIS DO SISTEMA ORIGINAL E DOS CLONES

% Define a variacao das condicoes iniciais para cada variavei de estado
% vX0 = [-0.7,0.1,0.5,0.7];
% vY0 = [-0.5,0.5];
% vZ0 = 0;

vX0 = 1;
vY0 = 1;
vZ0 = 0;

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

% Define o valor da perturbacao inicial aplicada aos CLONES
delta = 1e-4;

% Define as condicoes iniciais a serem passadas para o sistema ORIGINAL
y_init_orig = [ vCols_mGridX0, vCols_mGridY0, vCols_mGridZ0 ];

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
mAux1 = [ repmat(vCols_mGridX0,1,dim), repmat(vCols_mGridY0,1,dim), repmat(vCols_mGridZ0,1,dim) ];

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
epsilon = 0.25;
gamma = 0.3;
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
%   - Expande o instante inicial para todas as possiveis combinacoes das
%   condicoes iniciais;
T0 = t_init*ones( nComprimento, 1 );

% Define o instante de tempo final de simulacao
t_final = 3000;

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

% Aloca memoria para o vetor que armazena os instantes de tempo
Tempo = zeros( 1, nMaxIteracoes );

% Aloca memoria para variaveis empregadas na abordagem ClDyn
%   - Lembrar que sao integradas 'dim*(dim+1)' equacoes dinamicas;
Y = 255*ones( nComprimento, dim*(dim+1) );

y_orig = 255*ones( nComprimento, dim );
y_clon = 255*ones( nComprimento, dim*dim );

% Aloca memoria para o vetor utilizado para computar a diferenca entre a
% trajetoria fiducial e os clones
deltax = 255*ones( dim, dim*dim );
%% ===================================================================== %%
%% ALOCACAO DE MEMORIA PARA OS VETORES UTILIZADOS NO CALCULO DOS EXPOENTES

% Aloca memoria para os vetores usados no procedimento de
% reortonormalizacao de Gram-Schimidt
vk = zeros( nComprimento, dim*dim );
uk = zeros( nComprimento, dim*dim );
Normk = zeros( nComprimento, dim );

% Aloca memoria para os vetores que armaenam os valores dos expoentes
LyapLocal1 = zeros( nComprimento, nMaxIteracoes );
LyapSoma1  = zeros( nComprimento, nMaxIteracoes );
Lyap1      = zeros( nComprimento, nMaxIteracoes );

LyapLocal2 = zeros( nComprimento, nMaxIteracoes );
LyapSoma2  = zeros( nComprimento, nMaxIteracoes );
Lyap2      = zeros( nComprimento, nMaxIteracoes );

LyapLocal3 = zeros( nComprimento, nMaxIteracoes );
LyapSoma3  = zeros( nComprimento, nMaxIteracoes );
Lyap3      = zeros( nComprimento, nMaxIteracoes );
%% ===================================================================== %%
%% CONFIGURACAO DO OBJETO RESPONSAVEL PELA EXECUCAO DO KERNEL EM CUDA

kernel = parallel.gpu.CUDAKernel( 'cudaDuffing1989RK4_ClDyn_v01.ptx', 'cudaDuffing1989RK4_ClDyn_v01.cu' );

% Make sure we have sufficient blocks to cover all of the locations
kernel.ThreadBlockSize = [ kernel.MaxThreadsPerBlock, 1, 1 ];
kernel.GridSize        = [ ceil(numElementos/kernel.MaxThreadsPerBlock), 1 ];
%% ===================================================================== %%
%% ALOCA MEMORIA PARA OS VETORES DE SAIDA
Y1 = gpuArray.zeros( nComprimento, 1 );
Y2 = gpuArray.zeros( nComprimento, 1 );
Y3 = gpuArray.zeros( nComprimento, 1 );
Y4 = gpuArray.zeros( nComprimento, 1 );
Y5 = gpuArray.zeros( nComprimento, 1 );
Y6 = gpuArray.zeros( nComprimento, 1 );
Y7 = gpuArray.zeros( nComprimento, 1 );
Y8 = gpuArray.zeros( nComprimento, 1 );
Y9 = gpuArray.zeros( nComprimento, 1 );
Y10 = gpuArray.zeros( nComprimento, 1 );
Y11 = gpuArray.zeros( nComprimento, 1 );
Y12 = gpuArray.zeros( nComprimento, 1 );
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
%% EXECUCAO DO MODULO PRINCIPAL

for idxIteracao = 1:nMaxIteracoes
    
    % ------------------------------------------------------------------- %
    % CHAMADA DO KERNEL PARA A INTEGRACAO DO MODELO DINAMICO
    %   - Retorna o estado final do sistema apos o intervalo 't_gsr';
% % %     tGPU_RK4 = tic;
    
    [Y1, Y2, Y3, Y4, Y5, Y6, Y7, Y8, Y9, Y10, Y11, Y12] = feval( kernel, ...
        Y1, Y2, Y3, Y4, Y5, Y6, Y7, Y8, Y9, Y10, Y11, Y12, ...
        y010, y020, y030, y040, y050, y060, y070, y080, y090, y100, y110, y120, ...
        H, vGamma, vEpsilon, vOmega, ...
        nIntegra, numElementos);
    
% % %     tGPU_RK4 = toc(tGPU_RK4);
% % %     fprintf('[tGPU_RK4 = %.4f]', tGPU_RK4);
    % ------------------------------------------------------------------- %
    % Atualiza o instante de tempo para a proxima iteracao
    tempo = tempo+t_gsr;
    % ------------------------------------------------------------------- %
    % Salva os dados pertinentes ao final do intervalo de tempo considerado
    % (a priori, igual ah "t_gsr"):
    Y = [ Y1, Y2, Y3, Y4, Y5, Y6, Y7, Y8, Y9, Y10, Y11, Y12 ];
    % Faz a transformacao para a aplicacao da abordagem ClDyn;
    %   - Salva os valores obtidos ao final do intervalo de integracao;
    %   - Lembrar que cada LINHA representa o resultado obtido ao final do
    %   intervalo de tempo igual ah 't_gsr':
    y_orig = Y(:,1:dim);
    y_clon = Y(:,(dim+1):(dim*(dim+1)));
    % ------------------------------------------------------------------- %
    % Calcula os vetores de deslocamento que representam as distancias
    % entre cada um dos clones em relacao ao sistema original (fiducial)
% % %     tDeltax = tic;
    
    mAux1 = [repmat(y_orig(:,1), 1, dim), repmat(y_orig(:,2), 1, dim), repmat(y_orig(:,3), 1, dim)];
    mAux2 = mAux1 - y_clon;
    %   - Faz a transposicao em virtude de que a funcao 'reshape' faz uma
    %   busca ao longo das COLUNAS. Por isso deve-se inverter a matriz
    %   'deltax' para que fique condizente;
    deltax = reshape(mAux2', dim, []);
    
% % %     tDeltax = toc(tDeltax);
% % %     fprintf('[tDeltax = %.4f]', tDeltax);
    % ------------------------------------------------------------------- %
    % TRANSFERE OS DADOS PERTINENTES DA MEMORIA DA GPU PARA O WORKSPACE
% % %     tGATHER = tic;
    
    deltax = gather(deltax);
    
% % %     tGATHER = toc(tGATHER);
% % %     fprintf('[tGATHER = %.4f]', tGATHER);
    % ------------------------------------------------------------------- %
    % CALCULO DOS EXPOENTES DE LYAPUNOV
    
% % %     tGRS2 = tic;
    for jj = 1:nComprimento
        % --------------------------------------------------------------- %
        % Procedimento de Reortonormalizacao de Gram-Schimidt
        %   - Os vetores de reortonormalizacao sao armazenados em blocos de
        %   ordem "dim x dim" e cada um desses blocos representa os valores
        %   que se obtem para cada uma das condicoes iniciais adotadas;
        [mVk, mUk, mNormk] = GSR2(deltax( (1:dim), ((jj-1)*dim+1):(jj*dim) )', dim);
        
        vk(jj,:) = reshape(mVk',1,[]);
        uk(jj,:) = reshape(mUk',1,[]);
        Normk(jj,:) = mNormk;
        % --------------------------------------------------------------- %
    end
% % %     tGRS2 = toc(tGRS2);
% % %     fprintf('[tGRS2 = %.4f]\n', tGRS2);
    
    % ------------------------------------------------------------------- %
    [nRowsAux, nColsAux] = size(Normk);
    
    LyapLocal1(:,idxIteracao+1) = (1/t_gsr)*log( Normk(:,1)/delta );
    LyapLocal2(:,idxIteracao+1) = (1/t_gsr)*log( Normk(:,2)/delta );
    LyapLocal3(:,idxIteracao+1) = (1/t_gsr)*log( Normk(:,3)/delta );
    
    LyapSoma1(:,idxIteracao+1) = LyapSoma1(:,idxIteracao) + log( Normk(:,1)/delta );
    LyapSoma2(:,idxIteracao+1) = LyapSoma2(:,idxIteracao) + log( Normk(:,2)/delta );
    LyapSoma3(:,idxIteracao+1) = LyapSoma3(:,idxIteracao) + log( Normk(:,3)/delta );
    
    Lyap1(:,idxIteracao+1) = (1/(tempo-t_init))*LyapSoma1(:,idxIteracao+1);
    Lyap2(:,idxIteracao+1) = (1/(tempo-t_init))*LyapSoma2(:,idxIteracao+1);
    Lyap3(:,idxIteracao+1) = (1/(tempo-t_init))*LyapSoma3(:,idxIteracao+1);
    % ------------------------------------------------------------------- %
    % Atualiza as variaveis para a proxima iteracao
    Tempo(idxIteracao+1) = tempo;
    
    % Reinicializa o vetor com as condicoes iniciais
    y010 = Y(:,1);
    y020 = Y(:,2);
    y030 = Y(:,3);
    
    mAux3 = mAux1 + delta*uk;
    
    y040 = mAux3(:,1);
    y050 = mAux3(:,2);
    y060 = mAux3(:,3);
    y070 = mAux3(:,4);
    y080 = mAux3(:,5);
    y090 = mAux3(:,6);
    y100 = mAux3(:,7);
    y110 = mAux3(:,8);
    y120 = mAux3(:,9);
    % ------------------------------------------------------------------- %
    
end
%% ===================================================================== %%
Lyap = [ Lyap1; Lyap2; Lyap3 ];
Lyap(:,end)
%% ===================================================================== %%









%% ===================================================================== %%
%% EOF: End Of File
%% ===================================================================== %%