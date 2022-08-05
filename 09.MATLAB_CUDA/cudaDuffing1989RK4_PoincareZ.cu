/* ***************************************************************************************************************
 * 
 * VERSAO INICIAL: 12/02/2014
 * 
 * ULTIMA ATUALIZACAO: 19/02/2014
 * 
 * ***************************************************************************************************************
 * 
 * DESCRICAO:
 * 	- Constroi a secao de Poincare, tomando como base a amostragem da variavel de estado 'Z', para o modelo do
 * 	oscilador forcado de Duffing;
 * 	
 * 	- FAZ USO DA CARACTERISTICA DA ALOCACAO DE MEMORIA EM C (ROW-MAJOR LAYOUT): AS LINHAS DE UMA MATRIZ NXM PODEM
 * 	SER ALINHADAS NO FORMATO DE UM VETOR DE COMPRIMENTO TOTAL N*M (ver pg 71, sec 4.2, [Kirk, Hwu (2013)]);
 *
 * 	- Assim sendo, esse kernel retorna um vetor de comprimento igual ah nMaxPoincare*length(vEpsilon). NOTE QUE
 * 	A INDICACAO DE QUANDO SE INICIA UM NOVO TRECHO CORRESPONDENTE AH MODIFICACAO DO PARAMETRO DE CONTROLE EH
 * 	DEFINIDA PELA POSICAO DA THREAD;
 * 	
 * ***************************************************************************************************************
 * 
 * REFERENCIAS:
 * 
 * 	http://www.cprogramming.com/tutorial/cfileio.html
 * 	
 * 	http://stackoverflow.com/questions/11573974/write-to-txt-file
 * 	
 * *************************************************************************************************************** 	 
 */

#include <stdio.h>
#include <math.h>
#include <sys/time.h>

#include <cuda.h>
#include <cuda_runtime.h>	// For the CUDA runtime routines (prefixed with "cuda_")

//#define PI 3.14159265358979323846264338327







/* *************************************************************************************************************** */
/* DESCRICAO:
 * 	- Work out which piece of the global array this thread should operate on;
 * 	- Por ter sido definido como DEVICE, somente pode ser envocado pelo proprio "device", i.e., somente pela GPU;
 * 	- Essa funcao foi definida para facilitar a manipulacao dos indices envolvidos dentro do kernel principal;
 * 	- Utiliza como base o script originalmente desenvolvido para a iteracao do conjunto de Mandelbrot (MATLAB);
 */

__device__ size_t calculateGlobalIndex() {
	// Which block are we?
	size_t const globalBlockIndex = blockIdx.x + blockIdx.y * gridDim.x;
	// Which thread are we within the block?
	size_t const localThreadIdx = threadIdx.x + blockDim.x * threadIdx.y;
	// How big is each block?
	//	- As entradas serao matrizes;
	size_t const threadsPerBlock = blockDim.x*blockDim.y;
	// Which thread are we overall?
	return localThreadIdx + globalBlockIndex*threadsPerBlock;
}

/* *************************************************************************************************************** */

/* DESCRICAO:
 *  - Main entry point;
 *  - Works out where the current thread should read/write to global memory and calls doIterations to do the actual work.
 */
__global__ void Duffing_PoincareZ(
		double * X,
		double * Y,
		const double * X0,
		const double * Y0,
		const double * Z0,
		const double * H,
		const double * Gamma,
		const double * Epsilon,
		const double * Omega,
		const double Threshold,
		const unsigned int numel ) {

	/* **************************************************************************************** */
	// AQUISICAO DOS INDICES REFERENTES AHS THREADS E BLOCOS

	// Work out which thread we are
	size_t const globalThreadIdx = calculateGlobalIndex();

	// If we're off the end, return now
	if (globalThreadIdx >= numel) {
		return;
	}

	/* **************************************************************************************** */
	// DEFINE O NUMERO MAXIMO DE PONTOS QUE ESTARAO PRESENTES NA SECAO

	const int numMaxElementsPoincare = 5;

	/* **************************************************************************************** */
	// INICIALIZACAO DOS PARAMETROS DO SISTEMA DINAMICO

	double epsilon = Epsilon[globalThreadIdx];
	double gamma 	= Gamma[globalThreadIdx];
	double omega 	= Omega[globalThreadIdx];

	/* **************************************************************************************** */
	// DEFINE O PASSO DE INCREMENTO DO INTEGRADOR

	const double h = H[globalThreadIdx];

	/* **************************************************************************************** */
	// CARREGA AS CONDICOES INICIAIS PASSADAS AO KERNEL PELO SCRIPT EM MATLAB

	double x0 = X0[globalThreadIdx];
	double y0 = Y0[globalThreadIdx];
	double z0 = Z0[globalThreadIdx];

	/* **************************************************************************************** */
	// DEFINICAO DAS CONDICOES INICIAIS E OPCOES DE INTEGRACAO

	// Instante inicial de integracao
	const double t_init = 0;

	// Define o instante de tempo final de simulacao
	//const double t_final = 30.0;

	// Define o tempo de transitorio
	const double t_trans = 10.0;

	// Define o indice do transitorio, ou seja, os pontos que deverao ser desconsiderados
	// 		- Esse valor eh estimado a partir da construcao da secao de Poincare
	const int index_trans = round( (t_trans-t_init)/h );

	/* **************************************************************************************** */
	// ALOCA MEMORIA

	// Inicializa a variavel que representa o numero de pontos adquiridos
	int numPtosPoincareAdquiridos = 0;


	// Inicializacao do vetor que contem os indices do pontos que estao sobre
	// (ou muito proximos) da secao de Poincare
	int index[numMaxElementsPoincare];


	// Aloca memoria para os vetores que definem a secao de Poincare
	double vPoincareX[numMaxElementsPoincare];
	double vPoincareY[numMaxElementsPoincare];


	// Aloca memoria para os vetores auxiliares
	double varX[2], varY[2], varZ[2];

	double 	vAmostragem[2];

	/* **************************************************************************************** */
	// INICIALIZA OS PARAMETROS PARA A INTERPOLACAO DOS DADOS

	double CoefAngular;
	double CoefLinear;

	/* **************************************************************************************** */
	// INICIALIZA A VARIAVEL DE CONTAGEM PARA A INTEGRACAO DO MODELO DINAMICO

	unsigned int numContagemIteracoes = 0;

	/* **************************************************************************************** */
	// INICIALIZACAO DAS CONDICOES INICIAIS E DOS COEFICIENTES DO ALGORITMO RK4

	double k1x, k2x, k3x, k4x;
	double k1y, k2y, k3y, k4y;
	double k1z, k2z, k3z, k4z;

	double x, y, z;

	/* **************************************************************************************** */
	// INTEGRA O MODELO DINAMICO

	while (numPtosPoincareAdquiridos <= numMaxElementsPoincare){

		// ==================================================================================== //
		//							INTEGRACAO DO MODELO DINAMICO								//
		// ==================================================================================== //
		// 1o. PASSO: 	CALCULA OS COEFICIENTES PARCIAIS

		k1x = h*( y0 );
		k1y = h*( x0 - pow(x0,3) - epsilon*y0 + gamma*cos( omega*z0 ) );
		k1z = h*( 1 );

		k2x = h*( (y0+0.5*k1y) );
		k2y = h*( (x0+0.5*k1x) - pow((x0+0.5*k1x),3) - epsilon*(y0+0.5*k1y) + gamma*cos( omega*(z0+0.5*k1z) ) );
		k2z = h*( 1 );

		k3x = h*( (y0+0.5*k2y) );
		k3y = h*( (x0+0.5*k2x) - pow((x0+0.5*k2x),3) - epsilon*(y0+0.5*k2y) + gamma*cos( omega*(z0+0.5*k2z) ) );
		k3z = h*( 1 );

		k4x = h*( (y0+k3y) );
		k4y = h*( (x0+k3x) - pow((x0+k3x),3) - epsilon*(y0+k3y) + gamma*cos( omega*(z0+k3z) ) );
		k4z = h*( 1 );

		/* ----------------------------------------------------------------------------------- */
		// 2o. PASSO: 	ATUALIZA AS VARIAVEIS DE ESTADO

		x = x0 + (1/6.0)*( k1x + 2*k2x + 2*k3x + k4x );
		y = y0 + (1/6.0)*( k1y + 2*k2y + 2*k3y + k4y );
		z = z0 + (1/6.0)*( k1z + 2*k2z + 2*k3z + k4z );

		/* ----------------------------------------------------------------------------------- */
		// 3o. PASSO: 	ATUALIZA AS VARIAVEIS DE ESTADO PARA A PROXIMA ITERACAO

		//x0 = x;
		//y0 = y;
		//z0 = z;

		// ==================================================================================== //
		// 							ATUALIZA O NUMERO DE ITERACOES								//
		// ==================================================================================== //

		numContagemIteracoes = numContagemIteracoes + 1;

		// ==================================================================================== //
		// 					FAZ A ANALISE DOS PONTOS: INTERPOLACAO DOS DADOS					//
		// ==================================================================================== //

		if (numContagemIteracoes >= 2){
			//if (numContagemIteracoes >= index_trans){

			// -------------------------------------------------------------------------------- //

			varX[0] = x0;
			varY[0] = y0;
			varZ[0] = z0;

			varX[1] = x;
			varY[1] = y;
			varZ[1] = z;

			vAmostragem[0] = sin( omega*varZ[0] );
			vAmostragem[1] = sin( omega*varZ[1] );

			// -------------------------------------------------------------------------------- //

			if (vAmostragem[0] == Threshold){
				// ---------------------------------------------------------------------------- //
				// O PONTO ESTAH EXATAMENTE SOBRE A SECAO DE POINCARE
				// ---------------------------------------------------------------------------- //
				// SALVA OS PONTOS

				vPoincareX[numPtosPoincareAdquiridos] = x0;
				vPoincareY[numPtosPoincareAdquiridos] = y0;

				// ---------------------------------------------------------------------------- //
				// ARMAZENA OS INDICES

				index[numPtosPoincareAdquiridos] = numContagemIteracoes;

				// ---------------------------------------------------------------------------- //
				// ATUALIZA O NUMERO TOTAL DE PONTOS ADQUIRIDOS

				numPtosPoincareAdquiridos = numPtosPoincareAdquiridos + 1;

				// ---------------------------------------------------------------------------- //

			}else if ( (vAmostragem[0] <= Threshold) && (vAmostragem[1] >= Threshold) ){
				// ---------------------------------------------------------------------------- //
				// CONDICAO PARA AMOSTRAGEM NO SENTIDO POSITIVO
				// ---------------------------------------------------------------------------- //
				// Se o ponto nao estah sobre a secao de Poincare, serah verificado se ocorreu
				// cruzamento. Se sim, serah salvo o indice referente ao ponto imediatamente
				// anterior ah secao de Poincare
				// ---------------------------------------------------------------------------- //
				// AMOSTRAGEM DE 'X'

				// Calcula-se os coeficientes da reta que ajusta os pontos
				CoefAngular = ( vAmostragem[1]-vAmostragem[0] ) / ( varX[1]-varX[0] );
				CoefLinear  = ((vAmostragem[1]+vAmostragem[0])  - CoefAngular*( varX[1]+varX[0] ) )/2.0;

				// Calcula os pontos que seriam obtidos caso a secao de Poincare fosse atingida
				// de maneira ideal
				vPoincareX[numPtosPoincareAdquiridos] = (Threshold - CoefLinear)/CoefAngular;

				// ---------------------------------------------------------------------------- //
				// AMOSTRAGEM DE 'Y'

				// Calcula-se os coeficientes da reta que ajusta os pontos
				CoefAngular = ( vAmostragem[1]-vAmostragem[0] ) / ( varY[1]-varY[0] );
				CoefLinear  = ((vAmostragem[1]+vAmostragem[0])  - CoefAngular*( varY[1]+varY[0] ) )/2.0;

				// Calcula os pontos que seriam obtidos caso a secao de Poincare fosse atingida
				// de maneira ideal
				vPoincareY[numPtosPoincareAdquiridos] = (Threshold - CoefLinear)/CoefAngular;

				// ---------------------------------------------------------------------------- //
				// ARMAZENA OS INDICES

				index[numPtosPoincareAdquiridos] = numContagemIteracoes;

				// ---------------------------------------------------------------------------- //
				// ATUALIZA O NUMERO TOTAL DE PONTOS ADQUIRIDOS

				numPtosPoincareAdquiridos = numPtosPoincareAdquiridos + 1;

				// ---------------------------------------------------------------------------- //

			}

		}

		// ==================================================================================== //
		// 					ATUALIZA AS CONDICOES PARA A PROXIMA ITERACAO						//
		// ==================================================================================== //

		x0 = x;
		y0 = y;
		z0 = z;

		// ==================================================================================== //

	}

	/* **************************************************************************************** */
	//								RETORNA A SAIDA DO KERNEL
	/* **************************************************************************************** */

	for (int ii = 0; ii < numMaxElementsPoincare; ii++){

		X[globalThreadIdx*numMaxElementsPoincare + ii] = vPoincareX[ii];
		Y[globalThreadIdx*numMaxElementsPoincare + ii] = vPoincareY[ii];

	}

	/* **************************************************************************************** */

}

