/* ***************************************************************************************************************************
 * 
 * VERSAO INICIAL: 23/07/2013
 *
 * ULTIMA ATUALIZACAO: 18/03/2014
 *
 *****************************************************************************************************************************
 * DESCRICAO:
 * 		- Implementa o algoritmo completo de estimacao dos expoentes de Lyapunov pela abordagem ClDyn;
 *
 * 		- Emprega o algoritmo RK4;
 *
 * 		- Analise para o modelo dinamico do oscilador forcado de Duffing;
 *
 * 		- Retorna os vetores que contem os globais e as somas ao final do intervalo de tempo considerado;
 *
 *
 *****************************************************************************************************************************
 * PENDENCIAS E MODIFICACOES:
 * 		- Passar "nMaxItera" e/ou "t_final" como argumento do kernel;
 * 			- Talvez procurar alguma coisa relacionada com aloca��o din�mica
 *
 * 		- Empregar "shared memory":
 * 			- O algoritmo come�a r�pido mas a medida que as itera��es v�o sendo contempladas, o algoritmo
 * 			fica cada vez mais lento atingindo um limiar de, aproximadamente, 0.17 seg por itera��o;
 *
 *
 *****************************************************************************************************************************
 * REFERENCIAS
 * 			[1] https://www.google.com/search?q=error%3A+constant+value+is+not+known+CUDA&ie=UTF-8&sa=Search&channel=fe&client=browser-ubuntu&hl=en
 *
 * 			[2] http://stackoverflow.com/questions/9936376/how-to-defined-a-cuda-shared-memory-with-a-size-known-at-run-time
 *
 * 			[3] http://stackoverflow.com/questions/5531247/allocating-shared-memory/5531640#5531640
 *
 * 			[4] Referencias sobre a constru��o de ondas quadradas em C:
 * 			http://stackoverflow.com/questions/1073606/is-there-a-one-line-function-that-generates-a-triangle-wave
 *
 *
 *****************************************************************************************************************************
 */

#include <stdio.h>
#include <math.h>
#include <sys/time.h>

#include <cuda.h>
#include <cuda_runtime.h>	// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime_api.h>



/* ************************************************************************************************************************ */

#define nMaxIteracoes 	800		//( (t_final - t_init) / t_gsr );
#define nIntegra 		50		//( t_gsr / h );

/* ************************************************************************************************************************ */



/* ************************************************************************************************************************ */
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


/* ************************************************************************************************************************ */
/* ************************************************************************************************************************ */
//															KERNEL															//
/* ************************************************************************************************************************ */
/* ************************************************************************************************************************ */
__global__ void Duffing1989RK4_ClDyn(
		double * outLyap1, 	
		double * outLyap2, 
		double * outLyap3,
		double * vX0,
		double * vY0,
		double * vZ0,
		const double * Gamma,
		const double * Epsilon,
		const double * Omega,
		const unsigned int numElements) {


	/* ******************************************************************************************************************** */
	// AQUISICAO DOS INDICES REFERENTES AHS THREADS E AOS BLOCOS															//
	/* ******************************************************************************************************************** */

	// Work out which thread we are
	size_t const globalThreadIdx = calculateGlobalIndex();

	// If we're off the end, return now
	if (globalThreadIdx >= numElements) {
		return;
	}

	/* ******************************************************************************************************************** */
	// DEFINE A DIMENSAO DO SISTEMA DINAMICO A SER ANALISADO																//
	/* ******************************************************************************************************************** */

	const int dim = 3;

	/* ******************************************************************************************************************** */
	// DEFINE O PASSO DE INCREMENTO DO INTEGRADOR																			//
	/* ******************************************************************************************************************** */

	const double h = 0.01;//H[globalThreadIdx];

	/* ******************************************************************************************************************** */
	// INICIALIZACAO DOS PARAMETROS DO SISTEMA DINAMICO																		//
	/* ******************************************************************************************************************** */

	double epsilon 	= Epsilon[globalThreadIdx];
	double gamma 	= Gamma[globalThreadIdx];
	double omega 	= Omega[globalThreadIdx];

	/* ******************************************************************************************************************** */
	// DEFINICAO DAS CONDICOES INICIAIS E OPCOES DE INTEGRACAO																//
	/* ******************************************************************************************************************** */

	// Instante inicial de integracao
	const double t_init = 0;
	double tempo = 0;

	// Define o instante de tempo final de simulacao
	//const double t_final = 0.5;

	// Define o intervalo de tempo para o qual o sistema (ORIGINAL + CLONES) eh integrado antes da aplicacao do procedimento
	// de reortonormizacao de Gram-Schimidt
	const double t_gsr = 0.5;

	// Define o numero maximo de iteracoes
	//		- Deve ser definido como constante caso seja interessante empregar em indices de vetores;
	//const int nMaxIteracoes = ( (t_final - t_init) / t_gsr );

	// Define quantas iteracoes do algoritmo RK4 serao executadas a cada iteracao
	//const int nIntegra = ( t_gsr / h );

	/* ******************************************************************************************************************** */
	// ALOCA MEMORIA PARA AS VARIAVEIS NECESSARIAS AO CALCULO DO INTEGRADOR 'RK4'											//
	/* ******************************************************************************************************************** */

	// Coeficientes parciais para o sistema ORIGINAL
	double k1y1, k2y1, k3y1, k4y1;
	double k1y2, k2y2, k3y2, k4y2;
	double k1y3, k2y3, k3y3, k4y3;


	// Coeficientes parciais para os sistemas CLONADOS
	double k1y4, k2y4, k3y4, k4y4; 		// clones referente variavel y(1)
	double k1y5, k2y5, k3y5, k4y5;
	double k1y6, k2y6, k3y6, k4y6;

	double k1y7, k2y7, k3y7, k4y7; 		// clones referente variavel y(2)
	double k1y8, k2y8, k3y8, k4y8;
	double k1y9, k2y9, k3y9, k4y9;

	double k1y10, k2y10, k3y10, k4y10; // clones referente variavel y(3)
	double k1y11, k2y11, k3y11, k4y11;
	double k1y12, k2y12, k3y12, k4y12;


	// Coeficientes finais para o sistema ORIGINAL
	double ky1, ky2, ky3;


	// Coeficientes finais para os sistemas CLONADOS
	double ky4, ky5, ky6; 		// clones referente variavel y(1)
	double ky7, ky8, ky9; 		// clones referente variavel y(2)
	double ky10, ky11, ky12;	// clones referente variavel y(3)


	// Variaveis temporarias referente ah atualizacao dos estados
	double Y1, Y2, Y3; 		// sistema ORIGINAL

	double Y4, Y5, Y6; 		// clones referente variavel y(1)
	double Y7, Y8, Y9; 		// clones referente variavel y(2)
	double Y10, Y11, Y12;	// clones referente variavel y(3)

	/* ******************************************************************************************************************** */
	// ALOCA MEMORIA PARA AS VARIAVEIS NECESSARIAS AO PROCEDIMENTO GSR														//
	/* ******************************************************************************************************************** */

	double vk[dim][dim];
	double uk[dim][dim];

	double deltax[dim][dim];

	/* ******************************************************************************************************************** */
	// ALOCA MEMORIA PARA OS VETORES QUE ARMAZENAM OS VALORES DOS EXPOENTES													//
	/* ******************************************************************************************************************** */

	double LyapSoma[dim][nMaxIteracoes + 1];
	double Lyap[dim][nMaxIteracoes + 1];

	LyapSoma[0][0] = 0;
	LyapSoma[1][0] = 0;
	LyapSoma[2][0] = 0;

	Lyap[0][0] = 0;
	Lyap[1][0] = 0;
	Lyap[2][0] = 0;

	/* ******************************************************************************************************************** */
	// ALOCA VARIAVEIS AUXILIARES (contadores, loops, etc)																	//
	/* ******************************************************************************************************************** */

	unsigned int idxIteracoes = 0;
	unsigned int idxIntegra = 0;

	/* ******************************************************************************************************************** */
	// INICIALIZACAO DAS CONDICOES INICIAIS																					//
	/* ******************************************************************************************************************** */

	// Define o valor da perturbacao inicial aplicada aos clones
	const double delta = 0.0001;


	// Sistema ORIGINAL
	double y10 = vX0[globalThreadIdx];
	double y20 = vY0[globalThreadIdx];
	double y30 = vZ0[globalThreadIdx];

	// CLONES REFERENTE PARA AH VARIAVEL y(1)
	double y40 = y10 + delta;
	double y50 = y10;
	double y60 = y10;

	// CLONES REFERENTE PARA AH VARIAVEL y(2)
	double y70 = y20;
	double y80 = y20 + delta;
	double y90 = y20;

	// CLONES REFERENTE PARA AH VARIAVEL y(3)
	double y100 = y30;
	double y110 = y30;
	double y120 = y30 + delta;

	/* ******************************************************************************************************************** */
	// INICIALIZACAO DO MODULO PRINCIPAL																					//
	/* ******************************************************************************************************************** */

	while (idxIteracoes < nMaxIteracoes) {

		/* ================================================================================================================ */
		// INTEGRACAO DO MODELO DINAMICO
		while (idxIntegra < nIntegra) {

			/* ------------------------------------------------------------------------------------------------------------ */
			// 1o. PASSO
			/* ------------------------------------------------------------------------------------------------------------ */

			// 		- COEFICIENTES PARCIAIS PARA O SISTEMA ORIGINAL
			k1y1 = h*( y20 );
			k1y2 = h*( y10 - pow(y10,3) - epsilon*y20 + gamma*cos( omega*y30 ) );
			k1y3 = h*( 1 );

			k2y1 = h*( (y20+0.5*k1y2) );
			k2y2 = h*( (y10+0.5*k1y1) - pow((y10+0.5*k1y1),3) - epsilon*(y20+0.5*k1y2) + gamma*cos( omega*(y30+0.5*k1y3) ) );
			k2y3 = h*( 1 );

			k3y1 = h*( (y20+0.5*k2y2) );
			k3y2 = h*( (y10+0.5*k2y1) - pow((y10+0.5*k2y1),3) - epsilon*(y20+0.5*k2y2) + gamma*cos( omega*(y30+0.5*k2y3) ) );
			k3y3 = h*( 1 );

			k4y1 = h*( (y20+k3y2) );
			k4y2 = h*( (y10+k3y1) - pow((y10+k3y1),3) - epsilon*(y20+k3y2) + gamma*cos( omega*(y30+k3y3) ) );
			k4y3 = h*( 1 );


			// 		- COEFICIENTES PARCIAIS PARA OS CLONES
			k1y4 = h*( y70 );
			k1y5 = h*( y80 );
			k1y6 = h*( y90 );
			k1y7 = h*( y40 - pow(y40,3) - epsilon*y70 + gamma*cos( omega*y100 ) );
			k1y8 = h*( y50 - pow(y50,3) - epsilon*y80 + gamma*cos( omega*y110 ) );
			k1y9 = h*( y60 - pow(y60,3) - epsilon*y90 + gamma*cos( omega*y120 ) );
			k1y10 = h*( 1 );
			k1y11 = h*( 1 );
			k1y12 = h*( 1 );

			k2y4 = h*( (y70+0.5*k1y7) );
			k2y5 = h*( (y80+0.5*k1y8) );
			k2y6 = h*( (y90+0.5*k1y9) );
			k2y7 = h*( (y40+0.5*k1y4) - pow((y40+0.5*k1y4),3) - epsilon*(y70+0.5*k1y7) + gamma*cos( omega*(y100+0.5*k1y10) ) );
			k2y8 = h*( (y50+0.5*k1y5) - pow((y50+0.5*k1y5),3) - epsilon*(y80+0.5*k1y8) + gamma*cos( omega*(y110+0.5*k1y11) ) );
			k2y9 = h*( (y60+0.5*k1y6) - pow((y60+0.5*k1y6),3) - epsilon*(y90+0.5*k1y9) + gamma*cos( omega*(y120+0.5*k1y12) ) );
			k2y10 = h*( 1 );
			k2y11 = h*( 1 );
			k2y12 = h*( 1 );

			k3y4 = h*( (y70+0.5*k2y7) );
			k3y5 = h*( (y80+0.5*k2y8) );
			k3y6 = h*( (y90+0.5*k2y9) );
			k3y7 = h*( (y40+0.5*k2y4) - pow((y40+0.5*k2y4),3) - epsilon*(y70+0.5*k2y7) + gamma*cos( omega*(y100+0.5*k2y10) ) );
			k3y8 = h*( (y50+0.5*k2y5) - pow((y50+0.5*k2y5),3) - epsilon*(y80+0.5*k2y8) + gamma*cos( omega*(y110+0.5*k2y11) ) );
			k3y9 = h*( (y60+0.5*k2y6) - pow((y60+0.5*k2y6),3) - epsilon*(y90+0.5*k2y9) + gamma*cos( omega*(y120+0.5*k2y12) ) );
			k3y10 = h*( 1 );
			k3y11 = h*( 1 );
			k3y12 = h*( 1 );

			k4y4 = h*( (y70+k3y7) );
			k4y5 = h*( (y80+k3y8) );
			k4y6 = h*( (y90+k3y9) );
			k4y7 = h*( (y40+k3y4) - pow((y40+k3y4),3) - epsilon*(y70+k3y7) + gamma*cos( omega*(y100+k3y10) ) );
			k4y8 = h*( (y50+k3y5) - pow((y50+k3y5),3) - epsilon*(y80+k3y8) + gamma*cos( omega*(y110+k3y11) ) );
			k4y9 = h*( (y60+k3y6) - pow((y60+k3y6),3) - epsilon*(y90+k3y9) + gamma*cos( omega*(y120+k3y12) ) );
			k4y10 = h*( 1 );
			k4y11 = h*( 1 );
			k4y12 = h*( 1 );

			/* ------------------------------------------------------------------------------------------------------------ */
			// 2o. PASSO
			/* ------------------------------------------------------------------------------------------------------------ */

			// 		- COEFICIENTES PARA O SISTEMA ORIGINAL
			ky1 = (1/6.0)*( k1y1 + 2*k2y1 + 2*k3y1 + k4y1 );
			ky2 = (1/6.0)*( k1y2 + 2*k2y2 + 2*k3y2 + k4y2 );
			ky3 = (1/6.0)*( k1y3 + 2*k2y3 + 2*k3y3 + k4y3 );

			// 		- COEFICIENTES PARA OS CLONES
			ky4 = (1/6.0)*( k1y4 + 2*k2y4 + 2*k3y4 + k4y4 );
			ky5 = (1/6.0)*( k1y5 + 2*k2y5 + 2*k3y5 + k4y5 );
			ky6 = (1/6.0)*( k1y6 + 2*k2y6 + 2*k3y6 + k4y6 );

			ky7 = (1/6.0)*( k1y7 + 2*k2y7 + 2*k3y7 + k4y7 );
			ky8 = (1/6.0)*( k1y8 + 2*k2y8 + 2*k3y8 + k4y8 );
			ky9 = (1/6.0)*( k1y9 + 2*k2y9 + 2*k3y9 + k4y9 );

			ky10 = (1/6.0)*( k1y10 + 2*k2y10 + 2*k3y10 + k4y10 );
			ky11 = (1/6.0)*( k1y11 + 2*k2y11 + 2*k3y11 + k4y11 );
			ky12 = (1/6.0)*( k1y12 + 2*k2y12 + 2*k3y12 + k4y12 );

			/* ------------------------------------------------------------------------------------------------------------ */
			// 3o. PASSO:
			/* ------------------------------------------------------------------------------------------------------------ */

			//   - Atualizacao do sistema (aplicacao do PASSO de Euler);
			// SISTEMA ORIGINAL
			Y1 = y10 + ky1;
			Y2 = y20 + ky2;
			Y3 = y30 + ky3;

			// CLONES
			Y4 = y40 + ky4;
			Y5 = y50 + ky5;
			Y6 = y60 + ky6;

			Y7 = y70 + ky7;
			Y8 = y80 + ky8;
			Y9 = y90 + ky9;

			Y10 = y100 + ky10;
			Y11 = y110 + ky11;
			Y12 = y120 + ky12;

			/* ------------------------------------------------------------------------------------------------------------ */
			// ATUALIZACAO DAS CONDICOES PARA A PROXIMA ITERACAO
			/* ------------------------------------------------------------------------------------------------------------ */

			y10 = Y1;
			y20 = Y2;
			y30 = Y3;

			y40 = Y4;
			y50 = Y5;
			y60 = Y6;

			y70 = Y7;
			y80 = Y8;
			y90 = Y9;

			y100 = Y10;
			y110 = Y11;
			y120 = Y12;

			/* ------------------------------------------------------------------------------------------------------------ */
			// Atualiza o contador
			/* ------------------------------------------------------------------------------------------------------------ */

			idxIntegra++;

			/* ------------------------------------------------------------------------------------------------------------ */

		} // FINALIZACAO DA INTEGRACAO DO MODELO DINAMICO


		/* ================================================================================================================ */
		// Salva o instante de tempo final
		/* ================================================================================================================ */

		tempo = tempo + t_gsr;

		/* ================================================================================================================ */
		// CALCULA O VETOR DIFERENCA DE ESTADOS
		/* ================================================================================================================ */

		// Diferenca de estado referente ah variavel y(1)
		deltax[0][0] = Y1 - Y4;
		deltax[0][1] = Y1 - Y5;
		deltax[0][2] = Y1 - Y6;

		// Diferenca de estado referente ah variavel y(2)
		deltax[1][0] = Y2 - Y7;
		deltax[1][1] = Y2 - Y8;
		deltax[1][2] = Y2 - Y9;

		// Diferenca de estado referente ah variavel y(2)
		deltax[2][0] = Y3 - Y10;
		deltax[2][1] = Y3 - Y11;
		deltax[2][2] = Y3 - Y12;

		/* ================================================================================================================ */
		// PROCEDIMENTO DE REORTONORMALIZACAO DE GRAM-SCHIMIDT
		/* ================================================================================================================ */

		// Expoente Lyap1
		vk[0][0] = deltax[0][0];
		vk[1][0] = deltax[1][0];
		vk[2][0] = deltax[2][0];

		double Norm1 = sqrt( pow(vk[0][0],2)+pow(vk[1][0],2)+pow(vk[2][0],2) );

		uk[0][0] = vk[0][0] / Norm1;
		uk[1][0] = vk[1][0] / Norm1;
		uk[2][0] = vk[2][0] / Norm1;

		// Expoente Lyap2
		double pdi1 = ( uk[0][0]*deltax[0][1]+uk[1][0]*deltax[1][1]+uk[2][0]*deltax[2][1]) / ( uk[0][0]*uk[0][0]+uk[1][0]*uk[1][0]+uk[2][0]*uk[2][0] );

		vk[0][1] = deltax[0][1] - pdi1 * uk[0][0];
		vk[1][1] = deltax[1][1] - pdi1 * uk[1][0];
		vk[2][1] = deltax[2][1] - pdi1 * uk[2][0];

		double Norm2 = sqrt( pow(vk[0][1],2)+pow(vk[1][1],2)+pow(vk[2][1],2) );

		uk[0][1] = vk[0][1] / Norm2;
		uk[1][1] = vk[1][1] / Norm2;
		uk[2][1] = vk[2][1] / Norm2;

		// Expoente Lyap3
		double pdi2 = ( uk[0][0]*deltax[0][2]+uk[1][0]*deltax[1][2]+uk[2][0]*deltax[2][2]) / ( uk[0][0]*uk[0][0]+uk[1][0]*uk[1][0]+uk[2][0]*uk[2][0] );
		double pdi3 = ( uk[0][1]*deltax[0][2]+uk[1][1]*deltax[1][2]+uk[2][1]*deltax[2][2]) / ( uk[0][1]*uk[0][1]+uk[1][1]*uk[1][1]+uk[2][1]*uk[2][1] );

		vk[0][2] = deltax[0][2] - pdi2 * uk[0][0] - pdi3 * uk[0][1];
		vk[1][2] = deltax[1][2] - pdi2 * uk[1][0] - pdi3 * uk[1][1];
		vk[2][2] = deltax[2][2] - pdi2 * uk[2][0] - pdi3 * uk[2][1];

		double Norm3 = sqrt( pow(vk[0][2],2)+pow(vk[1][2],2)+pow(vk[2][2],2) );

		uk[0][2] = vk[0][2] / Norm3;
		uk[1][2] = vk[1][2] / Norm3;
		uk[2][2] = vk[2][2] / Norm3;

		/* ================================================================================================================ */
		// CALCULO DOS EXPOENTES DE LYAPUNOV
		/* ================================================================================================================ */

		// Somatorio das normas para ser utilizado no calculo do expoentes globais
		LyapSoma[0][idxIteracoes + 1] = LyapSoma[0][idxIteracoes] + log(Norm1 / delta);
		LyapSoma[1][idxIteracoes + 1] = LyapSoma[1][idxIteracoes] + log(Norm2 / delta);
		LyapSoma[2][idxIteracoes + 1] = LyapSoma[2][idxIteracoes] + log(Norm3 / delta);

		// Calcula os expoentes globais
		Lyap[0][idxIteracoes + 1] = (1 / (tempo - t_init)) * LyapSoma[0][idxIteracoes + 1];
		Lyap[1][idxIteracoes + 1] = (1 / (tempo - t_init)) * LyapSoma[1][idxIteracoes + 1];
		Lyap[2][idxIteracoes + 1] = (1 / (tempo - t_init)) * LyapSoma[2][idxIteracoes + 1];

		/* ================================================================================================================ */
		// INICIALIZACAO DAS CONDICOES INICIAIS PARA A PROXIMA ITERACAO
		/* ================================================================================================================ */

		// Sistema ORIGINAL
		y10 = Y1;
		y20 = Y2;
		y30 = Y3;

		// CLONES REFERENTE PARA AH VARIAVEL y(1)
		y40 = Y1 + delta*uk[0][0];
		y50 = Y1 + delta*uk[0][1];
		y60 = Y1 + delta*uk[0][2];

		// CLONES REFERENTE PARA AH VARIAVEL y(2)
		y70 = Y2 + delta*uk[1][0];
		y80 = Y2 + delta*uk[1][1];
		y90 = Y2 + delta*uk[1][2];

		// CLONES REFERENTE PARA AH VARIAVEL y(3)
		y100 = Y3 + delta*uk[2][0];
		y110 = Y3 + delta*uk[2][1];
		y120 = Y3 + delta*uk[2][2];

		/* ================================================================================================================ */
		// Atualizacoes para a iteracao seguinte
		/* ================================================================================================================ */

		idxIteracoes++;
		idxIntegra = 0;

		/* ================================================================================================================ */

	} // FIM DO LOOP REFERENTES AHS ITERACOES

	/* ******************************************************************************************************************** */

	__syncthreads();

	/* ******************************************************************************************************************** */
	// RETORNO DO KERNEL: EXPOENTE GLOBAL																					//
	/* ******************************************************************************************************************** */

	for (unsigned int ii = 0; ii <= nMaxIteracoes; ii++){

		outLyap1[globalThreadIdx*nMaxIteracoes + ii] = Lyap[0][ii];
		outLyap2[globalThreadIdx*nMaxIteracoes + ii] = Lyap[1][ii];
		outLyap3[globalThreadIdx*nMaxIteracoes + ii] = Lyap[2][ii];

	}

	/* ******************************************************************************************************************** */

} // FIM DO KERNEL
/* ************************************************************************************************************************ */
