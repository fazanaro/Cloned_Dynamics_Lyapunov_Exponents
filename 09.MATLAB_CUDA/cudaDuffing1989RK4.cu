/* ***************************************************************************************************************
 *
 * VERSAO INICIAL: 13/02/2014
 *
 * ULTIMA ATUALIZACAO: 13/02/2014
 *
 * ***************************************************************************************************************
 *
 * DESCRICAO:
 * 		- Integra o modelo dinamico o oscilador forcado de Duffing;
 *
 *
 * ***************************************************************************************************************
 *
 * 	LIMITACAO:
 * 		- Retorna apenas o estado do sistema no instante final do intervalo de tempo considerado;
 *
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

#include <math.h>
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



/* *************************************************************************************************************** */
/* DESCRICAO:
 *  - Main entry point;
 *  - Works out where the current thread should read/write to global memory and calls doIterations to do the actual work.
 */
__global__ void processDuffingTimeEvolution(
                      double * X, 
                      double * Y,
                      double * Z,
                      const double * X0, 
                      const double * Y0,
                      const double * Z0,
                      const double * H,
                      const double * Gamma,
                      const double * Epsilon,
                      const double * Omega,
                      const unsigned int nIntegra,
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
	// INICIALIZACAO DOS PARAMETROS DO SISTEMA DINAMICO

	double epsilon = Epsilon[globalThreadIdx];
	double gamma = Gamma[globalThreadIdx];
	double omega = Omega[globalThreadIdx];

	/* **************************************************************************************** */
	// DEFINE O PASSO DE INCREMENTO DO INTEGRADOR

	const double h = H[globalThreadIdx];

	/* **************************************************************************************** */
	// CARREGA AS CONDICOES INICIAIS PASSADAS AO KERNEL PELO SCRIPT EM MATLAB

	double x0 = X0[globalThreadIdx];
	double y0 = Y0[globalThreadIdx];
	double z0 = Z0[globalThreadIdx];

	/* **************************************************************************************** */
	// INICIALIZA A VARIAVEL DE CONTAGEM PARA A INTEGRACAO DO MODELO DINAMICO

	unsigned int count = 0;

	/* **************************************************************************************** */
	// Inicializacao de condicoes iniciais diversas
	double k1x, k2x, k3x, k4x;
	double k1y, k2y, k3y, k4y;
	double k1z, k2z, k3z, k4z;

	double x, y, z;

	/* **************************************************************************************** */
    // INTEGRA O MODELO DINAMICO

	while ( count < nIntegra ){

       	/* ----------------------------------------------------------------------------------- */
       	// 1o. PASSO:
       	//		- CALCULA OS COEFICIENTES PARCIAIS
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
       	// 2o. PASSO:
       	//		- ATUALIZA AS VARIAVEIS DE ESTADO
       	x = x0 + (1/6.0)*( k1x + 2*k2x + 2*k3x + k4x );
       	y = y0 + (1/6.0)*( k1y + 2*k2y + 2*k3y + k4y );
       	z = z0 + (1/6.0)*( k1z + 2*k2z + 2*k3z + k4z );
       	/* ----------------------------------------------------------------------------------- */
       	// 3o. PASSO:
       	//		- ATUALIZA AS VARIAVEIS DE ESTADO PARA A PROXIMA ITERACAO
       	x0 = x;
       	y0 = y;
       	z0 = z;
       	/* ----------------------------------------------------------------------------------- */
       	// 4o. PASSO:
       	//		- ATUALIZA O CONTADOR
       	count++;
       	/* ----------------------------------------------------------------------------------- */
	}

   	/* **************************************************************************************** */
	// RETORNA O ESTADO FINAL DO SISTEMA APOS O NUMERO DE ITERACOES MAXIMO CONSIDERADO

	X[globalThreadIdx] = x;
    Y[globalThreadIdx] = y;
    Z[globalThreadIdx] = z;


    /* **************************************************************************************** */


}
/* ************************************************************************************************ */
// EOF: End Of File
/* ************************************************************************************************ */
