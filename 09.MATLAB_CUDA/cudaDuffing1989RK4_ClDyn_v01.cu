/*
 * Versao inicial: 27/02/2013
 *
 * Ultima atualizacao: 27/02/2013
 *
 * DESCRICAO:
 * 		- Implementa a integracao do sistema original e clones referente ah analise do modelo
 * 		dinamico do oscilador forcado de Duffing;
 */


#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>	// For the CUDA runtime routines (prefixed with "cuda_")

//#define PI 3.14159265358979323846264338327


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




/* DESCRICAO:
 *  - Main entry point;
 *  - Works out where the current thread should read/write to global memory and calls doIterations to do the actual work.
 */
__global__ void processDuffing1989RK4_ClDyn(
                      double * Y1out, double * Y2out, double * Y3out,
                      double * Y4out, double * Y5out, double * Y6out,
                      double * Y7out, double * Y8out, double * Y9out,
                      double * Y10out, double * Y11out, double * Y12out,
                      const double * Y0_var1, const double * Y0_var2, const double * Y0_var3,
                      const double * Y0_var4, const double * Y0_var5, const double * Y0_var6,
                      const double * Y0_var7, const double * Y0_var8, const double * Y0_var9,
                      const double * Y0_var10, const double * Y0_var11, const double * Y0_var12,
                      const double * H,
                      const double * Gamma,
                      const double * Epsilon,
                      const double * Omega,
                      const unsigned int nMaxItera, 
                      const unsigned int numel ) {
	
	/* **************************************************************************************************** */
	// ALOCA MEMORIA PARA AS VARIAVEIS NECESSARIAS AO CALCULO DO INTEGRADOR 'RK4'

	// Coeficientes parciais para o sistema ORIGINAL
	double k1y1, k2y1, k3y1, k4y1;
	double k1y2, k2y2, k3y2, k4y2;
	double k1y3, k2y3, k3y3, k4y3;

	// Coeficientes parciais para os sistemas CLONADOS
	double k1y4, k2y4, k3y4, k4y4;		// clones referente variavel y(1)
	double k1y5, k2y5, k3y5, k4y5;
	double k1y6, k2y6, k3y6, k4y6;

	double k1y7, k2y7, k3y7, k4y7;		// clones referente variavel y(2)
	double k1y8, k2y8, k3y8, k4y8;
	double k1y9, k2y9, k3y9, k4y9;

	double k1y10, k2y10, k3y10, k4y10;	// clones referente variavel y(3)
	double k1y11, k2y11, k3y11, k4y11;
	double k1y12, k2y12, k3y12, k4y12;


	// Coeficientes finais para o sistema ORIGINAL
	double ky1, ky2, ky3;

	// Coeficientes finais para os sistemas CLONADOS
	double ky4, ky5, ky6;		// clones referente variavel y(1)
	double ky7, ky8, ky9;		// clones referente variavel y(2)
	double ky10, ky11, ky12;	// clones referente variavel y(3)


	// Variaveis temporarias referente ah atualizacao dos estados
	double Y1, Y2, Y3;		// sistema ORIGINAL

	double Y4, Y5, Y6;		// clones referente variavel y(1)
	double Y7, Y8, Y9;		// clones referente variavel y(2)
	double Y10, Y11, Y12;	// clones referente variavel y(3)
	/* **************************************************************************************************** */
	// INICIALIZA VARIAVEIS AUXILIARES DIVERSAS
	unsigned int idxItera = 0;
	/* **************************************************************************************************** */
	// AQUISICAO DOS INDICES REFERENTES AHS THREADS E BLOCOS

	// Work out which thread we are
    size_t const globalThreadIdx = calculateGlobalIndex();
    
    // If we're off the end, return now
    if (globalThreadIdx >= numel) {
        return;
    }
    /* **************************************************************************************************** */
    // INICIALIZACAO DOS PARAMETROS DO SISTEMA DINAMICO
    double h = H[globalThreadIdx];
    double epsilon = Epsilon[globalThreadIdx];
    double gamma   = Gamma[globalThreadIdx];
    double omega 	= Omega[globalThreadIdx];
    /* **************************************************************************************************** */
    // INICIALIZACAO DAS CONDICOES INICIAIS

	// Sistema ORIGINAL
    double y10 = Y0_var1[globalThreadIdx];
    double y20 = Y0_var2[globalThreadIdx];
    double y30 = Y0_var3[globalThreadIdx];

    // CLONES REFERENTE PARA AH VARIAVEL y(1)
    double y40 = Y0_var4[globalThreadIdx];
    double y50 = Y0_var5[globalThreadIdx];
    double y60 = Y0_var6[globalThreadIdx];

    // CLONES REFERENTE PARA AH VARIAVEL y(2)
    double y70 = Y0_var7[globalThreadIdx];
    double y80 = Y0_var8[globalThreadIdx];
    double y90 = Y0_var9[globalThreadIdx];

    // CLONES REFERENTE PARA AH VARIAVEL y(3)
    double y100 = Y0_var10[globalThreadIdx];
    double y110 = Y0_var11[globalThreadIdx];
    double y120 = Y0_var12[globalThreadIdx];
    /* **************************************************************************************************** */
    
    

    
    while ( idxItera < nMaxItera ){
    	/* -------------------------------------------------------------------------------- */
    	// 1o. PASSO
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
    	/* -------------------------------------------------------------------------------- */
    	// 2o. PASSO
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
    	/* -------------------------------------------------------------------------------- */
    	// 3o. PASSO:
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
    	/* -------------------------------------------------------------------------------- */
    	// ATUALIZACAO DAS CONDICOES PARA A PROXIMA ITERACAO
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
    	/* -------------------------------------------------------------------------------- */
    	// Atualiza o contador
    	idxItera++;
    	/* -------------------------------------------------------------------------------- */
    }
    
    // Retorna o estado final do sistema apos o numero de iteracoes maximo considerado
    Y1out[globalThreadIdx] = Y1;
    Y2out[globalThreadIdx] = Y2;
    Y3out[globalThreadIdx] = Y3;

    Y4out[globalThreadIdx] = Y4;
    Y5out[globalThreadIdx] = Y5;
    Y6out[globalThreadIdx] = Y6;

    Y7out[globalThreadIdx] = Y7;
    Y8out[globalThreadIdx] = Y8;
    Y9out[globalThreadIdx] = Y9;

    Y10out[globalThreadIdx] = Y10;
    Y11out[globalThreadIdx] = Y11;
    Y12out[globalThreadIdx] = Y12;
}
