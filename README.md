# Cloned_Dynamics_Lyapunov_Exponents


### Descrition

This repository contains scripts developed using MATLAB and C languages which are used to implement the Cloned Dynamics methodology. This approach aims to compute the Lyapunov exponents associate to nonlinear dynamical systems, specially the ones with some kind of non-smothness.


### History

The Cloned Dynamics approach was proposed by Diogo Coutinho Soriano (https://www.diogosoriano.com/) and me, during the very beginning of our phd, in 2008, inspired in the thesis of a good friend, Prof. Dr. Reinaldo Gonçalves Nogueira, intitled "Técnicas Alternativas de Reconhecimento de Caos em Sistemas com Dinâmica Complexa: Análise de um Sistema com Descontinuidade" (http://repositorio.unicamp.br/jspui/handle/REPOSIP/260285). In his work, Prof. Reinaldo proposed an implementation wich was capable to compute the Lyapunov spectrum of nonsmooth nonlinear dynamical system, particullarly, systems subject to impacts. The system studied was a pendulum with restrictions. 

The main idea of the Cloned Dynamics approach is to compute 

- colocar uma figura



### How to Cite

The Cloned Dynamics approach was used do develop some works. The main results were published in some importa journals as Chaos, Communications in Nonlinear Science and Numerical Simulation, International Journal of Bifurcation and Chaos, and in a few conferences and symposiums.

To cite the Cloned Dynamics approach, please use the following:

- colocar os bibtexs das referências


If any other work was helpfull to you research, the following bibitex citations could be usefull:

- colocar os bibtexs das referências



### Organization of the Repository

The repository is organized based on the following folder structure:

- Aux_Functions: you need to add this folder to you MATLAB path. It contains some auxiliar functions, as, for instance, the implementation of 4th order Runge-Kutta integration.


- MATLAB: in this folder, you will find the implementations of some of the most important dynamical systems, as the Duffing Oscillator, the Chua oscillator, the FitzHugh-Nagumo neuronal model, and others. Is was employed the MATLAB language. There are comments and descriptions inside the scripts to improve the understanding of each code.

You will find the scripts to generate the time evolution of each state variable, the Poincaré section, the implementation of the Cloned Dynamics approach to compute the Lyapunov exponents and the Lyapunov spectrum (the value of the exponent related to the variation of one constante of the model).


- MATLAB_parfor: the computation of the Lyapunov exponents is usually very computational expensive. Here, it was implemented the computation of the Lyapunov spectrum using the Parallel computational toolbox of the MATLAB. It is necessary to adjust the number of cores used during each simulation.


- C: as a great friend of mine, André Luiz Delai (http://lattes.cnpq.br/4609660221170758), usually says, "C is the mother language of all". In this folder, there are some implementations using the C language, basically, only to compute the Lyapunov exponents and the Lyapunov spectrum. As discussed in the work [citar CNCNS], it is possible to improve the computational cost to (i.e. decrease the total amount of time to) realize such computation when compare to the MATLAB implementation.
