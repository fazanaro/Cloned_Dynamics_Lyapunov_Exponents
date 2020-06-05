# Cloned_Dynamics_Lyapunov_Exponents


## (Under construction)

### Descrition

This repository contains scripts developed using MATLAB and C languages which are used to implement the Cloned Dynamics methodology. This approach aims to compute the Lyapunov exponents associate to nonlinear dynamical systems, specially the ones with some kind of non-smothness.


### History

The Cloned Dynamics approach was proposed by Diogo Coutinho Soriano (https://www.diogosoriano.com/) and me, with help and oriented by our professors and friends, Dr. Romis Attux, Dr. Ricardo Suyama and Dr. Marconi Kolm Madrid, during the very beginning of our phd, in 2008, inspired by the thesis of a good friend, Prof. Dr. Reinaldo Gonçalves Nogueira, intitled "Técnicas Alternativas de Reconhecimento de Caos em Sistemas com Dinâmica Complexa: Análise de um Sistema com Descontinuidade" (http://repositorio.unicamp.br/jspui/handle/REPOSIP/260285). In his work, Prof. Reinaldo proposed an implementation which was capable to compute the Lyapunov spectrum of non-smooth nonlinear dynamical system, particularly, systems subject to impacts. The system studied was a pendulum with restrictions.


The main idea of the Cloned Dynamics approach is to compute 

- colocar uma figura



### How to Cite

The Cloned Dynamics approach was used do develop some works. The main results were published in some important journals, such as Chaos, Communications in Nonlinear Science and Numerical Simulation, International Journal of Bifurcation and Chaos, and in a few conferences and symposiums.

To cite the Cloned Dynamics approach, please use the following:

> @ARTICLE{article:SorianoFazanaro2012,
> author = {Soriano, D. C. and Fazanaro, F. I. and Suyama, R. and Oliveira, J. R. and Attux, R. and Madrid, M. K.},
> title = {{A method for Lyapunov spectrum estimation using cloned dynamics and its application to the discontinuously-excited FitzHugh-Nagumo model}},
> journal = {Nonlinear Dynamics},
> year = {2012},
> volume = {67},
> pages = {413--424},
> number = {1},
> month = {Jan},
> doi = {10.1007/s11071-011-9989-2},
> publisher = {Springer Netherlands}
> }


If any other work was helpfull to you research, the following bibitex citations could be usefull:

@Article{article:Soriano2012b,
  author  = {Soriano, D. C. and Attux, R. and Suyama, R. and Romano, J. M. T.},
  title   = {Searching for specific periodic and chaotic oscillations in a periodically-excited Hodgkin-Huxley model},
  journal = {International Journal of Bifurcation and Chaos},
  year    = {2012},
  volume  = {22},
  number  = {1},
  pages   = {1230006},
}

@Article{article:Fazanaro2016,
  author  = {Fazanaro, F. I. and Soriano, D. C. and Suyama, R. and Madrid, M. K. and Oliveira, J. R. and Mu\~{n}oz, I. B. and Attux, R.},
  title   = {Numerical characterization of nonlinear dynamical systems using parallel computing: The role of GPUs approach},
  journal = {Communication in Nonlinear Science and Numerical Simulation},
  year    = {2016},
  volume  = {37},
  pages   = {143-162},
  doi     = {10.1016/j.cnsns.2015.12.021},
}



### Organization of the Repository

The repository is organized based on the following folder structure:

- Aux_Functions: you need to add this folder to you MATLAB path. It contains some auxiliar functions, as, for instance, the implementation of 4th order Runge-Kutta integration.


- MATLAB: in this folder, you will find the implementations of some of the most important dynamical systems, as the Duffing Oscillator, the Chua oscillator, the FitzHugh-Nagumo neuronal model, and others. Is was employed the MATLAB language. There are comments and descriptions inside the scripts to improve the understanding of each code.

You will find the scripts to generate the time evolution of each state variable, the Poincaré section, the implementation of the Cloned Dynamics approach to compute the Lyapunov exponents and the Lyapunov spectrum (the value of the exponent related to the variation of one constante of the model).


- MATLAB_parfor: the computation of the Lyapunov exponents is usually very computational expensive. Here, it was implemented the computation of the Lyapunov spectrum using the Parallel computational toolbox of the MATLAB. It is necessary to adjust the number of cores used during each simulation.


- C: as a great friend of mine, André Luiz Delai (http://lattes.cnpq.br/4609660221170758), usually says, "C is the mother language of all". In this folder, there are some implementations using the C language, basically, only to compute the Lyapunov exponents and the Lyapunov spectrum. As discussed in the work [citar CNCNS], it is possible to improve the computational cost to (i.e. decrease the total amount of time to) realize such computation when compare to the MATLAB implementation.
