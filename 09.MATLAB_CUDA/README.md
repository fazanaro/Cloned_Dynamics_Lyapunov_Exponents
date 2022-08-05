# Run CUDA or PTX Code on GPU using MATLAB 


## Description

The scripts inside this folder must be tested using the newest version of the CUDA and adapted based on the latest MATLAB release. They were extensively tested using CUDA 5.0 and MATLAB R2011-R2012.


## Compile a PTX File from a CU File

nvcc -ptx myfun.cu


## References

https://www.mathworks.com/help/parallel-computing/run-cuda-or-ptx-code-on-gpu.html;jsessionid=e5ee7a5358266354b018f5ce79c3#bsic4zj

https://www.mathworks.com/help/parallel-computing/gpu-cuda-and-mex-programming.html

https://www.mathworks.com/help/supportpkg/nvidia/ug/build-run-on-nvidia-hardware.html
