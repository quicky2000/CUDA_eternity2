/* -*- C++ -*- */
/*    This file provide some convenience code to make code compilable
      without CUDA in particular cases
      Copyright (C) 2016  Julien Thevenon ( julien_thevenon at yahoo.fr )

      This program is free software: you can redistribute it and/or modify
      it under the terms of the GNU General Public License as published by
      the Free Software Foundation, either version 3 of the License, or
      (at your option) any later version.

      This program is distributed in the hope that it will be useful,
      but WITHOUT ANY WARRANTY; without even the implied warranty of
      MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
      GNU General Public License for more details.

      You should have received a copy of the GNU General Public License
      along with this program.  If not, see <http://www.gnu.org/licenses/>
*/

#ifndef _MY_CUDA_H_
#define _MY_CUDA_H_

#ifndef __NVCC__
class dim3
{
  public:
  inline dim3(uint32_t p_x = 1, uint32_t p_y = 1, uint32_t p_z = 1):
    x(p_x),
    y(p_y),
    z(p_z)
  {
  }

   uint32_t x;
   uint32_t y;
   uint32_t z;
};

#define CUDA_KERNEL(name,args...) void name(const dim3 & threadIdx, const dim3 & blockIdx, const dim3 & blockDim, const dim3 & gridDim,args)
#define CUDA_METHOD_HD_I inline
#define gpuErrChk(ans) ans
#define cudaFree free
#define cudaMalloc(ptr,size) (*ptr) = (std::remove_pointer<decltype(ptr)>::type)malloc(size);
#define cudaMemcpy(dest, src , size, direction) memcpy(dest, src, size);
#define launch_kernels(name,grid,block,args...) { dim3 l_blockIdx(0,0,0);                                      \
  for(l_blockIdx.z = 0 ; l_blockIdx.z < grid.z ; ++l_blockIdx.z)                                               \
    {                                                                                                          \
      for(l_blockIdx.y = 0 ; l_blockIdx.y < grid.y ; ++l_blockIdx.y)                                           \
	{                                                                                                      \
	  for(l_blockIdx.x = 0 ; l_blockIdx.x < grid.x ; ++l_blockIdx.x)                                       \
	    {                                                                                                  \
	      dim3 l_threadIdx(0,0,0);                                                                         \
	      for(l_threadIdx.z = 0 ; l_threadIdx.z < block.z ; ++l_threadIdx.z)                               \
		{                                                                                              \
		  for(l_threadIdx.y = 0 ; l_threadIdx.y < block.y ; ++l_threadIdx.y)                           \
		    {                                                                                          \
		      for(l_threadIdx.x = 0 ; l_threadIdx.x < block.x ; ++l_threadIdx.x)                       \
			{                                                                                      \
			  name(l_threadIdx, l_blockIdx, block, grid,args);                                     \
			}                                                                                      \
		    }                                                                                          \
		}                                                                                              \
	    }                                                                                                  \
	}                                                                                                      \
    }                                                                                                          \
}
#define __global__
#else // __NVCC__
#define CUDA_KERNEL(name,args...) __global__ void name(args)
#define CUDA_METHOD_HD_I __device__ __host__

#define launch_kernels(name,grid,block,args...) { name<<<grid,block>>>(args);}

#define gpuErrChk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t p_code, const char * p_file, int p_line, bool p_abort = true)
{
   if (p_code != cudaSuccess) 
   {
      std::cerr << "GPUassert: " << cudaGetErrorString(p_code) << " @ " << p_file << ":" << p_line << std::endl ;
      if (p_abort)
	{
	  exit(p_code);
	}
   }
}
#endif // __NVCC__

#endif // _MY_CUDA_H_
// EOF
