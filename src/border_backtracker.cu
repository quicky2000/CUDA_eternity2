/* -*- C++ -*- */
/*    This file is part of CUDA_eternity2
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
#include "my_cuda.h"
#include "border_pieces.h"
#include "border_color_constraint.h"
#include "border_constraint_generator.h"
#include "octet_array.h"
#include "nibble6_array.h"
#include "nibble3_array.h"
#include "nibble5_array.h"
#include <iostream>
#include <sstream>
#include <iomanip>
#include <string>

CUDA_KERNEL(border_backtracker_kernel,
	    const nibble3_array & p_left_info,
	    const nibble5_array & p_center_info,
	    const nibble3_array & p_right_info,
	    nibble6_array * p_initial_constraint
	    )
{
  unsigned int l_index = 0;
  nibble3_array l_left_info = p_left_info;
  nibble3_array l_right_info = p_right_info;
  nibble5_array l_center_info = p_center_info;
  border_color_constraint l_available_pieces(true);
  nibble6_array l_solution(p_initial_constraint[threadIdx.x + blockIdx.x * blockDim.x]);
  bool l_ended = false;
  unsigned int l_previous_min = 0;
  do
    {
      unsigned int l_previous_index = l_index ? l_index - 1 : 0;
      unsigned int l_piece_id = l_solution.get_nibble6(l_previous_index);
      unsigned int l_color =  l_piece_id ? l_right_info.get_nibble3(l_piece_id - 1) : 0;
      border_color_constraint l_available_transitions = l_color ? l_left_info.collect(l_color) : UINT64_MAX;
      l_available_transitions & l_center_info.collect(p_initial_constraint[threadIdx.x + blockIdx.x * blockDim.x].get_octet(l_index));
      unsigned int l_next_index = l_index < 59 ? l_index + 1 : 0;
      l_available_transitions & l_available_pieces;
      uint64_t l_previous_mask = (~(( ((uint64_t)1) << l_previous_min) - 1)) ;
      l_available_transitions & l_previous_mask;

      int l_ffs = l_available_transitions.ffs();

      // Detect the end in case we have found no solution ( index 0 and no candidate)
      // or in case we are at the end ( next_index = 0 and there is one candidate)
      l_ended = (!l_index && !l_ffs) || (!l_next_index && l_ffs);

      // Remove the piece from list of available pieces if a transition was
      // possible or restablish it to prepare come back to previous state
      unsigned int l_toggled_index = l_ffs ? l_ffs : l_solution.get_nibble6(l_previous_index);
      l_available_pieces.toggle_bit(l_toggled_index - 1,true);

      // In case of no transition we need to restore the color constraint and keep
      l_previous_min = l_ffs ? 0 : l_solution.get_nibble6(l_previous_index);
      unsigned int l_previous_piece_id = l_solution.get_nibble6(l_previous_index);
      unsigned int l_previous_center_color = l_previous_piece_id ? p_border_pieces.get_center(l_previous_piece_id - 1) : 0;
      l_solution.set_nibble6(l_ffs ? l_index : l_previous_index, l_ffs ? l_ffs : l_previous_center_color);

      // Prepare for next pieces
      l_index = l_ffs ? l_next_index : l_previous_index;
 
    }
  while(!l_ended);
  p_initial_constraint[threadIdx.x + blockIdx.x * blockDim.x] = l_solution;
}

//------------------------------------------------------------------------------
int launch_border_bactracker(unsigned int p_nb_cases,
			     unsigned int p_nb_block,
			     unsigned int p_nb_thread,
			     const std::string & p_initial_situation,
			     const border_pieces & p_border_pieces,
			     border_color_constraint  (&p_border_constraints)[23],
			     const unsigned int (&p_border_edges)[60],
			     const std::map<unsigned int, unsigned int> & p_B2C_color_count,
			     const std::map<unsigned int, unsigned int> & p_reorganised_colors
			     )
{
  gpuErrChk(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1)); // type cudaFuncCache
  unsigned int l_block_size = p_nb_thread;
  std::cout << "Nb cases : " << p_nb_cases << std::endl;
  std::cout << "Nb blocks : " << p_nb_block << std::endl;
  std::cout << "Block_size : " << l_block_size << " threads" << std::endl;
  unsigned int l_nb_constraints = l_block_size * p_nb_block;
  nibble6_array * l_initial_constraint = new nibble6_array[l_nb_constraints];

  std::string l_situation_string = p_initial_situation;

  border_constraint_generator l_generator(p_B2C_color_count);

  nibble3_array l_right_colors;
  nibble3_array l_left_colors;
  nibble5_array l_center_colors;
  for(unsigned int l_index = 0;l_index < 60; ++l_index)
    {
      unsigned int l_color = p_border_pieces.get_right(l_index);
      std::map<unsigned int, unsigned int>::const_iterator l_iter = p_reorganised_colors.find(l_color);
      assert(p_reorganised_colors.end() != l_iter);
      l_right_colors.set_nibble3(l_index,l_iter->second);

      l_color = p_border_pieces.get_left(l_index);
      l_iter = p_reorganised_colors.find(l_color);
      assert(p_reorganised_colors.end() != l_iter);
      l_left_colors.set_nibble3(l_index,l_iter->second);

      // No need to reorganise center colors because they are coded on 32 bits and border colors
      // and center colors are separated and never used together
      l_color = p_border_pieces.get_center(l_index);
      l_center_colors.set_nibble5(l_index,l_color);
    }

  // Prepare pointers for memory allocation on GPU
  nibble6_array * l_initial_constraint_ptr = nullptr;
  border_pieces * l_border_pieces_ptr = nullptr;
  border_color_constraint  (* l_border_constraints_ptr)[23] = nullptr;
  nibble3_array * l_right_colors_ptr = nullptr;
  nibble3_array * l_left_colors_ptr = nullptr;
  nibble5_array * l_center_colors_ptr = nullptr;

  // Allocate pointers on GPU
  gpuErrChk(cudaMalloc(&l_initial_constraint_ptr, l_nb_constraints * sizeof(nibble6_array)));
  gpuErrChk(cudaMalloc(&l_border_pieces_ptr, sizeof(border_pieces)));
  gpuErrChk(cudaMalloc(&l_border_constraints_ptr, 23 * sizeof(border_color_constraint)));
  gpuErrChk(cudaMalloc(&l_right_colors_ptr,sizeof(nibble3_array)));
  gpuErrChk(cudaMalloc(&l_left_colors_ptr,sizeof(nibble3_array)));
  gpuErrChk(cudaMalloc(&l_center_colors_ptr,sizeof(nibble5_array)));

  gpuErrChk(cudaMemcpy(l_border_pieces_ptr, &p_border_pieces, sizeof(border_pieces), cudaMemcpyHostToDevice));
  gpuErrChk(cudaMemcpy(l_border_constraints_ptr, &p_border_constraints[0], 23 * sizeof(border_color_constraint), cudaMemcpyHostToDevice));
  gpuErrChk(cudaMemcpy(l_right_colors_ptr, &l_right_colors, sizeof(nibble3_array), cudaMemcpyHostToDevice));
  gpuErrChk(cudaMemcpy(l_left_colors_ptr, &l_left_colors, sizeof(nibble3_array), cudaMemcpyHostToDevice));
  gpuErrChk(cudaMemcpy(l_center_colors_ptr, &l_center_colors, sizeof(nibble5_array), cudaMemcpyHostToDevice));

  bool l_found = false;
  uint64_t l_fail_counter = 0;
  unsigned int l_nb_loop = 0;
  while(!l_found && l_fail_counter < p_nb_cases)
    {
      for(unsigned int l_index = 0; l_index < l_nb_constraints; ++l_index)
	{
	  l_generator.generate(l_initial_constraint[l_index]);

	  std::map<unsigned int,unsigned int> l_check;
	  for(unsigned int l_octet = 0; l_octet < 60; ++l_octet)
	    {
#if 0
	      std::cout << std::setw(2) << l_initial_constraint[l_index].get_nibble6(l_octet) << " " ;
#endif
	      if(l_initial_constraint[l_index].get_nibble6(l_octet))
		{
		  l_check[l_initial_constraint[l_index].get_nibble6(l_octet)]++;
		}
	    }
#if 0
	  std::cout << std::endl ;
#endif
	  assert(l_check == p_B2C_color_count);

	  if("" != l_situation_string)
	    {
	      assert(256 * 4 == l_situation_string.size());
	      for(unsigned int l_situation_index = 0 ; l_situation_index < 256 ; ++l_situation_index)
		{
		  std::string l_piece_id_str = l_situation_string.substr(l_situation_index * 4,3);
		  if("---" != l_piece_id_str)
		    {
		      unsigned int l_piece_id = std::stoi(l_piece_id_str);
		      unsigned int l_constraint_index= 0;
		      bool l_meaningful = true;
		      if(l_situation_index < 16)
			{
			  l_constraint_index = l_situation_index;
			}
		      else if(15 == l_situation_index % 16)
			{
			  l_constraint_index = 15 + (l_situation_index / 16);
			}
		      else if(15 == l_situation_index / 16)
			{
			  l_constraint_index = 255 - l_situation_index + 30;
			}
		      else if(0 == l_situation_index % 16)
			{
			  l_constraint_index = 45 - (l_situation_index / 16 ) + 15;
			}
		      else
			{
			  l_meaningful = false;
			}
		      if(l_meaningful)
			{
			  l_initial_constraint[l_index].set_nibble6(l_constraint_index, p_border_pieces.get_center(l_piece_id - 1));
			}
		    }
		}
	    }
	}

      gpuErrChk(cudaMemcpy(l_initial_constraint_ptr, &l_initial_constraint[0], l_nb_constraints * sizeof(nibble6_array), cudaMemcpyHostToDevice));

      dim3 dimBlock(l_block_size,1);
      dim3 dimGrid(p_nb_block,1);
      launch_kernels(border_backtracker_kernel,
		     dimGrid,
		     dimBlock,
		     *l_left_colors_ptr,
		     *l_center_colors_ptr,
		     *l_right_colors_ptr,
		     l_initial_constraint_ptr
		     );


      gpuErrChk(cudaMemcpy(&l_initial_constraint[0], l_initial_constraint_ptr, l_nb_constraints * sizeof(nibble6_array), cudaMemcpyDeviceToHost));

      for(unsigned int l_index = 0; l_index < l_nb_constraints ; ++l_index)
	{
	  if(l_initial_constraint[l_index].get_nibble6(0))
	    {
	      std::string l_result;
	      char l_orientation2string[4] = {'N', 'E', 'S', 'W'};
	      for(unsigned int l_y = 0; l_y < 16; ++l_y)
		{
		  for(unsigned int l_x = 0; l_x < 16; ++l_x)
		    {
		      std::stringstream l_stream;
		      if(0 == l_y && 0 == l_x)
			{
			  l_stream << std::setw(3) << l_initial_constraint[l_index].get_nibble6(0) << l_orientation2string[(p_border_edges[l_initial_constraint[l_index].get_nibble6(0) - 1] + 1) % 4];
			  l_result += l_stream.str();
			}
		      else if(0 == l_y && 15 == l_x)
			{
			  l_stream << std::setw(3) << l_initial_constraint[l_index].get_nibble6(15) << l_orientation2string[p_border_edges[l_initial_constraint[l_index].get_nibble6(15) - 1]];
			  l_result += l_stream.str();
			}
		      else if(15 == l_y && 15 == l_x)
			{
			  l_stream << std::setw(3) << l_initial_constraint[l_index].get_nibble6(30) << l_orientation2string[(p_border_edges[l_initial_constraint[l_index].get_nibble6(30) - 1] + 3) % 4];
			  l_result += l_stream.str();
			}
		      else if(15 == l_y && 0 == l_x)
			{
			  l_stream << std::setw(3) << l_initial_constraint[l_index].get_nibble6(45) << l_orientation2string[(p_border_edges[l_initial_constraint[l_index].get_nibble6(45) - 1] + 2) % 4];
			  l_result += l_stream.str();
			}
		      else if(0 == l_y)
			{
			  l_stream << std::setw(3) << l_initial_constraint[l_index].get_nibble6(l_x) << l_orientation2string[p_border_edges[l_initial_constraint[l_index].get_nibble6(l_x) - 1]];
			  l_result += l_stream.str();
			}
		      else if(15 == l_x)
			{
			  l_stream << std::setw(3) << l_initial_constraint[l_index].get_nibble6(15 + l_y) << l_orientation2string[(p_border_edges[l_initial_constraint[l_index].get_nibble6(l_x) - 1] + 3) % 4];
			  l_result += l_stream.str();
			}
		      else if(15 == l_y)
			{
			  l_stream << std::setw(3) << l_initial_constraint[l_index].get_nibble6(30 - l_x + 15) << l_orientation2string[(p_border_edges[l_initial_constraint[l_index].get_nibble6(l_x) - 1] + 2) % 4];
			  l_result += l_stream.str();
			}
		      else if(0 == l_x)
			{
			  l_stream << std::setw(3) << l_initial_constraint[l_index].get_nibble6(45 - l_y + 15) << l_orientation2string[(p_border_edges[l_initial_constraint[l_index].get_nibble6(l_x) - 1] + 1) % 4];
			  l_result += l_stream.str();
			}
		      else
			{
			  l_result += "----";
			}
		    }
		  //  l_result += "\n";
		}
	      std::cout << "\"" << l_result << "\"" << std::endl ;
	      l_found = true;
	    }
	  else
	    {
	      ++l_fail_counter;
	    }
	}
      ++l_nb_loop;
    }

  // Free pointers on GPU
  gpuErrChk(cudaFree(l_initial_constraint_ptr));
  gpuErrChk(cudaFree(l_border_pieces_ptr));
  gpuErrChk(cudaFree(l_border_constraints_ptr));
  gpuErrChk(cudaFree(l_right_colors_ptr));
  gpuErrChk(cudaFree(l_left_colors_ptr));
  gpuErrChk(cudaFree(l_center_colors_ptr));

  delete[] l_initial_constraint;
  std::cout << "Nb loop : " << l_nb_loop << std::endl;
  std::cout << l_fail_counter << " fails" << std::endl;
  return EXIT_SUCCESS;
}
// EOF
