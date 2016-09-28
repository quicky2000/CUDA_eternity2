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
#include <iostream>
#include <sstream>
#include <iomanip>
#include <string>

CUDA_KERNEL(border_backtracker_kernel, const border_pieces & p_border_pieces, border_color_constraint  (&p_border_constraints)[23], octet_array * p_initial_constraint)
{
  unsigned int l_index = 0;
  border_color_constraint l_available_pieces[60];
  l_available_pieces[59].fill(true);
  border_color_constraint l_available_transitions[60];
  l_available_transitions[0].fill(true);
  octet_array l_solution;
  bool l_ended = false;
  do
    {
      unsigned int l_previous_index = l_index ? l_index - 1 : 59;
      unsigned int l_piece_id = l_solution.get_octet(l_previous_index);
      unsigned int l_color =  l_piece_id ? p_border_pieces.get_right(l_piece_id - 1) : 0;
      l_available_transitions[l_index] & p_border_constraints[l_color];
      l_available_transitions[l_index] & p_border_constraints[p_initial_constraint[threadIdx.x].get_octet(l_index)];
      unsigned int l_next_index = l_index < 59 ? l_index + 1 : 0;
      l_piece_id = l_solution.get_octet(l_next_index);
      l_color = l_piece_id ? p_border_pieces.get_left(l_piece_id - 1) : 0;
      l_available_transitions[l_index] & p_border_constraints[l_color];
      uint64_t l_corner_mask = (0 == l_index || 15 == l_index || 30 == l_index || 45 == l_index) ? 0xF : UINT64_MAX;
      l_available_transitions[l_index] & l_corner_mask;
      l_available_transitions[l_index] & l_available_pieces[l_previous_index];
      int l_ffs = l_available_transitions[l_index].ffs();

      // Detect the end in case we have found no solution ( index 0 and no candidate)
      // or in case we are at the end ( next_index = 0 and there is one candidate)
      l_ended = (!l_index && !l_ffs) || (!l_next_index && l_ffs);

      // Apply mask to indicate the piece we will check for
      l_available_pieces[l_index] = l_available_pieces[l_previous_index];
      l_available_transitions[l_index].toggle_bit(l_ffs - 1, l_ffs);
      l_available_pieces[l_index].toggle_bit(l_ffs - 1, l_ffs);
      l_available_transitions[l_next_index].fill(true);

      // Prepare for next pieces
      l_solution.set_octet(l_index, l_ffs);
      l_index = l_ffs ? l_next_index : l_previous_index;
 
    }
  while(!l_ended);
  p_initial_constraint[threadIdx.x] = l_solution;
}

//------------------------------------------------------------------------------
int launch_border_bactracker(const border_pieces & p_border_pieces,
			     border_color_constraint  (&p_border_constraints)[23],
			     const unsigned int (&p_border_edges)[60],
			     const std::map<unsigned int, unsigned int> & p_B2C_color_count
			     )
{
  unsigned int l_block_size = 32;
  std::cout << "Block-size = " << l_block_size << std::endl;
  octet_array * l_initial_constraint = new octet_array[l_block_size];

  border_constraint_generator l_generator(p_B2C_color_count);

  bool l_found = false;
  uint64_t l_fail_counter = 0;
  unsigned int l_nb_loop = 0;
  while(!l_found && l_fail_counter < 1024 * 1024)
    {
      for(unsigned int l_index = 0; l_index < l_block_size; ++l_index)
	{
	  l_generator.generate(l_initial_constraint[l_index]);

	  std::map<unsigned int,unsigned int> l_check;
	  for(unsigned int l_octet = 0; l_octet < 60; ++l_octet)
	    {
#if 0
	      std::cout << std::setw(2) << l_initial_constraint.get_octet(l_octet) << " " ;
#endif
	      if(l_initial_constraint[l_index].get_octet(l_octet))
		{
		  l_check[l_initial_constraint[l_index].get_octet(l_octet)]++;
		}
	    }
#if 0
	  std::cout << std::endl ;
#endif
	  assert(l_check == p_B2C_color_count);
	}

      // Prepare pointers for memory allocation on GPU
      octet_array * l_initial_constraint_ptr = nullptr;
      border_pieces * l_border_pieces_ptr = nullptr;
      border_color_constraint  (* l_border_constraints_ptr)[23] = nullptr;

      // Allocate pointers on GPU
      gpuErrChk(cudaMalloc(&l_initial_constraint_ptr, l_block_size * sizeof(octet_array)));
      gpuErrChk(cudaMalloc(&l_border_pieces_ptr, sizeof(border_pieces)));
      gpuErrChk(cudaMalloc(&l_border_constraints_ptr, 23 * sizeof(border_color_constraint)));

      gpuErrChk(cudaMemcpy(l_initial_constraint_ptr, &l_initial_constraint[0], l_block_size * sizeof(octet_array), cudaMemcpyHostToDevice));
      gpuErrChk(cudaMemcpy(l_border_pieces_ptr, &p_border_pieces, sizeof(border_pieces), cudaMemcpyHostToDevice));
      gpuErrChk(cudaMemcpy(l_border_constraints_ptr, &p_border_constraints[0], 23 * sizeof(border_color_constraint), cudaMemcpyHostToDevice));

      dim3 dimBlock(l_block_size,1);
      dim3 dimGrid(1,1);
      launch_kernels(border_backtracker_kernel, dimGrid, dimBlock, *l_border_pieces_ptr,
		     *l_border_constraints_ptr,
		     l_initial_constraint_ptr
		     );


      gpuErrChk(cudaMemcpy(&l_initial_constraint[0], l_initial_constraint_ptr, l_block_size * sizeof(octet_array), cudaMemcpyDeviceToHost));

      // Free pointers on GPU
      gpuErrChk(cudaFree(l_initial_constraint_ptr));
      gpuErrChk(cudaFree(l_border_pieces_ptr));
      gpuErrChk(cudaFree(l_border_constraints_ptr));

      for(unsigned int l_index = 0; l_index < l_block_size; ++l_index)
	{
	  if(l_initial_constraint[l_index].get_octet(0))
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
			  l_stream << std::setw(3) << l_initial_constraint[l_index].get_octet(0) << l_orientation2string[(p_border_edges[l_initial_constraint[l_index].get_octet(0) - 1] + 1) % 4];
			  l_result += l_stream.str();
			}
		      else if(0 == l_y && 15 == l_x)
			{
			  l_stream << std::setw(3) << l_initial_constraint[l_index].get_octet(15) << l_orientation2string[p_border_edges[l_initial_constraint[l_index].get_octet(15) - 1]];
			  l_result += l_stream.str();
			}
		      else if(15 == l_y && 15 == l_x)
			{
			  l_stream << std::setw(3) << l_initial_constraint[l_index].get_octet(30) << l_orientation2string[(p_border_edges[l_initial_constraint[l_index].get_octet(30) - 1] + 3) % 4];
			  l_result += l_stream.str();
			}
		      else if(15 == l_y && 0 == l_x)
			{
			  l_stream << std::setw(3) << l_initial_constraint[l_index].get_octet(45) << l_orientation2string[(p_border_edges[l_initial_constraint[l_index].get_octet(45) - 1] + 2) % 4];
			  l_result += l_stream.str();
			}
		      else if(0 == l_y)
			{
			  l_stream << std::setw(3) << l_initial_constraint[l_index].get_octet(l_x) << l_orientation2string[p_border_edges[l_initial_constraint[l_index].get_octet(l_x) - 1]];
			  l_result += l_stream.str();
			}
		      else if(15 == l_x)
			{
			  l_stream << std::setw(3) << l_initial_constraint[l_index].get_octet(15 + l_y) << l_orientation2string[(p_border_edges[l_initial_constraint[l_index].get_octet(l_x) - 1] + 3) % 4];
			  l_result += l_stream.str();
			}
		      else if(15 == l_y)
			{
			  l_stream << std::setw(3) << l_initial_constraint[l_index].get_octet(30 - l_x + 15) << l_orientation2string[(p_border_edges[l_initial_constraint[l_index].get_octet(l_x) - 1] + 2) % 4];
			  l_result += l_stream.str();
			}
		      else if(0 == l_x)
			{
			  l_stream << std::setw(3) << l_initial_constraint[l_index].get_octet(45 - l_y + 15) << l_orientation2string[(p_border_edges[l_initial_constraint[l_index].get_octet(l_x) - 1] + 1) % 4];
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
  std::cout << "Nb loop : " << l_nb_loop << std::endl;
  std::cout << l_fail_counter << " fails" << std::endl;
  return EXIT_SUCCESS;
}
// EOF
