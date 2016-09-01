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
#include <iostream>
#include <cinttypes>
#include <cstring>
#include <sstream>
#include <cassert>
#include <stdexcept>

#include "my_cuda.h"
#include "constraint.h"
#include "situation_orientation.h"
#include "situation.h"
#include "piece.h"

typedef enum class orientation {NORTH=0,EAST,SOUTH,WEST} t_orientation;



CUDA_KERNEL(kernel, const piece * const p_pieces, constraint (*p_constraints)[18][4], situation * p_initial_situation, int * p_initial_index)
{
  constraint (&l_constraints)[18][4] = *p_constraints;
  situation l_situation(p_initial_situation[threadIdx.x]);
  constraint l_available_pieces[196];
  int l_max_index = *p_initial_index - 1;
  situation l_max_situation(l_situation);
  uint32_t l_nb_iteration = UINT32_MAX - 10;
  
  // Main loop
  int l_index = *p_initial_index;
  while(l_index < 239 && l_nb_iteration)
    {
      unsigned int l_north_index = l_index - 16;
      unsigned int l_east_index = l_index + 1;
      unsigned int l_south_index = l_index + 16;
      unsigned int l_west_index = l_index - 1;

      // get neighbour piece ids
      uint8_t l_north_piece_id = l_situation.m_piece_ids[l_north_index];
      uint8_t l_east_piece_id = l_situation.m_piece_ids[l_east_index];
      uint8_t l_south_piece_id = l_situation.m_piece_ids[l_south_index];
      uint8_t l_west_piece_id = l_situation.m_piece_ids[l_west_index];

      // get neighbour orientations
      uint32_t l_north_piece_orientation = l_situation.m_orientations.get_orientation(l_north_index);
      uint32_t l_east_piece_orientation = l_situation.m_orientations.get_orientation(l_east_index);
      uint32_t l_south_piece_orientation = l_situation.m_orientations.get_orientation(l_south_index);
      uint32_t l_west_piece_orientation = l_situation.m_orientations.get_orientation(l_west_index);

      // Get colours defining the constraint
      uint8_t l_north_color = p_pieces[l_north_piece_id].get_color((unsigned int)t_orientation::SOUTH,l_north_piece_orientation);
      uint8_t l_east_color = p_pieces[l_east_piece_id].get_color((unsigned int)t_orientation::WEST,l_east_piece_orientation);
      uint8_t l_south_color = p_pieces[l_south_piece_id].get_color((unsigned int)t_orientation::NORTH,l_south_piece_orientation);
      uint8_t l_west_color = p_pieces[l_west_piece_id].get_color((unsigned int)t_orientation::EAST, l_west_piece_orientation);

      // Compute constraint for each orientation
      constraint l_north_constraint(l_constraints[l_north_color][(unsigned int)t_orientation::NORTH],
				    l_constraints[l_east_color][(unsigned int)t_orientation::EAST],
				    l_constraints[l_south_color][(unsigned int)t_orientation::SOUTH],
				    l_constraints[l_west_color][(unsigned int)t_orientation::WEST],
				    l_available_pieces[l_index - 17]
				    );
      constraint l_east_constraint(l_constraints[l_north_color][(unsigned int)t_orientation::EAST],
				   l_constraints[l_east_color][(unsigned int)t_orientation::SOUTH],
				   l_constraints[l_south_color][(unsigned int)t_orientation::WEST],
				   l_constraints[l_west_color][(unsigned int)t_orientation::NORTH],
				   l_available_pieces[l_index - 17]
				   );
      constraint l_south_constraint(l_constraints[l_north_color][(unsigned int)t_orientation::SOUTH],
				    l_constraints[l_east_color][(unsigned int)t_orientation::WEST],
				    l_constraints[l_south_color][(unsigned int)t_orientation::NORTH],
				    l_constraints[l_west_color][(unsigned int)t_orientation::EAST],
				    l_available_pieces[l_index - 17]
				    );
      constraint l_west_constraint(l_constraints[l_north_color][(unsigned int)t_orientation::WEST],
				   l_constraints[l_east_color][(unsigned int)t_orientation::NORTH],
				   l_constraints[l_south_color][(unsigned int)t_orientation::EAST],
				   l_constraints[l_west_color][(unsigned int)t_orientation::SOUTH],
				   l_available_pieces[l_index - 17]
				   );

      
      int l_ffs[4] = {l_north_constraint.ffs(), l_east_constraint.ffs(), l_south_constraint.ffs(), l_west_constraint.ffs()};
      int l_ffs_1_0 = l_ffs[0] ? l_ffs[0] : l_ffs[1];
      unsigned int l_orientation_1_0 = l_ffs[0] ? 0 : 1;
      int l_ffs_1_1 = l_ffs[2] ? l_ffs[2] : l_ffs[3];
      unsigned int l_orientation_1_1 = l_ffs[2] ? 2 : 3;
      int l_ffs_result = l_ffs_1_0 ? l_ffs_1_0 : l_ffs_1_1;
      unsigned int l_orientation_result = l_ffs_1_0 ? l_orientation_1_0 : l_orientation_1_1;

      // Assign piece
      l_situation.m_piece_ids[l_index] = l_ffs_result;
      l_situation.m_orientations.set_orientation(l_index, l_orientation_result);
      l_available_pieces[l_index - 17].toggle_bit(l_situation.m_piece_ids[l_index] - (l_ffs_result ? 1 : 0),l_ffs_result ? true : false);
      int l_backward_increment = ((l_index - 1) & 0xF) ? -1 : -3;
      int l_forward_increment = ((l_index + 2) & 0xF) ? 1 : 3;
      l_available_pieces[l_index - 17 + l_forward_increment] = l_available_pieces[l_index - 17];
      l_index += l_ffs_result ? l_forward_increment : l_backward_increment;
      if(l_index > l_max_index)
	{
	  l_max_situation = l_situation;
	  l_max_index = l_index;
	}
      ++l_nb_iteration;
    }
  p_initial_situation[threadIdx.x] = l_max_situation;

}

//------------------------------------------------------------------------------
int main(void)
{
  test_constraint();
  test_orientation();

  // Binary representation of pieces
  piece l_pieces[197];

  uint32_t l_center_id_to_piece_id[196];

  // Binary representation of constraints by colors
  // Color 0 represent no pieces
  constraint l_constraints[18][4];
  for(unsigned int l_index = 0; l_index < 4 ; ++l_index)
    {
      l_constraints[0][l_index].fill(true);
    }

  unsigned int l_color_id_to_center_color_id[23] = 
    {
      256,
      256,
      256,
      256,
      256,
      256,
      256,
      256,
      256,
      256,
      256,
      256,
      256,
      256,
      256,
      256,
      256,
      256,
      256,
      256,
      256,
      256,
      256
    };

  unsigned int l_center_color_id_to_color_id[18] = 
    {
      128,
      128,
      128,
      128,
      128,
      128,
      128,
      128,
      128,
      128,
      128,
      128,
      128,
      128,
      128,
      128,
      128,
      128
    };

  // Compute pieces bitfield representation
  unsigned int l_center_piece_index = 1;

  unsigned int l_center_color_index = 1;

#include "eternity2_pieces.h"

  for(unsigned int l_all_pieces_index = 0; l_all_pieces_index < 256; ++l_all_pieces_index)
    {
      if(0 != l_all_pieces[l_all_pieces_index][1 + (unsigned int)t_orientation::NORTH] &&
	 0 != l_all_pieces[l_all_pieces_index][1 + (unsigned int)t_orientation::EAST] &&
	 0 != l_all_pieces[l_all_pieces_index][1 + (unsigned int)t_orientation::SOUTH] &&
	 0 != l_all_pieces[l_all_pieces_index][1 + (unsigned int)t_orientation::WEST]
	 )
	{
	  for(unsigned int l_orientation_index = (unsigned int)t_orientation::NORTH; l_orientation_index <= (unsigned int)t_orientation::WEST; ++l_orientation_index)
	    {
	      unsigned int l_color_id = l_all_pieces[l_all_pieces_index][1 + l_orientation_index];

	      // Compute center_color_id if not already done
	      if(256 == l_color_id_to_center_color_id[l_color_id])
		{
		  l_color_id_to_center_color_id[l_color_id] = l_center_color_index;
		  l_center_color_id_to_color_id[l_center_color_index] = l_color_id;
		  ++l_center_color_index;
		}
	      unsigned int l_center_color_id = l_color_id_to_center_color_id[l_color_id];
	      // Store bitfield representation
	      l_pieces[l_center_piece_index].set_color(l_center_color_id, l_orientation_index);

	      // Record this color in constraint table. We consider l_center_piece_index - 1 because piece 0 mean no piece
	      l_constraints[l_center_color_id][l_orientation_index].set_bit(l_center_piece_index - 1);
	    }

	  // Keep memory of global piece id
	  l_center_id_to_piece_id[l_center_piece_index] = l_all_pieces[l_all_pieces_index][0];

	  ++l_center_piece_index;
	}
    }

  int l_initial_index = 17;

  situation l_initial_situation;

  l_initial_situation.m_piece_ids[1] = 1;
  l_initial_situation.m_piece_ids[2] = 77;
  l_initial_situation.m_piece_ids[18] = 1;
  l_initial_situation.m_piece_ids[16] = 1;
  l_initial_situation.m_piece_ids[33] = 1;
  l_initial_situation.m_orientations.set_orientation(1,(unsigned int)t_orientation::SOUTH);
  l_initial_situation.m_orientations.set_orientation(18,(unsigned int)t_orientation::SOUTH);
  l_initial_situation.m_orientations.set_orientation(16,(unsigned int)t_orientation::SOUTH);
  l_initial_situation.m_orientations.set_orientation(33,(unsigned int)t_orientation::SOUTH);

  piece * l_pieces_ptr = nullptr;
  constraint (*l_constraints_ptr)[18][4] = nullptr;
  situation * l_initial_situation_ptr = nullptr;
  int * l_initial_index_ptr = nullptr;

  gpuErrChk(cudaMalloc(&l_pieces_ptr, 197 * sizeof(piece)));
  gpuErrChk(cudaMalloc(&l_constraints_ptr, 18 * 4 * sizeof(constraint)));
  gpuErrChk(cudaMalloc(&l_initial_situation_ptr, sizeof(situation)));
  gpuErrChk(cudaMalloc(&l_initial_index_ptr, sizeof(int)));
  
  gpuErrChk(cudaMemcpy(l_pieces_ptr, &l_pieces[0], 197 * sizeof(piece), cudaMemcpyHostToDevice));
  gpuErrChk(cudaMemcpy(l_constraints_ptr, &l_constraints[0][0], 18 * 4 * sizeof(constraint), cudaMemcpyHostToDevice));
  gpuErrChk(cudaMemcpy(l_initial_situation_ptr, &l_initial_situation, sizeof(situation), cudaMemcpyHostToDevice));
  gpuErrChk(cudaMemcpy(l_initial_index_ptr, &l_initial_index, sizeof(int), cudaMemcpyHostToDevice));
  
  dim3 dimBlock(1,1);
  dim3 dimGrid(1,1);
  launch_kernels(kernel, dimGrid, dimBlock, l_pieces_ptr,
		 l_constraints_ptr,
		 l_initial_situation_ptr,
		 l_initial_index_ptr
		 );

  gpuErrChk(cudaMemcpy(&l_initial_situation, l_initial_situation_ptr, sizeof(situation), cudaMemcpyDeviceToHost));
  gpuErrChk(cudaFree(l_pieces_ptr));
  gpuErrChk(cudaFree(l_constraints_ptr));
  gpuErrChk(cudaFree(l_initial_situation_ptr));
  gpuErrChk(cudaFree(l_initial_index_ptr));
//#endif

  return 0;
}

// EOF
