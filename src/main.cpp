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

#include "border_enumeration.h"
#include "piece.h"
#include "constraint.h"
#include "eternity2_types.h"

int launch_cuda_code(piece (&p_pieces)[197], constraint (&p_constraints)[18][4]);

int main(void)
{
  enumerate();

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


  return launch_cuda_code(l_pieces, l_constraints);
}
// EOF
