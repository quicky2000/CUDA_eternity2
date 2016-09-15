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

#include "border_pieces.h"
#include "border_color_constraint.h"
#include "border_backtracker.h"
#include <map>

int launch_cuda_code(piece (&p_pieces)[197], constraint (&p_constraints)[18][4]);

int main(void)
{
  enumerate();

  // Binary representation of border_pieces
  border_pieces l_border_pieces;

  // Border pieces constraint summary. Index 0 correspond to no pieces
  border_color_constraint l_border_constraints[23];
  l_border_constraints[0].fill(true);

  // Binary representation of center pieces. Index 0 is for empty piece
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

  // Count number of occurences for each border2center color
  std::map<unsigned int, unsigned int> l_B2C_color_count;

  // Compute pieces bitfield representation
  unsigned int l_center_piece_index = 1;
  unsigned int l_center_color_index = 1;

  unsigned int l_border_edges[60];
#include "eternity2_pieces.h"

  for(unsigned int l_all_pieces_index = 0; l_all_pieces_index < 256; ++l_all_pieces_index)
    {
      unsigned int l_border_edge_count = 0;
      for(unsigned int l_border_index = 1 ; l_border_index < 5; ++l_border_index)
	{
	  l_border_edge_count += 0 == l_all_pieces[l_all_pieces_index][l_border_index];
	}
      unsigned int l_piece_id = l_all_pieces[l_all_pieces_index][0];
      switch(l_border_edge_count)
	{
	case 0:
	  {
	    assert(60 < l_piece_id);
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
	    l_center_id_to_piece_id[l_center_piece_index] = l_piece_id;

	    ++l_center_piece_index;
	  }
	  break;
	case 1:
	  {
	    assert(l_piece_id <= 60 && l_piece_id > 4);
	    // Search for border edge
	    unsigned int l_border_edge_index = 0;
	    while(l_all_pieces[l_all_pieces_index][1 + l_border_edge_index])
	      {
		++l_border_edge_index;
	      }
	    l_border_edges[l_piece_id - 1] = l_border_edge_index;
	    unsigned int l_center_color = l_all_pieces[l_all_pieces_index][1 + ( 2 + l_border_edge_index) % 4];
	    l_border_pieces.set_colors(l_piece_id - 1,
				       l_all_pieces[l_all_pieces_index][1 + ( 3 + l_border_edge_index) % 4],
				       l_center_color,
				       l_all_pieces[l_all_pieces_index][1 + ( 1 + l_border_edge_index) % 4]
				       );
	    std::map<unsigned int, unsigned int>::iterator l_iter = l_B2C_color_count.find(l_center_color);
	    if(l_B2C_color_count.end() != l_iter)
	      {
		++(l_iter->second);
	      }
	    else
	      {
		l_B2C_color_count.insert(std::map<unsigned int, unsigned int>::value_type(l_center_color,1));
	      }
	  }
	  break;
	case 2:
	  {
	    assert(l_piece_id < 5);
	    // Search for border edge
	    unsigned int l_border_edge_index = 0;
	    while(l_all_pieces[l_all_pieces_index][1 + l_border_edge_index] || l_all_pieces[l_all_pieces_index][1 + ((1 + l_border_edge_index) % 4)])
	      {
		++l_border_edge_index;
	      }
	    l_border_edges[l_piece_id - 1] = l_border_edge_index;
	    l_border_pieces.set_colors(l_piece_id - 1,
				       l_all_pieces[l_all_pieces_index][1 + ( 3 + l_border_edge_index) % 4],
				       0, // Virtual center piece in case of corner
				       l_all_pieces[l_all_pieces_index][1 + ( 2 + l_border_edge_index) % 4]
				       );
	  }
	  break;
	default:
	  // Throw an exception here
	  break;
	}
      if(l_border_edge_count)
	{
	  l_border_constraints[l_border_pieces.get_left(l_piece_id - 1)].set_bit(l_piece_id - 1);
	  l_border_constraints[l_border_pieces.get_center(l_piece_id - 1)].set_bit(l_piece_id - 1);
	  l_border_constraints[l_border_pieces.get_right(l_piece_id - 1)].set_bit(l_piece_id - 1);
	}
    }

  launch_border_bactracker(l_border_pieces, l_border_constraints, l_border_edges, l_B2C_color_count);
  return launch_cuda_code(l_pieces, l_constraints);
}
// EOF