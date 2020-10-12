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
#ifndef _BORDER_BACK_TRACKER_H_
#define _BORDER_BACK_TRACKER_H_

#include <map>

class border_pieces;
class border_color_constraint;
class octet_array;

int launch_border_bactracker( unsigned int p_nb_cases
                            , unsigned int p_nb_block
                            , unsigned int p_nb_thread
                            , const std::string & p_initial_situation
                            , const border_pieces & p_border_pieces
                            , border_color_constraint  (&p_border_constraints)[23]
                            , const unsigned int (&p_border_edges)[60]
                            , const std::map<unsigned int, unsigned int> & p_B2C_color_count
                            , const std::map<unsigned int, unsigned int> & p_reorganised_colors
                            );

void extract_initial_constraint( const std::string & p_situation_string
                               , octet_array & p_initial_constraint
                               , const border_pieces & p_border_pieces
                               );

void constraint_to_string( std::string & p_result
                         , const octet_array & p_situation
                         , const unsigned int (&p_border_edges)[60]
                         );
#endif // _BORDER_BACK_TRACKER_H_
// EOF
