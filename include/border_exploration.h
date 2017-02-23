/*    This file is part of CUDA_eternity2
      Copyright (C) 2017  Julien Thevenon ( julien_thevenon at yahoo.fr )

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
#ifndef _BORDER_EXPLORATION_H_
#define _BORDER_EXPLORATION_H_

#include "enumerator.h"
#include "octet_array.h"
#include "border_color_constraint.h"
#include "border_backtracker.h"
#include "border_pieces.h"
#include <map>

extern CUDA_KERNEL(border_backtracker_kernel,
		   const border_pieces & p_border_pieces,
		   border_color_constraint  (&p_border_constraints)[23],
		   octet_array * p_initial_constraint
		   );

class border_exploration
{
 public:
    border_exploration(const std::map<unsigned int, unsigned int> & p_B2C_color_count,
                       const std::map<unsigned int, unsigned int> & p_reorganised_colors,
                       border_color_constraint  (&p_border_constraints)[23],
                       const border_pieces & p_border_pieces,
                       const std::string & p_situation_string
                       );
    inline ~border_exploration(void);
    void run(void);
 private:
  border_color_constraint  m_border_constraints[23];
  border_pieces m_border_pieces;
  combinatorics::enumerator * m_enumerator;
  unsigned int * m_reference_word;
};


//-----------------------------------------------------------------------------
border_exploration::~border_exploration(void)
{
  delete m_enumerator;
  delete m_reference_word;
}

#endif // _BORDER_EXPLORATION_H_
// EOF
