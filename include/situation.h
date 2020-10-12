/* -*- C++ -*- */
/*    This file is part of CUDA_ternity2
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
#ifndef _SITUATION_H_
#define _SITUATION_H_

#include <cinttypes>

class situation
{
  public:
    situation(void) = default;
    CUDA_METHOD_HD_I situation(const situation & p_situation);
    CUDA_METHOD_HD_I void operator=(const situation & p_situation);

    situation_orientation m_orientations;
    uint8_t m_piece_ids[256];
};

//------------------------------------------------------------------------------
situation::situation(const situation & p_situation)
{
    memcpy(&m_orientations, &p_situation.m_orientations, sizeof(m_orientations));
    memcpy(&m_piece_ids, &p_situation.m_piece_ids, sizeof(m_piece_ids));
}

//------------------------------------------------------------------------------
void situation::operator=(const situation & p_situation)
{
    memcpy(&m_orientations, &p_situation.m_orientations, sizeof(m_orientations));
    memcpy(&m_piece_ids, &p_situation.m_piece_ids, sizeof(m_piece_ids));
}

#endif // _SITUATION_H_
// EOF
