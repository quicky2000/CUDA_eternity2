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

#ifndef _PIECE_H_
#define _PIECE_H_
#include <cinttypes>

class piece
{
public:
  inline piece(void);

  inline void set_color(uint8_t p_north_color,
			unsigned int p_orientation);

  CUDA_METHOD_HD_I uint8_t get_color(unsigned int p_side) const;

  CUDA_METHOD_HD_I uint8_t get_color(unsigned int p_side, unsigned int p_orientation) const;
private:
  uint8_t m_colors[4];
};

//------------------------------------------------------------------------------
piece::piece(void):
  m_colors{0,0,0,0}
{
}

//------------------------------------------------------------------------------
void piece::set_color(uint8_t p_color,
		      unsigned int p_orientation)
{
  m_colors[p_orientation] = p_color;
}

//------------------------------------------------------------------------------
uint8_t piece::get_color(unsigned int p_side) const
{
  assert(p_side <= 3);
  return m_colors[p_side];
}

//------------------------------------------------------------------------------
uint8_t piece::get_color(unsigned int p_side, unsigned int p_orientation) const
{
  return get_color((p_side + p_orientation) & 0x3);
}

#endif // _PIECE_H_
// EOF
