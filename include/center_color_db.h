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
#ifndef _CENTER_COLOR_DB_H_
#define _CENTER_COLOR_DB_H_

#define NB_COLOR_ID        23
#define NB_CENTER_COLOR_ID 18

#include <cinttypes>

class center_color_db
{
  public:
    inline center_color_db();

    inline unsigned int register_color_id(unsigned int p_color_id);
    inline unsigned int get_color_id(unsigned p_center_color_id)const;
    inline unsigned int get_center_color_id(unsigned int p_color_id)const;

    inline unsigned int register_piece_id(unsigned int p_piece_id);

  private:
    unsigned int m_center_color_index;

    unsigned int m_color_id_to_center_color_id[NB_COLOR_ID];
    unsigned int m_center_color_id_to_color_id[NB_CENTER_COLOR_ID];

    unsigned int m_center_piece_index;
    uint32_t m_center_id_to_piece_id[196];
};

//------------------------------------------------------------------------------
center_color_db::center_color_db()
: m_center_color_index(1)
, m_color_id_to_center_color_id
{ 256
, 256
, 256
, 256
, 256
, 256
, 256
, 256
, 256
, 256
, 256
, 256
, 256
, 256
, 256
, 256
, 256
, 256
, 256
, 256
, 256
, 256
, 256
}
, m_center_color_id_to_color_id
{ 128
, 128
, 128
, 128
, 128
, 128
, 128
, 128
, 128
, 128
, 128
, 128
, 128
, 128
, 128
, 128
, 128
, 128
}
, m_center_piece_index(1)
{
}

//------------------------------------------------------------------------------
unsigned int center_color_db::register_color_id(unsigned int p_color_id)
{
    // Check if computation of center_color_id was already done
    if(256 == m_color_id_to_center_color_id[p_color_id])
    {
        m_color_id_to_center_color_id[p_color_id] = m_center_color_index;
        m_center_color_id_to_color_id[m_center_color_index] = p_color_id;
        ++m_center_color_index;
    }
    return m_color_id_to_center_color_id[p_color_id];
}

//------------------------------------------------------------------------------
unsigned int center_color_db::get_color_id(unsigned p_center_color_id)const
{
    assert(p_center_color_id < NB_CENTER_COLOR_ID);
    return m_center_color_id_to_color_id[p_center_color_id];
}

//------------------------------------------------------------------------------
unsigned int center_color_db::get_center_color_id(unsigned int p_color_id)const
{
    assert(p_color_id < NB_COLOR_ID);
    return m_color_id_to_center_color_id[p_color_id];
}

//------------------------------------------------------------------------------
unsigned int center_color_db::register_piece_id(unsigned int p_piece_id)
{
    m_center_id_to_piece_id[m_center_piece_index] = p_piece_id;
    return m_center_piece_index++;
}

#undef NB_COLOR_ID
#undef NB_CENTER_COLOR_ID

#endif // _CENTER_COLOR_DB_H_
// EOF
