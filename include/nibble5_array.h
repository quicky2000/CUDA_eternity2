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
#ifndef _NIBBLE5_ARRAY_H_
#define _NIBBLE5_ARRAY_H_

#include "my_cuda.h"
#include <cinttypes>
#include <cassert>
#include <limits>

/**
   Class nibbles composed of 3 bits
 */
class nibble5_array
{
  public:
    typedef uint32_t t_value;
    typedef uint64_t t_storage;
    CUDA_METHOD_HD_I nibble5_array(void);
    CUDA_METHOD_HD_I void set_nibble5( unsigned int p_index
                                     , t_value p_value
                                     );
    CUDA_METHOD_HD_I t_value get_nibble5(unsigned int p_index) const;
    CUDA_METHOD_HD_I t_storage collect(unsigned int p_nibble) const;

  private:
    t_storage m_nibble_0;
    t_storage m_nibble_1;
    t_storage m_nibble_2;
    t_storage m_nibble_3;
    t_storage m_nibble_4;
};

//------------------------------------------------------------------------------
nibble5_array::nibble5_array(void)
: m_nibble_0(0)
, m_nibble_1(0)
, m_nibble_2(0)
, m_nibble_3(0)
, m_nibble_4(0)
{
}

//------------------------------------------------------------------------------
void nibble5_array::set_nibble5(unsigned int p_index, t_value p_value)
{
    assert(p_index < 8 * sizeof(t_storage));
    t_storage l_bit_mask = ((t_storage)0x1) << p_index;
    m_nibble_0 = (m_nibble_0 & (~l_bit_mask)) | ((p_value & 0x01) ? l_bit_mask : 0);
    m_nibble_1 = (m_nibble_1 & (~l_bit_mask)) | ((p_value & 0x02) ? l_bit_mask : 0);
    m_nibble_2 = (m_nibble_2 & (~l_bit_mask)) | ((p_value & 0x04) ? l_bit_mask : 0);
    m_nibble_3 = (m_nibble_3 & (~l_bit_mask)) | ((p_value & 0x08) ? l_bit_mask : 0);
    m_nibble_4 = (m_nibble_4 & (~l_bit_mask)) | ((p_value & 0x10) ? l_bit_mask : 0);
}

//------------------------------------------------------------------------------
nibble5_array::t_value nibble5_array::get_nibble5(unsigned int p_index) const
{
    assert(p_index < 8 * sizeof(t_storage));
    t_storage l_bit_mask = ((t_storage)0x1) << p_index;
    return ((!!(m_nibble_4 & l_bit_mask)) << 4) | ((!!(m_nibble_3 & l_bit_mask)) << 3) | ((!!(m_nibble_2 & l_bit_mask)) << 2) |  ((!!(m_nibble_1 & l_bit_mask)) << 1) | (!!(m_nibble_0 & l_bit_mask));
}

//------------------------------------------------------------------------------
nibble5_array::t_storage nibble5_array::collect(unsigned int p_nibble) const
{
    return ((p_nibble & 0x1) ? m_nibble_0 : (~m_nibble_0)) & ((p_nibble & 0x2) ? m_nibble_1 : (~m_nibble_1)) & ((p_nibble & 0x4) ? m_nibble_2 : (~m_nibble_2)) & ((p_nibble & 0x8) ? m_nibble_3 : (~m_nibble_3)) & ((p_nibble & 0x10) ? m_nibble_4 : (~m_nibble_4));
}

#endif // _NIBBLE5_ARRAY_H_
// EOF
