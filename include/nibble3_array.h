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
#ifndef _NIBBLE3_ARRAY_H_
#define _NIBBLE3_ARRAY_H_

#include "my_cuda.h"
#include <cinttypes>
#include <cassert>
#include <limits>

/**
   Class nibbles composed of 3 bits
 */
class nibble3_array
{
  public:
    typedef uint32_t t_value;
    typedef uint64_t t_storage;
    CUDA_METHOD_HD_I nibble3_array(void);
    CUDA_METHOD_HD_I void set_nibble3(unsigned int p_index, t_value p_value);
    CUDA_METHOD_HD_I t_value get_nibble3(unsigned int p_index) const;
    CUDA_METHOD_HD_I t_storage collect(unsigned int p_nibble) const;
  private:
    t_storage m_nibble_0;
    t_storage m_nibble_1;
    t_storage m_nibble_2;
};

//------------------------------------------------------------------------------
nibble3_array::nibble3_array(void)
: m_nibble_0(0)
, m_nibble_1(0)
, m_nibble_2(0)
{
}

//------------------------------------------------------------------------------
void nibble3_array::set_nibble3( unsigned int p_index
                               , t_value p_value
                               )
{
    assert(p_index < 8 * sizeof(t_storage));
    t_storage l_bit_mask = ((t_storage)0x1) << p_index;
    m_nibble_0 = (m_nibble_0 & (~l_bit_mask)) | ((p_value & 0x1) ? l_bit_mask : 0);
    m_nibble_1 = (m_nibble_1 & (~l_bit_mask)) | ((p_value & 0x2) ? l_bit_mask : 0);
    m_nibble_2 = (m_nibble_2 & (~l_bit_mask)) | ((p_value & 0x4) ? l_bit_mask : 0);
}

//------------------------------------------------------------------------------
nibble3_array::t_value nibble3_array::get_nibble3(unsigned int p_index) const
{
    assert(p_index < 8 * sizeof(t_storage));
    t_storage l_bit_mask = ((t_storage)0x1) << p_index;
    return ((!!(m_nibble_2 & l_bit_mask)) << 2) |  ((!!(m_nibble_1 & l_bit_mask)) << 1) | (!!(m_nibble_0 & l_bit_mask));
}

//------------------------------------------------------------------------------
nibble3_array::t_storage nibble3_array::collect(unsigned int p_nibble) const
{
    return ((p_nibble & 0x1) ? m_nibble_0 : (~m_nibble_0)) & ((p_nibble & 0x2) ? m_nibble_1 : (~m_nibble_1)) & ((p_nibble & 0x4) ? m_nibble_2 : (~m_nibble_2));
}

#endif // _NIBBLE3_ARRAY_H_
// EOF
