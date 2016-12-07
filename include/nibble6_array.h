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
#ifndef _NIBBLE6_ARRAY_H_
#define _NIBBLE6_ARRAY_H_

#include "my_cuda.h"
#include <cinttypes>
#include <cassert>
#include <limits>

/**
   Class nibbles composed of 3 bits
 */
class nibble6_array
{
 public:
  typedef uint32_t t_value;
  typedef uint64_t t_storage;
  CUDA_METHOD_HD_I nibble6_array(void);
  CUDA_METHOD_HD_I void set_nibble6(unsigned int p_index, t_value p_value);
  CUDA_METHOD_HD_I t_value get_nibble6(unsigned int p_index) const;
  CUDA_METHOD_HD_I t_storage collect(unsigned int p_nibble) const;
 private:
  t_storage m_nibble_0;
  t_storage m_nibble_1;
  t_storage m_nibble_2;
  t_storage m_nibble_3;
  t_storage m_nibble_4;
  t_storage m_nibble_5;
};

//------------------------------------------------------------------------------
nibble6_array::nibble6_array(void):
  m_nibble_0(0),
  m_nibble_1(0),
  m_nibble_2(0),
  m_nibble_3(0),
  m_nibble_4(0),
  m_nibble_5(0)
{
}

//------------------------------------------------------------------------------
void nibble6_array::set_nibble6(unsigned int p_index, t_value p_value)
{
  assert(p_index < 8 * sizeof(t_storage));
  t_storage l_bit_mask = ((t_storage)0x1) << p_index;
  m_nibble_0 = (m_nibble_0 & (~l_bit_mask)) | ((p_value & 0x01) ? l_bit_mask : 0);
  m_nibble_1 = (m_nibble_1 & (~l_bit_mask)) | ((p_value & 0x02) ? l_bit_mask : 0);
  m_nibble_2 = (m_nibble_2 & (~l_bit_mask)) | ((p_value & 0x04) ? l_bit_mask : 0);
  m_nibble_3 = (m_nibble_3 & (~l_bit_mask)) | ((p_value & 0x08) ? l_bit_mask : 0);
  m_nibble_4 = (m_nibble_4 & (~l_bit_mask)) | ((p_value & 0x10) ? l_bit_mask : 0);
  m_nibble_5 = (m_nibble_5 & (~l_bit_mask)) | ((p_value & 0x20) ? l_bit_mask : 0);
}

//------------------------------------------------------------------------------
nibble6_array::t_value nibble6_array::get_nibble6(unsigned int p_index) const
{
  assert(p_index < 8 * sizeof(t_storage));
  t_storage l_bit_mask = ((t_storage)0x1) << p_index;
  t_value l_result = !!(m_nibble_5 & l_bit_mask);
  l_result = (l_result << 1) | !!(m_nibble_4 & l_bit_mask);
  l_result = (l_result << 1) | !!(m_nibble_3 & l_bit_mask);
  l_result = (l_result << 1) | !!(m_nibble_2 & l_bit_mask);
  l_result = (l_result << 1) | !!(m_nibble_1 & l_bit_mask);
  l_result = (l_result << 1) | !!(m_nibble_0 & l_bit_mask);
  return l_result;
}

//------------------------------------------------------------------------------
nibble6_array::t_storage nibble6_array::collect(unsigned int p_nibble) const
{
  t_storage l_result =   ((p_nibble & 0x01) ? m_nibble_0 : (~m_nibble_0));
  l_result = l_result &  ((p_nibble & 0x02) ? m_nibble_1 : (~m_nibble_1));
  l_result = l_result &  ((p_nibble & 0x04) ? m_nibble_2 : (~m_nibble_2));
  l_result = l_result &  ((p_nibble & 0x08) ? m_nibble_3 : (~m_nibble_3));
  l_result = l_result &  ((p_nibble & 0x10) ? m_nibble_4 : (~m_nibble_4));
  l_result = l_result &  ((p_nibble & 0x20) ? m_nibble_5 : (~m_nibble_5));
  return l_result;
}

#endif // _NIBBLE6_ARRAY_H_
// EOF
