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
#ifndef _CONSTRAINT_H_
#define _CONSTRAINT_H_

#include <cinttypes>
#include <cstring>
#include "my_cuda.h"
#include <sstream>
#include <stdexcept>

class constraint
{
public:

  CUDA_METHOD_HD_I constraint(bool p_init=false);
  CUDA_METHOD_HD_I constraint(const constraint & p_constraint);

  CUDA_METHOD_HD_I void operator=(const constraint & p_constraint);

  CUDA_METHOD_HD_I constraint(const constraint & p_c1,
			      const constraint & p_c2,
			      const constraint & p_c3,
			      const constraint & p_c4,
			      const constraint & p_c5
			      );
  CUDA_METHOD_HD_I void toggle_bit(uint32_t p_index, bool p_value);

  inline void fill(bool p_init);
  inline void set_bit(uint32_t p_index);
  inline void unset_bit(uint32_t p_index);

  inline bool get_bit(uint32_t p_index) const;

  CUDA_METHOD_HD_I int ffs(void) const;
private:
  uint32_t m_words[7];
};

inline void test_constraint(void);

//------------------------------------------------------------------------------
constraint::constraint(const constraint & p_constraint)
{
  memcpy(&m_words[0], &p_constraint.m_words[0], 7 * sizeof(uint32_t));
}

//------------------------------------------------------------------------------
constraint::constraint(bool p_bool):
  m_words
   {
     p_bool ? 0xFFFFFFFF : 0,
     p_bool ? 0xFFFFFFFF : 0,
     p_bool ? 0xFFFFFFFF : 0,
     p_bool ? 0xFFFFFFFF : 0,
     p_bool ? 0xFFFFFFFF : 0,
     p_bool ? 0xFFFFFFFF : 0,
     p_bool ? 0xFFFFFFFF : 0
   }
{
}

//------------------------------------------------------------------------------
void constraint::operator=(const constraint & p_constraint)
{
 memcpy(&m_words[0], &p_constraint.m_words[0], 7 * sizeof(uint32_t));
}

//------------------------------------------------------------------------------
void constraint::fill(bool p_bool)
{
  memset(m_words, p_bool ? 0xFF : 0x0, 7 * sizeof(uint32_t));
}

//------------------------------------------------------------------------------
constraint::constraint(const constraint & p_c1,
		       const constraint & p_c2,
		       const constraint & p_c3,
		       const constraint & p_c4,
		       const constraint & p_c5
		       ):
  m_words{p_c1.m_words[0] & p_c2.m_words[0] & p_c3.m_words[0] & p_c4.m_words[0] & p_c5.m_words[0],
          p_c1.m_words[1] & p_c2.m_words[1] & p_c3.m_words[1] & p_c4.m_words[1] & p_c5.m_words[1],
          p_c1.m_words[2] & p_c2.m_words[2] & p_c3.m_words[2] & p_c4.m_words[2] & p_c5.m_words[2],
          p_c1.m_words[3] & p_c2.m_words[3] & p_c3.m_words[3] & p_c4.m_words[3] & p_c5.m_words[3],
          p_c1.m_words[4] & p_c2.m_words[4] & p_c3.m_words[4] & p_c4.m_words[4] & p_c5.m_words[4],
          p_c1.m_words[5] & p_c2.m_words[5] & p_c3.m_words[5] & p_c4.m_words[5] & p_c5.m_words[5],
          p_c1.m_words[6] & p_c2.m_words[6] & p_c3.m_words[6] & p_c4.m_words[6] & p_c5.m_words[6]
         }
{
}

//------------------------------------------------------------------------------
void constraint::set_bit(uint32_t p_index)
{
  m_words[ p_index >> 5] |= (1 << (p_index & 0x1F));
}

//------------------------------------------------------------------------------
void constraint::unset_bit(uint32_t p_index)
{
  m_words[ p_index >> 5] &= ~(1 << (p_index & 0x1F));
}

//------------------------------------------------------------------------------
void constraint::toggle_bit(uint32_t p_index, bool p_value)
{
  m_words[ p_index >> 5] ^= (((uint32_t)p_value) << (p_index & 0x1F));
}

//------------------------------------------------------------------------------
bool constraint::get_bit(uint32_t p_index) const
{
  return m_words[ p_index >> 5] & (1 << (p_index & 0x1F));
}

//------------------------------------------------------------------------------
int constraint::ffs(void) const
{
  int l_ffs[7];
  for(int l_index = 0; l_index < 7; ++l_index)
    {
#ifdef __CUDA_ARCH__
      l_ffs[l_index] = __ffs(m_words[l_index]) + 32 * l_index;
#else
      l_ffs[l_index] = ::ffs(m_words[l_index]) + 32 * l_index;
#endif
    }
  int l_ffs_1_0 = l_ffs[0] ? l_ffs[0] : l_ffs[1];
  int l_ffs_1_1 = l_ffs[2] ? l_ffs[2] : l_ffs[3];
  int l_ffs_1_2 = l_ffs[4] ? l_ffs[4] : l_ffs[5];

  int l_ffs_2_0 = l_ffs_1_0 ? l_ffs_1_0 : l_ffs_1_1;
  int l_ffs_2_1 = l_ffs_1_2 ? l_ffs_1_2 : l_ffs[6];
  return l_ffs_2_0 ? l_ffs_2_0 : l_ffs_2_1;
}

//------------------------------------------------------------------------------
void test_constraint(void)
{
  std::cout << "Start test_constraint" << std::endl ;
  constraint l_constraint;
  for(unsigned int l_index = 0 ; l_index < 196 ; ++l_index)
    {
      if(l_constraint.get_bit(l_index))
	{
	  std::stringstream l_stream;
	  l_stream << l_index;
	  throw std::logic_error("Bit[" + l_stream.str() + "] should be zero");
	}
    }

  for(unsigned int l_tested_index = 0; l_tested_index < 196; ++l_tested_index)
    {
      l_constraint.set_bit(l_tested_index);
      for(unsigned int l_index = 0; l_index < 196; ++l_index)
	{
	  bool l_expected_result = l_index == l_tested_index;
	  if(l_expected_result != l_constraint.get_bit(l_index))
	    {
	      std::stringstream l_stream;
	      l_stream << l_index;
	      throw std::logic_error("Bit[" + l_stream.str() + "] should be " + (l_expected_result ? "true" : "false"));
	    }
	}
      l_constraint.unset_bit(l_tested_index);
    }
  std::cout << "test_constraint OK" << std::endl ;
}


#endif // _CONSTRAINT_H_
// EOF
