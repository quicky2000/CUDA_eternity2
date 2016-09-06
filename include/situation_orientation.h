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
#ifndef _SITUATION_ORIENTATION_H_
#define _SITUATION_ORIENTATION_H_

class situation_orientation
{
public:
  CUDA_METHOD_HD_I situation_orientation(void);
  CUDA_METHOD_HD_I situation_orientation(const situation_orientation & p_orientation);

  CUDA_METHOD_HD_I void set_orientation(unsigned int p_index,
					uint32_t p_orientation);
  CUDA_METHOD_HD_I uint32_t get_orientation(unsigned int p_index) const;
  inline void set_orientation(unsigned int p_x_index,
			      unsigned int p_y_index,
			      uint32_t p_orientation);
  inline uint32_t get_orientation(unsigned int p_x_index,
				  unsigned int p_y_index) const;
private:
  uint32_t m_orientations[16];
};

inline void test_orientation(void);

//------------------------------------------------------------------------------
situation_orientation::situation_orientation(void)
{
  memset(&m_orientations[0], 0, 64);
}

//------------------------------------------------------------------------------
situation_orientation::situation_orientation(const situation_orientation & p_orientation)
{
  memcpy(&m_orientations[0], & p_orientation.m_orientations[0], 16 * sizeof(uint32_t));
}

//------------------------------------------------------------------------------
void situation_orientation::set_orientation(unsigned int p_index,
					    uint32_t p_orientation)
{
  assert(p_orientation <= 3);
  unsigned int l_shift = (p_index & 0xF) << 1;
  unsigned int l_index = p_index >> 4;
  m_orientations[l_index] &= ~(((uint32_t)0x3) << l_shift);
  m_orientations[l_index] |= (p_orientation << l_shift);
}

//------------------------------------------------------------------------------
void situation_orientation::set_orientation(unsigned int p_x_index,
					    unsigned int p_y_index,
					    uint32_t p_orientation)
{
  unsigned int l_index = (p_y_index << 4) | p_x_index;
  set_orientation(l_index, p_orientation);
}

//------------------------------------------------------------------------------
uint32_t situation_orientation::get_orientation(unsigned int p_index) const
{
  unsigned int l_shift = (p_index & 0xF) << 1;
  unsigned int l_index = p_index >> 4;
  return (m_orientations[l_index] >> l_shift) & 0x3; 
}

//------------------------------------------------------------------------------
uint32_t situation_orientation::get_orientation(unsigned int p_x_index,
						unsigned int p_y_index) const
{
  unsigned int l_index = (p_y_index << 4) | p_x_index;
  return get_orientation(l_index);
}

//------------------------------------------------------------------------------
void test_orientation(void)
{
  std::cout << "Start test_orientation" << std::endl ;
  situation_orientation l_orientation;
  for(unsigned int l_index = 0 ; l_index < 256 ; ++l_index)
    {
      if(l_orientation.get_orientation(l_index))
	{
	  std::stringstream l_stream;
	  l_stream << l_index;
	  throw std::logic_error("orientation[" + l_stream.str() + "] should be zero");
	}
    }

  for(unsigned int l_orientation_value = 0; l_orientation_value < 4; ++l_orientation_value)
    {
      std::stringstream l_value_stream;
      l_value_stream << l_orientation_value;
      for(unsigned int l_tested_index = 0; l_tested_index < 256; ++l_tested_index)
	{
	  std::stringstream l_tested_stream;
	  l_tested_stream << l_tested_index;
	  l_orientation.set_orientation(l_tested_index, l_orientation_value);
	  for(unsigned int l_index = 0; l_index < 256; ++l_index)
	    {
	      unsigned int l_expected_result = l_index == l_tested_index ? l_orientation_value : 0;
	      unsigned int l_result = l_orientation.get_orientation(l_index);
	      if(l_expected_result != l_result)
		{
		  std::stringstream l_stream;
		  l_stream << l_index;
		  std::stringstream l_expected_stream;
		  l_expected_stream << l_expected_result;
		  std::stringstream l_result_stream;
		  l_result_stream << l_result;
		  throw std::logic_error("tested[" + l_tested_stream.str() + "] = " + l_value_stream.str() + ": orientation[" + l_stream.str() + "] value (" + l_result_stream.str() + ") is not the expected value (" + l_expected_stream.str() + ")");
		}
	    }
	  l_orientation.set_orientation(l_tested_index, 0);
	}
    }
  std::cout << "test_orientation OK" << std::endl ;
}

#endif // _SITUATION_ORIENTATION_H_
// EOF
