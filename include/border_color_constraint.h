/* -*- C++ -*- */
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
#ifndef _BORDER_COLOR_CONSTRAINT_H_
#define _BORDER_COLOR_CONSTRAINT_H_

#include "my_cuda.h"
#include <cinttypes>
#include <cstring>

/**
   Class representing corner and border pieces matching a color constraint
 */
class border_color_constraint
{
 public:
  CUDA_METHOD_HD_I border_color_constraint(bool p_init = false);
  CUDA_METHOD_HD_I border_color_constraint(const uint64_t & p_value);
  CUDA_METHOD_HD_I border_color_constraint(const border_color_constraint & p_constraint);
  CUDA_METHOD_HD_I void operator&(const border_color_constraint & p_constraint);
  CUDA_METHOD_HD_I void operator&(const uint64_t & p_constraint);
  CUDA_METHOD_HD_I void operator=(const border_color_constraint & p_constraint);
  CUDA_METHOD_HD_I void toggle_bit(unsigned int p_index, bool p_value);

  CUDA_METHOD_HD_I void fill(bool p_init);
  inline void set_bit(uint32_t p_index);
  inline void unset_bit(uint32_t p_index);

  inline bool get_bit(uint32_t p_index) const;

  CUDA_METHOD_HD_I int ffs(void) const;

 private:
  uint64_t m_constraint;
};

//------------------------------------------------------------------------------
border_color_constraint::border_color_constraint(bool p_init):
m_constraint(p_init ? UINT64_MAX : 0x0)
{
}

//------------------------------------------------------------------------------
border_color_constraint::border_color_constraint(const uint64_t & p_value):
  m_constraint(p_value)
{
}

//------------------------------------------------------------------------------
border_color_constraint::border_color_constraint(const border_color_constraint & p_constraint):
m_constraint(p_constraint.m_constraint)
{
}

//------------------------------------------------------------------------------
void border_color_constraint::operator=(const border_color_constraint & p_constraint)
{
  m_constraint = p_constraint.m_constraint;  
}

//------------------------------------------------------------------------------
void border_color_constraint::operator&(const border_color_constraint & p_constraint)
{
  m_constraint &= p_constraint.m_constraint;  
}

//------------------------------------------------------------------------------
void border_color_constraint::operator&(const uint64_t & p_constraint)
{
  m_constraint &= p_constraint;  
}

//------------------------------------------------------------------------------
void border_color_constraint::toggle_bit(unsigned int p_index, bool p_value)
{
  m_constraint ^= ((uint64_t)p_value) << p_index;
}

//------------------------------------------------------------------------------
void border_color_constraint::fill(bool p_init)
{
  m_constraint = p_init ?  UINT64_MAX : 0x0;
}

//------------------------------------------------------------------------------
void border_color_constraint::set_bit(uint32_t p_index)
{
  m_constraint |= ((uint64_t)0x1) << p_index;
}

//------------------------------------------------------------------------------
void border_color_constraint::unset_bit(uint32_t p_index)
{
  m_constraint &= ~(((uint64_t)0x1) << p_index);
}

//------------------------------------------------------------------------------
bool border_color_constraint::get_bit(uint32_t p_index) const
{
  return m_constraint & (((uint64_t)0x1) << p_index);
}

//------------------------------------------------------------------------------
int border_color_constraint::ffs(void) const
{
#ifdef __CUDA_ARCH__
  return __ffsll(m_constraint);
#else
  return ::ffsll(m_constraint);
#endif
}

#endif // _BORDER_COLOR_CONSTRAINT_H_
// EOF
