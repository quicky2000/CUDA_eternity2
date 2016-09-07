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

#ifndef _BORDER_ENUMERATION_H_
#define _BORDER_ENUMERATION_H_

#include <iostream>
#include <cassert>

class symbol
{
  friend std::ostream & operator<<(std::ostream & p_stream, const symbol & p_symbol);
public:
  inline symbol(unsigned int p_index,
		unsigned int p_number);
  inline unsigned int get_index(void) const;
  inline unsigned int get_number(void) const;
  inline unsigned int decr(void);
  inline unsigned int incr(void);
private:
  unsigned int m_index;
  unsigned int m_number;
  unsigned int m_max_number;
};

std::ostream & operator<<(std::ostream & p_stream, const symbol & p_symbol)
{
  p_stream << (char)('A' - 1 + p_symbol.m_index) ;
  return p_stream;
}

//------------------------------------------------------------------------------
symbol::symbol(unsigned int p_index,
	       unsigned int p_number):
  m_index(p_index),
  m_number(p_number),
  m_max_number(p_number)
{
}

//------------------------------------------------------------------------------
unsigned int symbol::get_index(void) const
{
  return m_index;
}

//------------------------------------------------------------------------------
unsigned int symbol::get_number(void) const
{
  return m_number;
}

//------------------------------------------------------------------------------
unsigned int symbol::decr(void)
{
  assert(m_number);
  --m_number;
}

//------------------------------------------------------------------------------
unsigned int symbol::incr(void)
{
  assert(m_number < m_max_number);
  ++m_number;
}

//------------------------------------------------------------------------------
void enumerate(void)
{
  symbol l_symbol_table[3] = {symbol(1,1),symbol(2,2),symbol(3,3)};
  unsigned int l_nb_symbol = 3; 
  unsigned int l_word[6];
  for(unsigned int l_index = 0 ; l_index < 6 ; ++l_index)
    {
      l_word[l_index] = 0;
    }
  unsigned int l_index = 0;
  unsigned int l_count = 0;
  bool l_end = false;
  while(!l_end)
    {
      if(0 == l_word[l_index])
	{
	  unsigned int l_symbol_index = 0;
	  while(0 == l_symbol_table[l_symbol_index].get_number())
	    {
	      ++l_symbol_index;
	    }
	  l_word[l_index] = l_symbol_table[l_symbol_index].get_index();
	  l_symbol_table[l_symbol_index].decr();
	  ++l_index;
	}

      if(6 == l_index)
	{
	  std::cout << l_count << "\t" ;
	  for(unsigned int l_display_index = 0 ; l_display_index < 6; ++l_display_index)
	    {
	      std::cout << (char)('A' - 1 + l_word[l_display_index]);
	    }
	  std::cout << std::endl ;
	  ++l_count;
	  l_index = 5;
	  //	  l_symbol_table[l_word[l_index] - 1].incr();
	  //	  l_word[l_index] = 0;
	  //	  l_index = 4;
	  bool l_reverse_end = false;
	  while(!l_reverse_end)
	    {
	      if(l_nb_symbol == l_word[l_index])
		{
		  l_symbol_table[l_word[l_index] - 1].incr();
		  l_word[l_index] = 0;
		  if(l_index)
		    {
		      --l_index;
		    }
		  else
		    {
		      l_end = true;
		      l_reverse_end = true;
		    }
		}
	      else
		{
		  unsigned int l_old = l_word[l_index];
		  l_symbol_table[l_old - 1].incr();
		  l_word[l_index] = 0;
		  for(unsigned int l_candidate = l_old + 1; l_candidate <= l_nb_symbol && ! l_reverse_end; ++ l_candidate)
		    {
		      if(l_symbol_table[l_candidate - 1].get_number())
			{
			  l_word[l_index] = l_candidate;
			  l_symbol_table[l_candidate - 1].decr();
			  ++l_index;
			  l_reverse_end = true;
			}
		    }
		  if(!l_reverse_end)
		    {
		      --l_index;
		    }
		}
	    }
	}
    }
}

#endif // _BORDER_ENUMERATION_H_
