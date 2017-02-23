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
#include "border_exploration.h"

//-----------------------------------------------------------------------------
border_exploration::border_exploration(const std::map<unsigned int, unsigned int> & p_B2C_color_count,
                                       const std::map<unsigned int, unsigned int> & p_reorganised_colors,
                                       border_color_constraint  (&p_border_constraints)[23],
                                       const border_pieces & p_border_pieces,
                                       const std::string & p_situation_string
                                      ):
        m_enumerator(nullptr),
        m_reference_word(nullptr)
{
    // Iterate on reorganised colors to count their number
    std::vector<combinatorics::symbol> l_symbols;
    for(auto l_count_iter:p_B2C_color_count)
    {
        std::map<unsigned int, unsigned int>::const_iterator l_reorganised_iter = p_reorganised_colors.find(l_count_iter.first);
        assert(p_B2C_color_count.end() != l_reorganised_iter);
        l_symbols.push_back(combinatorics::symbol(l_reorganised_iter->second,l_count_iter.second));
    }

    for(auto l_iter:l_symbols)
    {
        std::cout << l_iter.get_index() << " : " << l_iter.get_number() << std::endl;
    }

    // Create enumerator
    m_enumerator = new combinatorics::enumerator(l_symbols);

    // Create a temporary generator to obtain the first combination
    combinatorics::enumerator l_enumerator(l_symbols);
    l_enumerator.generate();

    // Rotate the first word to create the reference one
    assert(0 == (m_enumerator->get_word_size() % 4));
    unsigned int * l_tmp_word = new unsigned int[m_enumerator->get_word_size()];
    l_enumerator.get_word(l_tmp_word, m_enumerator->get_word_size());
    m_reference_word = new unsigned int[m_enumerator->get_word_size()];
    for(unsigned int l_index = 0;
        l_index < m_enumerator->get_word_size();
        ++l_index)
    {
        m_reference_word[l_index] = l_tmp_word[(l_index + (m_enumerator->get_word_size() / 4)) % m_enumerator->get_word_size()];
        std::cout << (char)('A' - 1 + m_reference_word[l_index]) ;
    }
    std::cout << std::endl;
    // Rebuild border constraints using the reorganised colors
    m_border_constraints[0] = p_border_constraints[0];
    for(unsigned int l_index = 1;
        l_index < 23;
        ++l_index)
    {
        std::map<unsigned int, unsigned int>::const_iterator l_iter = p_reorganised_colors.find(l_index);
        assert(p_reorganised_colors.end() != l_iter);
        m_border_constraints[l_iter->second] = p_border_constraints[l_index];
    }

    // Rebuild border pieces using the reorganised colors
    for(unsigned int l_index = 0;
        l_index < 60;
        ++l_index)
    {
        uint32_t l_left_color;
        uint32_t l_center_color;
        uint32_t l_right_color;
        p_border_pieces.get_colors(l_index, l_left_color, l_center_color, l_right_color);
        std::map<unsigned int, unsigned int>::const_iterator l_iter = p_reorganised_colors.find(l_left_color);
        assert(p_reorganised_colors.end() != l_iter);
        l_left_color = l_iter->second;
        l_iter = p_reorganised_colors.find(l_center_color);
        assert(p_reorganised_colors.end() != l_iter);
        l_center_color = l_iter->second;
        l_iter = p_reorganised_colors.find(l_right_color);
        assert(p_reorganised_colors.end() != l_iter);
        l_right_color = l_iter->second;
        m_border_pieces.set_colors(l_index, l_left_color, l_center_color, l_right_color);
    }
    // Create word representing a known solution
    if("" != p_situation_string)
    {
        octet_array l_solution_example;
        extract_initial_constraint(p_situation_string,
                                   l_solution_example,
                                   m_border_pieces
                                  );
        unsigned int * l_solution_word = new unsigned int[m_enumerator->get_word_size()];
        for(unsigned int l_index = 0;
            l_index < 56;
            ++l_index)
        {
            //	  std::cout << l_solution_example.get_octet((1 + l_index + l_index / ( 56 /4 ))) << std::endl;
            l_solution_word[l_index] = l_solution_example.get_octet((1 + l_index + l_index / ( 56 /4 )));
        }
        combinatorics::enumerator l_enumerator(l_symbols);
        l_enumerator.set_word(l_solution_word,l_enumerator.get_word_size());
        std::cout << "------------------------------------------------" << std::endl;
        std::cout << "Known solution : " << std::endl;
        std::cout << "------------------------------------------------" << std::endl;
        l_enumerator.display_word();
        delete[] l_solution_word;
        dim3 dimBlock(1,1);
        dim3 dimGrid(1,1);
        launch_kernels(border_backtracker_kernel,dimGrid,dimBlock, m_border_pieces,
                       m_border_constraints,
                       &l_solution_example
                      );
        std::cout << "==> Corner = " << l_solution_example.get_octet(0) << std::endl ;
        std::cout << "Max = " << l_solution_example.get_octet(59) << std::endl ;
    }
}

//-----------------------------------------------------------------------------
void border_exploration::run(void)
{
    uint64_t l_nb_solution = 0;
    bool l_continu = true;
    octet_array l_initial_constraint;
    dim3 dimBlock(1,1);
    dim3 dimGrid(1,1);
    if(m_enumerator->get_word_size() != 56)
    {
        throw quicky_exception::quicky_logic_exception("Algorithm hardcoded for Eternity2 !", __LINE__, __FILE__);
    }
    while(l_continu && m_enumerator->generate())
    {
        l_continu = m_enumerator->compare_word(m_reference_word) < 0;
        if(l_continu)
        {
            for(unsigned int l_index = 0;
                l_index < 56;
                ++l_index
                    )
            {
                l_initial_constraint.set_octet((1 + l_index + l_index / ( 56 / 4 )),
                                               m_enumerator->get_word_item(l_index)
                                              );
            }
            launch_kernels(border_backtracker_kernel,dimGrid,dimBlock, m_border_pieces,
                           m_border_constraints,
                           &l_initial_constraint
                          );
            if(!l_initial_constraint.get_octet(0))
            {
                unsigned int l_max_index = l_initial_constraint.get_octet(59);
                //	      std::cout << "Max index in border = " << l_max_index << std::endl ;
                // Max index should never be 0 as there are no constraints on first corner
                assert(l_max_index);
                l_max_index = l_max_index - 1 - l_max_index / 15;
                //	      std::cout << "Max index in word = " << l_max_index << std::endl ;
                // We invalide l_max_index + 1 because index start at 0 so if
                // max_index is I is valid it means that range [0:I] of size I + 1
                // is not valid
                m_enumerator->invalidate_root(1 + l_max_index);
                // Reset Max
                l_initial_constraint.set_octet(59,0);
            }
            else
            {
                //	      std::cout << "==> Corner = " << l_initial_constraint.get_octet(0) << std::endl ;
                ++l_nb_solution;
                std::cout << "[" << l_nb_solution << "] : ";
                m_enumerator->display_word();
                for(unsigned int l_index = 0;
                    l_index < 60;
                    ++l_index
                        )
                {
                    l_initial_constraint.set_octet(l_index,0);
                }
            }
        }
    }
    m_enumerator->display_word();
}
//EOF
