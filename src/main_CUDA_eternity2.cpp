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

#include "quicky_exception.h"
#include "parameter_manager.h"
#include "parameter.h"
#include "border_enumeration.h"
#include "piece.h"
#include "constraint.h"
#include "eternity2_types.h"

#include "center_color_db.h"
#include "border_pieces.h"
#include "border_color_constraint.h"
#include "border_backtracker.h"
#include "border_exploration.h"
#include <map>
#include <set>

int launch_cuda_code(piece (&p_pieces)[197], constraint (&p_constraints)[18][4]);

int main(int argc,char ** argv)
{
    try
    {
        // Defining application command line parameters
        parameter_manager::parameter_manager l_param_manager("CUDA_eternity2.exe","--",1);

        parameter_manager::parameter<int> l_nb_block_param("nb_block",true,1);
        l_param_manager.add(l_nb_block_param);

        parameter_manager::parameter<int> l_block_size_param("block_size",true,128);
        l_param_manager.add(l_block_size_param);

        parameter_manager::parameter<int> l_nb_cases_param("nb_cases",true,1024 * 1024);
        l_param_manager.add(l_nb_cases_param);

        parameter_manager::parameter<std::string> l_initial_situation_param("initial_situation",true,"");
        l_param_manager.add(l_initial_situation_param);

        parameter_manager::parameter<std::string> l_feature_param("feature", false);
        l_param_manager.add(l_feature_param);

        // Treating parameters
        l_param_manager.treat_parameters(argc,argv);

        unsigned int l_nb_block = l_nb_block_param.get_value();
        unsigned int l_block_size = l_block_size_param.get_value();
        unsigned int l_nb_cases = l_nb_cases_param.get_value();
        std::string l_initial_situation = l_initial_situation_param.get_value();
        std::string l_feature = l_feature_param.get_value();

        if("enumerate" == l_feature)
        {
            enumerate();
        }
        else
        {
            // Binary representation of border_pieces
            border_pieces l_border_pieces;

            // Border pieces constraint summary. Index 0 correspond to no pieces
            border_color_constraint l_border_constraints[23];
            l_border_constraints[0].fill(true);

            // Binary representation of center pieces. Index 0 is for empty piece
            piece l_pieces[197];

            // Binary representation of constraints by colors
            // Color 0 represent no pieces
            constraint l_constraints[18][4];
            for (unsigned int l_index = 0; l_index < 4; ++l_index)
            {
                l_constraints[0][l_index].fill(true);
            }

            // Count number of occurences for each border2center color
            std::map<unsigned int, unsigned int> l_B2C_color_count;

            center_color_db l_center_color_db;

            unsigned int l_border_edges[60];
#include "eternity2_pieces.h"

            for (unsigned int l_all_pieces_index = 0; l_all_pieces_index < 256; ++l_all_pieces_index)
            {
                unsigned int l_border_edge_count = 0;
                for (unsigned int l_border_index = 1; l_border_index < 5; ++l_border_index)
                {
                    l_border_edge_count += 0 == l_all_pieces[l_all_pieces_index][l_border_index];
                }
                unsigned int l_piece_id = l_all_pieces[l_all_pieces_index][0];
                switch (l_border_edge_count)
                {
                    // Center Piece
                    case 0:
                    {
                        assert(60 < l_piece_id);
                        // Keep memory of global piece id
                        unsigned int l_center_piece_index = l_center_color_db.register_piece_id(l_piece_id);
                        for (unsigned int l_orientation_index = (unsigned int) t_orientation::NORTH; l_orientation_index <= (unsigned int) t_orientation::WEST; ++l_orientation_index)
                        {
                            unsigned int l_color_id = l_all_pieces[l_all_pieces_index][1 + l_orientation_index];

                            // Compute center_color_id
                            unsigned int l_center_color_id = l_center_color_db.register_color_id(l_color_id);

                            // Store bitfield representation
                            l_pieces[l_center_piece_index].set_color(l_center_color_id, l_orientation_index);

                            // Record this color in constraint table. We consider l_center_piece_index - 1 because piece 0 mean no piece
                            l_constraints[l_center_color_id][l_orientation_index].set_bit(l_center_piece_index - 1);
                        }
                    }
                    break;
                    // Border piece
                    case 1:
                    {
                        assert(l_piece_id <= 60 && l_piece_id > 4);
                        // Search for border edge
                        unsigned int l_border_edge_index = 0;
                        while (l_all_pieces[l_all_pieces_index][1 + l_border_edge_index])
                        {
                            ++l_border_edge_index;
                        }
                        l_border_edges[l_piece_id - 1] = l_border_edge_index;
                        unsigned int l_center_color = l_all_pieces[l_all_pieces_index][1 + (2 + l_border_edge_index) % 4];
                        l_border_pieces.set_colors( l_piece_id - 1
                                                  , l_all_pieces[l_all_pieces_index][1 + (3 + l_border_edge_index) % 4]
                                                  , l_center_color
                                                  , l_all_pieces[l_all_pieces_index][1 + (1 + l_border_edge_index) % 4]
                                                  );
                        std::map<unsigned int, unsigned int>::iterator l_iter = l_B2C_color_count.find(l_center_color);
                        if (l_B2C_color_count.end() != l_iter)
                        {
                            ++(l_iter->second);
                        } else
                        {
                            l_B2C_color_count.insert(std::map<unsigned int, unsigned int>::value_type(l_center_color, 1));
                        }
                    }
                    break;
                    // Corner piece
                    case 2:
                    {
                        assert(l_piece_id < 5);
                        // Search for border edge
                        unsigned int l_border_edge_index = 0;
                        while (l_all_pieces[l_all_pieces_index][1 + l_border_edge_index] ||
                               l_all_pieces[l_all_pieces_index][1 + ((1 + l_border_edge_index) % 4)]
                              )
                        {
                            ++l_border_edge_index;
                        }
                        l_border_edges[l_piece_id - 1] = l_border_edge_index;
                        l_border_pieces.set_colors( l_piece_id - 1
                                                  , l_all_pieces[l_all_pieces_index][1 + (3 + l_border_edge_index) % 4]
                                                  , 0 // Virtual center piece in case of corner
                                                  , l_all_pieces[l_all_pieces_index][1 + (2 + l_border_edge_index) % 4]
                                                  );
                    }
                    break;
                    default:
                        // Throw an exception here
                        break;
                }
                if (l_border_edge_count)
                {
                    l_border_constraints[l_border_pieces.get_left(l_piece_id - 1)].set_bit(l_piece_id - 1);
                    l_border_constraints[l_border_pieces.get_center(l_piece_id - 1)].set_bit(l_piece_id - 1);
                }
            }

            std::set<unsigned int> l_border_colors;
            for (unsigned int l_index = 0; l_index < 60; ++l_index)
            {
                l_border_colors.insert(l_border_pieces.get_left(l_index));
                l_border_colors.insert(l_border_pieces.get_right(l_index));
            }

            unsigned int l_unaffected_B_color = 1;
            std::map<unsigned int, unsigned int> l_reorganised_B_colors;
            for (auto l_iter: l_border_colors)
            {
                l_reorganised_B_colors.insert(std::map<unsigned int, unsigned int>::value_type(l_iter, l_unaffected_B_color));
                l_reorganised_B_colors.insert(std::map<unsigned int, unsigned int>::value_type(l_unaffected_B_color, l_iter));
                std::cout << "Reorganised border colors : " << l_iter << " <=> " << l_unaffected_B_color << std::endl;
                ++l_unaffected_B_color;
            }

            unsigned int l_unaffected_B2C_color = 1;
            std::map<unsigned int, unsigned int> l_reorganised_B2C_colors;
            std::map<unsigned int, unsigned int> l_reorganised_all_colors;
            for (auto l_iter: l_B2C_color_count)
            {
                l_reorganised_B2C_colors.insert(std::map<unsigned int, unsigned int>::value_type(l_iter.first, l_unaffected_B2C_color));
                l_reorganised_all_colors.insert(std::map<unsigned int, unsigned int>::value_type(l_iter.first,l_unaffected_B2C_color));
                std::cout << "Reorganised border2center colors : " << l_iter.first << " <=> " << l_unaffected_B2C_color << std::endl;
                ++l_unaffected_B2C_color;
            }

            for(auto l_iter: l_border_colors)
            {
                l_reorganised_all_colors.insert(std::map<unsigned int, unsigned int>::value_type(l_iter,l_unaffected_B2C_color));
                std::cout << "Reorganised all colors : " << l_iter << " <=> " << l_unaffected_B2C_color << std::endl ;
                ++l_unaffected_B2C_color;
            }

            if("border_backtracker" == l_feature)
            {
                launch_border_bactracker( l_nb_cases
                                        , l_nb_block
                                        , l_block_size
                                        , l_initial_situation
                                        , l_border_pieces
                                        , l_border_constraints
                                        , l_border_edges
                                        , l_B2C_color_count
                                        , l_reorganised_B_colors
                                        );
            }
            else if("border_exploration" == l_feature)
            {
                border_exploration l_border_explorer( l_B2C_color_count
                                                    , l_reorganised_all_colors
                                                    , l_border_constraints
                                                    , l_border_pieces
                                                    , l_initial_situation
                                                    );
                l_border_explorer.run(l_border_edges);
            }
#ifdef ACTIVATE_ETERNITY2_KERNEL
            else if("old_cuda" == l_feature)
            {
                launch_cuda_code(l_pieces, l_constraints);
            }
#endif // ACTIVATE_ETERNITY2_KERNEL
            else
            {
                throw quicky_exception::quicky_logic_exception(R"(Unknown feature ")" + l_feature + R"(")", __LINE__, __FILE__);
            }
        }
    }
    catch(quicky_exception::quicky_runtime_exception & e)
    {
        std::cout << "ERROR : " << e.what() << std::endl ;
        return(-1);
    }
    catch(quicky_exception::quicky_logic_exception & e)
    {
        std::cout << "ERROR : " << e.what() << std::endl ;
        return(-1);
    }
    catch(std::exception & e)
    {
        std::cout << "ERROR from std::exception : " << e.what() << std::endl ;
        return(-1);
    }
    return EXIT_SUCCESS;
}
// EOF
