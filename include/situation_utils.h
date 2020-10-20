/*
      This file is part of CUDA_eternity2
      Copyright (C) 2020  Julien Thevenon ( julien_thevenon at yahoo.fr )

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

#ifndef CUDA_ETERNITY2_SITUATION_UTILS_H
#define CUDA_ETERNITY2_SITUATION_UTILS_H

#include <memory>

namespace CUDA_eternity2
{
    template<unsigned int SIZE>
    class situation_capability;

    void launch();

    /**
     * Launch CUDA kernels
     */
    void launch( unsigned int p_nb_transiion
               , const situation_capability<512> & p_situation
               , const std::shared_ptr< situation_capability<512>[]> & p_results
               , const std::shared_ptr<situation_capability<512>[]> & p_transitions
               );
}
#endif //CUDA_ETERNITY2_SITUATION_UTILS_H
