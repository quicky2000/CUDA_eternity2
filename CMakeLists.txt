#-----------------------------------------
# CMakeLists Skeleton v1.0
#-----------------------------------------
cmake_minimum_required(VERSION 3.11.4)
project(CUDA_eternity2 LANGUAGES CXX)

set(CMAKE_VERBOSE_MAKEFILE OFF)
set(CMAKE_CXX_STANDARD 17)
set(ENABLE_CUDA_CODE off)

set(MY_SOURCE_FILES
    include/situation_orientation.h
    include/constraint.h
    include/border_exploration.h
    include/border_pieces.h
    include/nibble6_array.h
    include/piece.h
    include/situation_utils.h
    include/border_backtracker.h
    include/piece_position_info.h
    include/center_color_db.h
    include/situation.h
    include/border_color_constraint.h
    include/eternity2_pieces.h
    include/eternity2_types.h
    include/situation_capability.h
    include/nibble5_array.h
    include/border_enumeration.h
    include/octet_array.h
    include/nibble3_array.h
    include/border_constraint_generator.h
    src/situation_utils.cpp
   )

if(${ENABLE_CUDA_CODE})
    set (MY_CPP_FLAGS -DENABLE_CUDA_CODE)
    set (MY_SOURCE_FILES ${MY_SOURCE_FILES}
         src/border_exploration.cu
         src/border_backtracker.cu
         src/situation_utils.cu
         src/eternity2.cu
        )
else()
    set (MY_CPP_FLAGS -pedantic)
endif(${ENABLE_CUDA_CODE})

# List header directories in project
set(MY_INCLUDE_DIRECTORIES
    ${CMAKE_CURRENT_SOURCE_DIR}/include
   )

#Create dependancy list
set(DEPENDANCY_LIST "")
LIST(APPEND DEPENDANCY_LIST "my_cuda")
LIST(APPEND DEPENDANCY_LIST "parameter_manager")
LIST(APPEND DEPENDANCY_LIST "combinatorics")

#------------------------------
#- Generic part
#------------------------------
# Search for CUDA files
foreach(SOURCE_FILE IN ITEMS ${MY_SOURCE_FILES})
    if(${SOURCE_FILE} MATCHES "(.*\).cu$")
        set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
        enable_language(CUDA)
        include(CheckLanguage)
        check_language(CUDA)
        if(NOT DEFINED CMAKE_CUDA_STANDARD)
            set(CMAKE_CUDA_STANDARD 11)
            set(CMAKE_CUDA_STANDARD_REQUIRED ON)
            set(MY_CPP_FLAGS ${MY_CPP_FLAGS} -std=c++11)
            set(CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS} "--ptxas-options=-v -arch=sm_75 -gencode=arch=compute_75,code=sm_75")
        endif()
    endif()
endforeach(SOURCE_FILE)

set(GENERATED_FILES "")

# Search for Bison files
foreach(SOURCE_FILE IN ITEMS ${MY_SOURCE_FILES})
    if(${SOURCE_FILE} MATCHES "(.*\).ypp$")
        get_filename_component(BISON_NAME ${CMAKE_MATCH_1} NAME)
        message("Bison parser detected: " ${BISON_NAME})
        set(BISON_OUTPUT ${BISON_NAME}.hpp ${BISON_NAME}.cpp)
        string(FIND "$GENERATED_FILES" location.hh GENERIC_BISON_OK)
        if(GENERIC_BISON_OK EQUAL -1)
            set(BISON_OUTPUT ${BISON_OUTPUT} location.hh position.hh stack.hh)
            message("Add generic bison files")
        endif()
        add_custom_command(OUTPUT ${BISON_OUTPUT}
                           COMMAND bison -o ${BISON_NAME}.cpp ${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE_FILE}
                           DEPENDS ${SOURCE_FILE}
                           COMMENT "Generate Bison parser for \"${SOURCE_FILE}\""
                           )
        set(GENERATED_FILES ${GENERATED_FILES} ${BISON_OUTPUT})
    endif()
endforeach(SOURCE_FILE)

#Search for flex files
foreach(SOURCE_FILE IN ITEMS ${MY_SOURCE_FILES})
    if(${SOURCE_FILE} MATCHES "(.*\).ll$")
        get_filename_component(FLEX_NAME ${CMAKE_MATCH_1} NAME)
        message("Flex lexer detected: " ${FLEX_NAME})
        set(FLEX_OUTPUT ${FLEX_NAME}.h ${FLEX_NAME}.cpp)
        add_custom_command(OUTPUT ${FLEX_OUTPUT}
                           COMMAND flex -o${FLEX_NAME}.cpp --header-file=${FLEX_NAME}.h ${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE_FILE}
                           DEPENDS ${SOURCE_FILE}
                           COMMENT "Generate Flex lexer for \"${SOURCE_FILE}\""
                           )
        set(GENERATED_FILES ${GENERATED_FILES} ${FLEX_OUTPUT})
    endif()
endforeach(SOURCE_FILE)

if(GENERATED_FILES)
    set(MY_INCLUDE_DIRECTORIES
        ${MY_INCLUDE_DIRECTORIES}
        ${CMAKE_CURRENT_BINARY_DIR}
       )
endif()

set(DEPENDANCY_OBJECTS "")
set(LINKED_LIBRARIES "")
#iterate on dependancies
foreach(DEPENDANCY_ITEM IN ITEMS ${DEPENDANCY_LIST})
    message("${PROJECT_NAME} need dependancy ${DEPENDANCY_ITEM}")
    set(IS_IMPORTED FALSE)
    #Get information from dependancies if not already done
    if(NOT TARGET ${DEPENDANCY_ITEM})
        set(SUB_DEPENDANCY_OBJECTS "")
        set(SUB_LINKED_LIBRARIES "")
        add_subdirectory($ENV{QUICKY_REPOSITORY}/${DEPENDANCY_ITEM} ${CMAKE_CURRENT_BINARY_DIR}/${DEPENDANCY_ITEM})
        get_target_property(IS_IMPORTED ${DEPENDANCY_ITEM} IMPORTED)
        if(NOT ${IS_IMPORTED})
            list(APPEND DEPENDANCY_OBJECTS "$<TARGET_OBJECTS:${DEPENDANCY_ITEM}>" ${SUB_DEPENDANCY_OBJECTS})
        else()
            list(APPEND LINKED_LIBRARIES ${DEPENDANCY_ITEM})
        endif()
        list(APPEND LINKED_LIBRARIES ${SUB_LINKED_LIBRARIES})
    endif()

    # Complete include directories with directories coming from dependancy because of transitivity issues with Object libraries
    if(NOT ${IS_IMPORTED})
        set(MY_INCLUDE_DIRECTORIES ${MY_INCLUDE_DIRECTORIES} $<TARGET_PROPERTY:${DEPENDANCY_ITEM},INCLUDE_DIRECTORIES>)
    endif()

endforeach(DEPENDANCY_ITEM)

#Prepare targets
get_directory_property(HAS_PARENT PARENT_DIRECTORY)
if(IS_DIRECTORY ${HAS_PARENT})
    message("Declare library ${PROJECT_NAME}")
    add_library(${PROJECT_NAME} OBJECT ${MY_SOURCE_FILES} ${GENERATED_FILES})
    set(SUB_DEPENDANCY_OBJECTS ${DEPENDANCY_OBJECTS} PARENT_SCOPE)
    set(SUB_LINKED_LIBRARIES ${LINKED_LIBRARIES} PARENT_SCOPE)
else()
    #set(CMAKE_VERBOSE_MAKEFILE ON)
    # string(REPLACE " " ";" DEPENDANCY_OBJECTS ${DEPENDANCY_OBJECTS})
    add_executable(${PROJECT_NAME} ${MY_SOURCE_FILES} ${DEPENDANCY_OBJECTS} ${GENERATED_FILES} src/main_${PROJECT_NAME}.cpp)
    target_compile_definitions(${PROJECT_NAME} PRIVATE -D$<UPPER_CASE:${PROJECT_NAME}>_SELF_TEST)
    message(Linked librarries ${LINKED_LIBRARIES})
    target_link_libraries(${PROJECT_NAME} ${LINKED_LIBRARIES})
    target_compile_options(${PROJECT_NAME} PUBLIC -Wall -ansi -g -O0 -DNDEBUG ${MY_CPP_FLAGS})
endif()

target_include_directories(${PROJECT_NAME} PUBLIC ${MY_INCLUDE_DIRECTORIES})

foreach(DEPENDANCY_ITEM IN ITEMS ${DEPENDANCY_LIST})
    add_dependencies(${PROJECT_NAME} ${DEPENDANCY_ITEM})
endforeach(DEPENDANCY_ITEM)

#Force use of -std=c++11 instead of -std=gnu++11
set_target_properties(${PROJECT_NAME} PROPERTIES CXX_EXTENSIONS OFF)

if(DEFINED CMAKE_CUDA_HOST_COMPILER)
    set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    function(CUDA_CONVERT_FLAGS EXISTING_TARGET)
        get_property(old_flags TARGET ${EXISTING_TARGET} PROPERTY COMPILE_OPTIONS)
        if(NOT "${old_flags}" STREQUAL "")
            string(REPLACE ";" "," CUDA_flags "${old_flags}")
            set_property(TARGET ${EXISTING_TARGET} PROPERTY COMPILE_OPTIONS
                         "$<$<BUILD_INTERFACE:$<COMPILE_LANGUAGE:CXX>>:${old_flags}>$<$<BUILD_INTERFACE:$<COMPILE_LANGUAGE:CUDA>>:-Xcompiler=${CUDA_flags}>"
                        )
        endif()
    endfunction()
    CUDA_CONVERT_FLAGS(${PROJECT_NAME})
endif(DEFINED CMAKE_CUDA_HOST_COMPILER)
#EOF
