cmake_minimum_required (VERSION 2.8.11)
project (hanabi_learning_environment_example)

set(CMAKE_C_FLAGS "-O2 -std=c++11 -fPIC")
set(CMAKE_CXX_FLAGS "-O2 -std=c++11 -fPIC")

add_subdirectory (../hanabi_learning_environment/hanabi_lib hanabi_lib)

add_executable (game_example game_example.cc)
target_link_libraries (game_example LINK_PUBLIC hanabi)