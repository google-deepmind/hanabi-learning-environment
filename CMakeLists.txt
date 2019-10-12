cmake_minimum_required (VERSION 2.8.11)
project (hanabi_learning_environment)

set(CMAKE_C_FLAGS "-O2 -std=c++11 -fPIC")
set(CMAKE_CXX_FLAGS "-O2 -std=c++11 -fPIC")

add_subdirectory (hanabi_learning_environment/hanabi_lib)
add_subdirectory (hanabi_learning_environment)
