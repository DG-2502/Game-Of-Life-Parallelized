cmake_minimum_required(VERSION 3.15)
project(Game_Of_Life_Parallelization C)

set(CMAKE_C_STANDARD 11)

add_subdirectory(third_party/raylib ${CMAKE_BINARY_DIR}/raylib)

add_compile_options(-fopenmp)
add_link_options(-fopenmp)

add_executable(gol_serial gol_serial.c)
add_executable(gol_parallel gol_parallel.c)

target_link_libraries(gol_serial PRIVATE raylib)
target_link_libraries(gol_parallel PRIVATE raylib)
