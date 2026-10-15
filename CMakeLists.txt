cmake_minimum_required(VERSION 3.20)

project(tinyNN LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_executable(main main.cpp)

target_include_directories(main PRIVATE include)

target_compile_options(main PRIVATE -O2 -g -Wall -Wextra -Werror)
