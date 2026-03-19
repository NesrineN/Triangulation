cmake_minimum_required(VERSION 3.10)
project(TriangulationTest)

# 1. C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 2. Include Directories
# Tells the compiler to look in the current folder and the CppUnitLite folder for .h files
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/CppUnitLite)

# 3. Source Files
# We list your logic files, your test main, and the CppUnitLite implementation files
# Note: Ensure the paths to CppUnitLite files match your folder structure exactly
set(SOURCE_FILES
    main.cpp
    LinearEigen.cpp
    LinearLS.cpp
    IterativeEigen.cpp
    IterativeLS.cpp
    CppUnitLite/Test.cpp
    CppUnitLite/TestResult.cpp
    CppUnitLite/TestRegistry.cpp
    CppUnitLite/Failure.cpp
    CppUnitLite/SimpleString.cpp  
)

# 4. Define the Executable
add_executable(triangulation_runner ${SOURCE_FILES})

# 5. Link Libraries
# CppUnitLite is header/source based, so it usually doesn't need external linking
# unless your system requires specific math or threading libraries.
if(UNIX)
    target_link_libraries(triangulation_runner m)
endif()