CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
PROJECT(ArrayFire-Pi)

SET_PROPERTY(GLOBAL PROPERTY USE_FOLDERS ON)

SET(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/CMakeModules")

FIND_PACKAGE(ArrayFire REQUIRED)

FIND_PACKAGE(CUDA QUIET)
FIND_PACKAGE(OpenCL QUIET)

FIND_PACKAGE(CUDA QUIET)
IF(${CUDA_FOUND})
    IF(${ArrayFire_CUDA_FOUND})  # variable defined by FIND(ArrayFire ...)
        SET(BUILD_CUDA ON CACHE BOOL "")
    ELSE(${ArrayFire_CUDA_FOUND})
        MESSAGE(STATUS "BUILD_CUDA: Set to OFF as ArrayFire CUDA not found")
    ENDIF(${ArrayFire_CUDA_FOUND})
ELSE(${CUDA_FOUND})
    MESSAGE(STATUS "BUILD_CUDA: Set to OFF as CUDA not found")
ENDIF(${CUDA_FOUND})
OPTION(BUILD_CUDA "Build ArrayFire with a CUDA backend" OFF)

FIND_PACKAGE(OpenCL QUIET)
IF(${OpenCL_FOUND})
    IF(${ArrayFire_OpenCL_FOUND})  # variable defined by FIND(ArrayFire ...)
        SET(BUILD_OPENCL ON CACHE BOOL "")
    ELSE(${ArrayFire_OpenCL_FOUND})
        MESSAGE(STATUS "BUILD_OPENCL: Set to OFF as ArrayFire OpenCL not found")
    ENDIF(${ArrayFire_OpenCL_FOUND})
ELSE(${OpenCL_FOUND})
    MESSAGE(STATUS "BUILD_OpenCL: Set to OFF as OpenCL not found")
ENDIF(${OpenCL_FOUND})
OPTION(BUILD_OPENCL "Build ArrayFire with a OpenCL backend" OFF)

# Set a default build type if none was specified
IF(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    SET(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
    # Set the possible values of build type for cmake-gui
    SET_PROPERTY(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
                 "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
ENDIF()

INCLUDE_DIRECTORIES(
    ${ArrayFire_INCLUDE_DIRS}
    )

IF(${UNIX})
    ADD_DEFINITIONS(-Wall -std=c++11 -fvisibility=hidden)
ENDIF(${UNIX})

# OS Definitions
IF(UNIX)
    IF(APPLE)   #OSX
        ADD_DEFINITIONS(-DOS_MAC)

        SET(CMAKE_MACOSX_RPATH ON)
        SET(CMAKE_SKIP_BUILD_RPATH  FALSE)
        SET(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
        SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/${AF_INSTALL_LIB_DIR}")
        SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

        LIST(FIND CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES
             "${CMAKE_INSTALL_PREFIX}/${AF_INSTALL_LIB_DIR}" isSystemDir)

        IF("${isSystemDir}" STREQUAL "-1")
            SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/${AF_INSTALL_LIB_DIR}")
        ENDIF("${isSystemDir}" STREQUAL "-1")

    ELSE(APPLE) #Linux
        ADD_DEFINITIONS(-DOS_LNX)
    ENDIF()
ELSE(${UNIX}) #Windows
    ADD_DEFINITIONS(-DOS_WIN -DNOMINMAX)
ENDIF()

SET(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)

IF(${BUILD_CUDA})
    ADD_SUBDIRECTORY(src/cuda)
ENDIF()

IF(${BUILD_OPENCL})
    ADD_SUBDIRECTORY(src/opencl)
ENDIF()
