#
# @BEGIN LICENSE
#
# mp2f12 by Psi4 Developer, a plugin to:
#
# Psi4: an open-source quantum chemistry software package
#
# Copyright (c) 2007-2022 The Psi4 Developers.
#
# The copyrights for code used from other parties are included in
# the corresponding files.
#
# This file is part of Psi4.
#
# Psi4 is free software; you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, version 3.
#
# Psi4 is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License along
# with Psi4; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
#
# @END LICENSE
#

cmake_minimum_required(VERSION 3.20 FATAL_ERROR)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(mp2f12 CXX)
set(mp2f12_AUTHORS      "Erica C. Mitchell")
set(mp2f12_Description  "Explicitly Correlated MP2 plugin to Psi4")
set(mp2f12_URL  "github.com/EricaCMitchell/mp2f12")


find_package(psi4 1.8 REQUIRED)

find_package(OpenMP)
if (OpenMP_FOUND)
    message(STATUS "OpenMP Version ${OpenMP_CXX_VERSION_MAJOR}.${OpenMP_CXX_VERSION_MINOR}")
endif()

add_psi4_plugin(mp2f12 
    plugin.cc
    mp2f12.cc
    integrals.cc
    f12_intermediates.cc
)

add_subdirectory(_deps)

target_link_libraries(mp2f12
    PUBLIC
        einsums
)
