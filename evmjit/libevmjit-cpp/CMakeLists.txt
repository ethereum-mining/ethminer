set(TARGET_NAME evmjit-cpp)

# Boost
find_package(Boost REQUIRED)

set(SOURCES
    Env.cpp
    JitVM.cpp   JitVM.h
    Utils.h
)
source_group("" FILES ${SOURCES})

if(${CMAKE_CXX_COMPILER_ID} STREQUAL "MSVC")
else()
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC") # add PIC for archive
endif()

add_library(${TARGET_NAME} STATIC ${SOURCES})
set_property(TARGET ${TARGET_NAME} PROPERTY FOLDER "libs")

include_directories(../..)
include_directories(${Boost_INCLUDE_DIRS})

target_link_libraries(${TARGET_NAME} evmjit)
