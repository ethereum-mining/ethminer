set(TARGET_NAME evmjit)

file(GLOB SOURCES "*.cpp")
file(GLOB HEADERS "*.h")
set(INTERFACE_HEADERS interface.h)
source_group("" FILES ${HEADERS})
source_group("" FILES ${SOURCES})

if (NOT "${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
	# Disable rtti for Cache as LLVM has no rtti
	set_source_files_properties(Cache.cpp PROPERTIES COMPILE_FLAGS -fno-rtti)
endif()

find_package(Git)
if(GIT_FOUND)
	execute_process(COMMAND ${GIT_EXECUTABLE} -C ${CMAKE_CURRENT_SOURCE_DIR} describe --dirty --always
					OUTPUT_VARIABLE EVMJIT_VERSION OUTPUT_STRIP_TRAILING_WHITESPACE)
endif()
if(NOT EVMJIT_VERSION)
	set(EVMJIT_VERSION "unknown")
endif()

message("EVM JIT version: ${EVMJIT_VERSION}")

add_library(${TARGET_NAME} SHARED ${SOURCES} ${HEADERS})
set_target_properties(${TARGET_NAME} PROPERTIES VERSION ${EVMJIT_VERSION} FOLDER "libs")

include_directories(${LLVM_INCLUDE_DIRS})

target_link_libraries(${TARGET_NAME} PRIVATE ${LLVM_LIBS})

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-rtti")

install(TARGETS ${TARGET_NAME} LIBRARY DESTINATION lib ARCHIVE DESTINATION lib RUNTIME DESTINATION bin)
#install(FILES ${INTERFACE_HEADERS} DESTINATION include/${TARGET_NAME})
