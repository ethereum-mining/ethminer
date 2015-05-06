set(TARGET_NAME evmjit)

set(SOURCES
    Arith256.cpp		Arith256.h
    Array.cpp			Array.h
    BasicBlock.cpp		BasicBlock.h
    Cache.cpp			Cache.h
    Common.h
    Compiler.cpp		Compiler.h
    CompilerHelper.cpp	CompilerHelper.h
    ${EVMJIT_INCLUDE_DIR}/evmjit/DataTypes.h
    Endianness.cpp		Endianness.h
    ExecStats.cpp		ExecStats.h
    ExecutionEngine.cpp	ExecutionEngine.h
    Ext.cpp				Ext.h
    GasMeter.cpp		GasMeter.h
    Instruction.cpp		Instruction.h
    interface.cpp		interface.h
    JIT.cpp             ${EVMJIT_INCLUDE_DIR}/evmjit/JIT.h
    Memory.cpp			Memory.h
    Optimizer.cpp		Optimizer.h
    Runtime.cpp			Runtime.h
    RuntimeData.h
    RuntimeManager.cpp	RuntimeManager.h
    Stack.cpp			Stack.h
    Type.cpp			Type.h
    Utils.cpp			Utils.h
)
source_group("" FILES ${SOURCES})

if(${CMAKE_CXX_COMPILER_ID} STREQUAL "MSVC")
else()
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-rtti")
endif()


set(EVMJIT_VERSION "0.0.0")
set(EVMJIT_VERSION_MAJOR 0)
set(EVMJIT_VERSION_MINOR 0)
set(EVMJIT_VERSION_PATCH 0)
set(EVMJIT_VERSION_FULL "v0.0.0-nogit")

find_package(Git)
if(GIT_FOUND)
	execute_process(COMMAND ${GIT_EXECUTABLE} -C ${CMAKE_CURRENT_SOURCE_DIR} describe --dirty --always --match v*
					OUTPUT_VARIABLE EVMJIT_VERSION_FULL OUTPUT_STRIP_TRAILING_WHITESPACE)
endif()

if(${EVMJIT_VERSION_FULL} MATCHES "^v[0-9]+\\.[0-9]+")
	string(SUBSTRING ${EVMJIT_VERSION_FULL} 1 -1 EVMJIT_VERSION_FULL) # skip "v"
	string(REPLACE "-" ";" VERSION_COMPONENTS ${EVMJIT_VERSION_FULL})
	list(LENGTH VERSION_COMPONENTS NUM_VERSION_COMPONENTS)
	list(GET VERSION_COMPONENTS 0 EVMJIT_VERSION)
	string(REPLACE "." ";" VERSION_NUMBERS ${EVMJIT_VERSION})
	list(LENGTH VERSION_NUMBERS NUM_VERSION_NUMBERS)
	list(GET VERSION_NUMBERS 0 EVMJIT_VERSION_MAJOR)
	list(GET VERSION_NUMBERS 1 EVMJIT_VERSION_MINOR)
	if(${NUM_VERSION_NUMBERS} GREATER 2)
		list(GET VERSION_NUMBERS 2 EVMJIT_VERSION_PATCH) # patch number is optional
	endif()
	if(${NUM_VERSION_COMPONENTS} GREATER 1)
		list(GET VERSION_COMPONENTS 1 VERSION_PRERELEASE_CANDIDATE)
		string(REGEX MATCH "^[a-zA-Z]+.*" EVMJIT_VERSION_PRERELEASE ${VERSION_PRERELEASE_CANDIDATE}) # prerelease starts with letter
	endif()
endif()

if(${EVMJIT_VERSION_MAJOR} EQUAL 0)
	set(EVMJIT_SOVERSION "0.${EVMJIT_VERSION_MINOR}")
else()	
	set(EVMJIT_SOVERSION ${EVMJIT_VERSION_MAJOR})
endif()


string(COMPARE EQUAL "${LLVM_ENABLE_ASSERTIONS}" "ON" LLVM_DEBUG)
configure_file(BuildInfo.h.in ${CMAKE_CURRENT_BINARY_DIR}/gen/BuildInfo.gen.h)

message(STATUS "EVM JIT version: ${EVMJIT_VERSION_MAJOR}.${EVMJIT_VERSION_MINOR}.${EVMJIT_VERSION_PATCH} ${EVMJIT_VERSION_PRERELEASE} (${EVMJIT_VERSION_FULL})")

add_library(${TARGET_NAME} SHARED ${SOURCES} gen/BuildInfo.gen.h)
set_target_properties(${TARGET_NAME} PROPERTIES 
						VERSION ${EVMJIT_VERSION} SOVERSION ${EVMJIT_SOVERSION}
						FOLDER "libs")

include_directories(${EVMJIT_INCLUDE_DIR})
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_BINARY_DIR}/gen)

target_link_libraries(${TARGET_NAME} PRIVATE ${LLVM_LIBS})

install(TARGETS ${TARGET_NAME} LIBRARY DESTINATION lib ARCHIVE DESTINATION lib RUNTIME DESTINATION bin)
