set(TARGET_NAME evmjit)

get_filename_component(EVMJIT_INCLUDE_DIR ../include ABSOLUTE)

set(SOURCES
	JIT.cpp				${EVMJIT_INCLUDE_DIR}/evmjit/JIT.h
	JIT-c.cpp			${EVMJIT_INCLUDE_DIR}/evmjit/JIT-c.h
	Arith256.cpp		Arith256.h
	Array.cpp			Array.h
	BasicBlock.cpp		BasicBlock.h
	Cache.cpp			Cache.h
						Common.h
	Compiler.cpp		Compiler.h
	CompilerHelper.cpp	CompilerHelper.h
	Endianness.cpp		Endianness.h
	ExecStats.cpp		ExecStats.h
	Ext.cpp				Ext.h
	GasMeter.cpp		GasMeter.h
	Instruction.cpp		Instruction.h
	Memory.cpp			Memory.h
	Optimizer.cpp		Optimizer.h
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

if(${EVMJIT_VERSION_MAJOR} EQUAL 0)
	set(EVMJIT_SOVERSION "0.${EVMJIT_VERSION_MINOR}")
else()
	set(EVMJIT_SOVERSION ${EVMJIT_VERSION_MAJOR})
endif()


string(COMPARE EQUAL "${LLVM_ENABLE_ASSERTIONS}" "ON" LLVM_DEBUG)
configure_file(BuildInfo.h.in ${CMAKE_CURRENT_BINARY_DIR}/gen/BuildInfo.gen.h)

message(STATUS "EVM JIT version: ${EVMJIT_VERSION_MAJOR}.${EVMJIT_VERSION_MINOR}.${EVMJIT_VERSION_PATCH}")

add_library(${TARGET_NAME} SHARED ${SOURCES} gen/BuildInfo.gen.h)
set_target_properties(${TARGET_NAME} PROPERTIES
						VERSION ${EVMJIT_VERSION} SOVERSION ${EVMJIT_SOVERSION}
						FOLDER "libs")

target_include_directories(${TARGET_NAME} PUBLIC ${EVMJIT_INCLUDE_DIR})
target_include_directories(${TARGET_NAME} PRIVATE ${LLVM_INCLUDE_DIRS})
target_include_directories(${TARGET_NAME} PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/gen)
target_compile_definitions(${TARGET_NAME} PRIVATE ${LLVM_DEFINITIONS})
target_link_libraries(${TARGET_NAME} PRIVATE ${LLVM_LIBS})

install(TARGETS ${TARGET_NAME} LIBRARY DESTINATION lib ARCHIVE DESTINATION lib RUNTIME DESTINATION bin)
install(DIRECTORY ${EVMJIT_INCLUDE_DIR} DESTINATION include)
