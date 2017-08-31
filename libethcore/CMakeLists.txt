set(SOURCES
	BlockHeader.h BlockHeader.cpp
	EthashAux.h EthashAux.cpp
	Exceptions.h
	Farm.h
	Miner.h Miner.cpp
	PoolClient.h PoolClient.cpp
	PoolManager.h PoolManager.cpp
)

if(ETHDBUS)
	find_package(PkgConfig)
	set( ENV{PKG_CONFIG_PATH} "$ENV{PKG_CONFIG_PATH}:/usr/lib/x86_64-linux-gnu/pkgconfig" )
	pkg_check_modules(DBUS dbus-1)
	include_directories(${DBUS_INCLUDE_DIRS})
	link_directories(${DBUS_LIBRARY_DIRS})
	target_link_libraries(ethcore ${DBUS_LIBRARIES})
	add_library(ethcore DBusInt.h)
endif()

include_directories(BEFORE ..)

add_library(ethcore ${SOURCES})
target_link_libraries(ethcore ethash devcore)

if(ETHASHCL)
	target_link_libraries(ethcore ethash-cl)
endif()
if(ETHASHCUDA)
	target_link_libraries(ethcore ethash-cuda)
endif()
