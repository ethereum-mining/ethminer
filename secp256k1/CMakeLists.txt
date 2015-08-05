cmake_policy(SET CMP0015 NEW)
set(CMAKE_AUTOMOC OFF)

set(EXECUTABLE secp256k1)
file(GLOB HEADERS "*.h") 

add_library(${EXECUTABLE} ${EXECUTABLE}.c)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DHAVE_CONFIG_H -g -O2 -W -std=c89 -pedantic -Wall -Wextra -Wcast-align -Wnested-externs -Wshadow -Wstrict-prototypes -Wno-unused-function -Wno-long-long -Wno-overlength-strings")
target_link_libraries(${EXECUTABLE} ${GMP_LIBRARIES})	

install( TARGETS ${EXECUTABLE} RUNTIME DESTINATION bin ARCHIVE DESTINATION lib LIBRARY DESTINATION lib )
install( FILES ${HEADERS} DESTINATION include/${EXECUTABLE} )
