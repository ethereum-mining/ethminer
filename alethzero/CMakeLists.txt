cmake_policy(SET CMP0015 NEW)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
aux_source_directory(. SRC_LIST)

include_directories(..)
link_directories(../libethereum)

find_package(Qt5Widgets REQUIRED)

add_executable(alethzero Main.ui ${SRC_LIST})
qt5_wrap_ui(ui_Main.h Main.ui)

qt5_use_modules(alethzero Widgets Network )
target_link_libraries(alethzero ethereum)

install( TARGETS alethzero RUNTIME DESTINATION bin )
