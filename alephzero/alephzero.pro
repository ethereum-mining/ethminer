#-------------------------------------------------
#
# Project created by QtCreator 2014-01-22T11:47:38
#
#-------------------------------------------------

QT       += core gui widgets network


TARGET = alephzero
TEMPLATE = app

CONFIG(debug, debug|release): DEFINES += ETH_DEBUG

QMAKE_CXXFLAGS += -std=c++11

INCLUDEPATH += ../../cpp-ethereum
QMAKE_LIBDIR += ../../cpp-ethereum-build/libethereum
CONFIG(debug, debug|release): LIBS += -Wl,-rpath,../../cpp-ethereum-build/libethereum
LIBS += -lethereum -lcryptoppeth -lminiupnpc -lsecp256k1 -lleveldb -lgmp -lboost_filesystem -lboost_system

SOURCES += main.cpp \
    MainWin.cpp

HEADERS  += \
    MainWin.h

FORMS    += Main.ui



