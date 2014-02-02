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

QMAKE_LIBDIR += ../../cpp-ethereum-build/libethereum ../../secp256k1 ../../cryptopp562
LIBS += -Wl,-rpath,../../cpp-ethereum-build/libethereum -Wl,-rpath,../../secp256k1 -Wl,-rpath,../../cryptopp562 -lethereum -lcryptopp -lminiupnpc -lsecp256k1 -lleveldb -lgmp -lboost_filesystem -lboost_system
INCLUDEPATH = ../../secp256k1/include ../../cpp-ethereum

SOURCES += main.cpp \
    MainWin.cpp

HEADERS  += \
    MainWin.h

FORMS    += Main.ui



