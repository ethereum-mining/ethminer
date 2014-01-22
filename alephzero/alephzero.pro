#-------------------------------------------------
#
# Project created by QtCreator 2014-01-22T11:47:38
#
#-------------------------------------------------

QT       += core gui widgets

TARGET = alephzero
TEMPLATE = app

QMAKE_LIBDIR += ../../cpp-ethereum-build/libethereum ../../secp256k1 ../../cryptopp562

LIBS += -lethereum -lsecp256k1 -lleveldb -lcryptopp -lgmp -lboost_system -lboost_filesystem

SOURCES += main.cpp\
        Main.cpp

HEADERS  += Main.h

FORMS    += Main.ui

INCLUDEPATH = ../../secp256k1/include ../../cryptopp562


