QT += core
QT -= gui

TARGET = npyLoader
CONFIG += console
CONFIG -= app_bundle

TEMPLATE = app

SOURCES += main.cpp \
    cnpy.cpp

HEADERS += \
    cnpy.h

INCLUDEPATH += /usr/include
LIBS += -lz
