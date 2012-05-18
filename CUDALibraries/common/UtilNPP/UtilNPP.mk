
STATIC_LIBRARY    ?= UtilNPP
PROJ_DIR          := test
GENCODE           :=

include $(ROOTDIR)/build/config/DetectOS.mk

FILES += Exceptions.cpp
FILES += Image.cpp
FILES += ImageIO.cpp
FILES += Signal.cpp

ifeq ($(OS), win32)
FILES += StopWatchWin.cpp
endif

ifeq ($(OS), Linux)
FILES += StopWatchLinux.cpp
endif

ifeq ($(OS), Darwin)
FILES += StopWatchLinux.cpp
endif

INCLUDES += ../src
INCLUDES += ../../cuda/inc
INCLUDES += ../../cuda/tools/cudart
INCLUDES += ./npp/
INCLUDES += ../../../npp/include
INCLUDES += ../SDK/common/UtilNPP/
INCLUDES += ../SDK/common/FreeImage/include
INCLUDES += ../FreeImage/include
INCLUDES_ABSPATH += $(DRIVELETTER)$(BINDIR)

INCLUDE = $(CUDACC_INCLUDES) -I$(ROOTDIR)/cuda/tools/cudart

ifeq ($(OS), Linux)
CUDACC_FLAGS += -Xcompiler -Wno-uninitialized

#
# No need to build UtilNPP on GCC 3.x
#
GCC_VERSION_MAJOR := $(shell gcc -dumpversion | cut -c1-1)
ifeq ($(GCC_VERSION_MAJOR),3)
FILES = 
STATIC_LIBRARY = 
endif

endif

#
# Darwin Warnings
#
ifeq ($(OS), Darwin)
CUDACC_FLAGS += -Xcompiler -Wno-unknown-pragmas -Xcompiler -Wno-uninitialized -Xcompiler -Wno-sign-compare -Xcompiler -Wno-unused-variable
endif


include $(ROOTDIR)/build/common.mk

