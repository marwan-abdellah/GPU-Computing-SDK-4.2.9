################################################################################
#
# Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
#
# NVIDIA Corporation and its licensors retain all intellectual property and 
# proprietary rights in and to this software and related documentation. 
# Any use, reproduction, disclosure, or distribution of this software 
# and related documentation without an express license agreement from
# NVIDIA Corporation is strictly prohibited.
#
# Please refer to the applicable NVIDIA end user license agreement (EULA) 
# associated with this source code for terms and conditions that govern 
# your use of this NVIDIA software.
#
################################################################################
#
# Common build script for CUDA NPP source projects for Linux and Mac platforms
#
################################################################################

CUDA_INSTALL_PATH ?= /usr/local/cuda

ifdef cuda-install
	CUDA_INSTALL_PATH := $(cuda-install)
endif

SDK_INSTALL_PATH ?= ../..

# Detect OS the being used
OSUPPER = $(shell uname -s 2>/dev/null | tr [:lower:] [:upper:])
OSLOWER = $(shell uname -s 2>/dev/null | tr [:upper:] [:lower:])
# 'linux' is output for Linux system, 'darwin' for OS X
DARWIN = $(strip $(findstring DARWIN, $(OSUPPER)))
ifneq ($(DARWIN),)
   SNOWLEOPARD = $(strip $(findstring 10.6, $(shell egrep "<string>10\.6" /System/Library/CoreServices/SystemVersion.plist)))
   LION = $(strip $(findstring 10.7, $(shell egrep "<string>10\.7" /System/Library/CoreServices/SystemVersion.plist)))
endif

ifeq ($(dbg),1)
   BINDIR ?= ../../bin/$(OSLOWER)/debug
else
   BINDIR ?= ../../bin/$(OSLOWER)/release
endif

# Autodetect 32-bit or 64-bit OS Architecture
HP_64 = $(shell uname -m | grep 64)
OSARCH= $(shell uname -m)

ifneq ($(HP_64),)
   CUDALIBARCH := 64
   LIBARCH     := 64
   LIB_ARCH    := x86_64
else
   CUDALIBARCH :=
   LIBARCH     := 32
   LIB_ARCH    := i386
endif
ifeq ($(x86_64),1)
   CUDALIBARCH := 64
   LIBARCH     := 64
   LIB_ARCH    := x86_64
endif
ifeq ($(i386),1)
   CUDALIBARCH := 
   LIBARCH     := 32
   LIB_ARCH    := i386
endif
FREEIMAGELIBARCH := $(LIBARCH)

ifneq ($(DARWIN),)
   ifneq ($(SNOWLEOPARD),)  
      OS_VERSION   := 10_6
   else
      ifneq ($(LION),)
         OS_VERSION   := 10_6
      else
         OS_VERSION   := 10_5
      endif
   endif
endif

ifneq ($(DARWIN),)
   CUDALIBARCH :=
   ifeq ($(x86_64),1)
      CXX_ARCH_FLAGS += -arch x86_64 -m64
   else
      ifeq ($(i386),1)
         CXX_ARCH_FLAGS += -arch i386 -m32
      else
         ifneq ($(HP_64),)
            CXX_ARCH_FLAGS += -arch x86_64 -m64
         else
            CXX_ARCH_FLAGS += -arch i386 -m32
         endif
      endif
   endif
   FREEIMAGELIBARCH := _$(OS_VERSION)
endif

CXX := g++ -fPIC $(CXX_ARCH_FLAGS)
INC := -I. -I../../common/UtilNPP/ -I../../common/FreeImage/include -I../../../shared/inc -I$(CUDA_INSTALL_PATH)/include/

LIB := -L$(SDK_INSTALL_PATH)/common/lib -L$(CUDA_INSTALL_PATH)/lib$(CUDALIBARCH) -lnpp -lcudart 
LIB += -L$(SDK_INSTALL_PATH)/common/FreeImage/lib/$(OSLOWER)

ifneq ($(DARWIN),)
   LIB += -Xlinker -rpath $(SDK_INSTALL_PATH)/common/lib/ -Xlinker -rpath $(CUDA_INSTALL_PATH)/lib/
endif
