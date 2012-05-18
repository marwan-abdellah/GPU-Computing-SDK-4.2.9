.PHONY: update_freeimage_files

include $(ROOTDIR)/build/config/DetectOS.mk

FREEIMAGE_FILES += include/FreeImage.h

ifeq ($(OS), win32)
ifeq ($(HOST_ARCH), i686)
FREEIMAGE_FILES += lib/FreeImage.dll lib/FreeImage.lib
else
FREEIMAGE_FILES += lib/FreeImage64.dll lib/FreeImage64.lib
endif
endif

ifeq ($(OS), Linux)
ifeq ($(HOST_ARCH), i686)
FREEIMAGE_FILES += lib/linux/libfreeimage32.a
else
FREEIMAGE_FILES += lib/linux/libfreeimage64.a
endif
endif

DARWIN = $(strip $(findstring DARWIN, $(OSUPPER)))
ifneq ($(DARWIN),)
   SNOWLEOPARD = $(strip $(findstring 10.6, $(shell egrep "<string>10\.6" /System/Library/CoreServices/SystemVersion.plist)))
endif

ifneq ($(SNOWLEOPARD),)
FREEIMAGE_FILES += lib/darwin/libfreeimage_10_6.a
else
FREEIMAGE_FILES += lib/darwin/libfreeimage_10_5.a
endif

update_freeimage_files:
	$(foreach freeimage_files,$(FREEIMAGE_FILES),$(CP) -rf $(SRC_CWD)/$(freeimage_files) $(DRIVELETTER)$(BINDIR);)

include $(ROOTDIR)/build/common.mk
