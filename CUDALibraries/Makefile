PROJECTS := $(shell find src -name Makefile)

%.ph_build : lib/libcutil.so lib/libUtilNPP.a shared/libshrutil.so
	+@$(MAKE) -C $(dir $*) $(MAKECMDGOALS)
%.ph_build : lib/libparamgl.so
	+@$(MAKE) -C $(dir $*) $(MAKECMDGOALS)
%.ph_build : lib/librendercheckgl.so
	+@$(MAKE) -C $(dir $*) $(MAKECMDGOALS)

%.ph_clean : 
	+@$(MAKE) -C $(dir $*) clean $(USE_DEVICE)

%.ph_clobber :
	+@$(MAKE) -C $(dir $*) clobber $(USE_DEVICE)

all:  $(addsuffix .ph_build,$(PROJECTS))
	@echo "Finished building all"

shared/libshrutil.so:
	+@$(MAKE) -C ../shared/ 

lib/libcutil.so:
	+@$(MAKE) -C common

lib/libUtilNPP.a:
	+@$(MAKE) -C common/UtilNPP

lib/librendercheckgl.so:
	+@$(MAKE) -C common -f Makefile_rendercheckgl

tidy:
	@find * | egrep "#" | xargs rm -f
	@find * | egrep "\~" | xargs rm -f

clean: tidy $(addsuffix .ph_clean,$(PROJECTS))
	+@$(MAKE) -C ../shared clean
	+@$(MAKE) -C common clean
	+@$(MAKE) -C common -f Makefile_rendercheckgl clean
	+@$(MAKE) -C common/UtilNPP clean

clobber: clean $(addsuffix .ph_clobber,$(PROJECTS))
	+@$(MAKE) -C ../shared clobber
	+@$(MAKE) -C common clobber
	+@$(MAKE) -C common -f Makefile_rendercheckgl clobber
	+@$(MAKE) -C common/UtilNPP clobber

