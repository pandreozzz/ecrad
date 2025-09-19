SOURCES = ./f_src/interp_2d.F90 ./f_src/vertintp.F90

MAKE_INCLUDES = ../bin/make_intfbl.1.pl
INCLUDE_DIR = ../include

OBJECTS := $(SOURCES:.F90=.so)

all: $(OBJECTS)

FCSHAREDFLAGS = -fpic -shared
%.so: %.F90
	$(FC) $(FCFLAGS) $(FCSHAREDFLAGS) -o $@ $<

clean:
	rm -f *.o $(OBJECTS)
