NVCC ?= nvcc
MPI_HOME ?= /usr
NVCC_GENCODE ?= -gencode=arch=compute_70,code=sm_70

NVCUFLAGS ?= -I$(MPI_HOME)/include -L$(MPI_HOME)/lib $(NVCC_GENCODE) -dc
NVCUFLAGS += -Xcompiler -Wall,-Wextra,-Wno-unused-function,-Wno-unused-parameter
NVCUFLAGS += -g
#NVCUFLAGS += -DDEBUG

ifeq ($(USE_MPI_PARTITIONED), 1)
NVCUFLAGS += -DUSE_MPI_PARTITIONED
endif

SRCFILES = src/init.cpp \
           src/triggered.cpp \
           src/sendrecv.cu \
           src/partitioned.cu

OBJFILES  = $(patsubst %.c, %.o, $(filter %.c, $(SRCFILES)))
OBJFILES += $(patsubst %.cpp, %.o, $(filter %.cpp, $(SRCFILES)))
OBJFILES += $(patsubst %.cu, %.o, $(filter %.cu, $(SRCFILES)))

LIBNAME = libmpi-acx.a

.PHONY: default clean

default: $(LIBNAME)

$(LIBNAME): $(OBJFILES)
	$(NVCC) -lib -o $@ $+

%.o: %.c
	$(NVCC) -Iinclude $(NVCUFLAGS) -c -o $@ $+

%.o: %.cpp
	$(NVCC) -Iinclude $(NVCUFLAGS) -c -o $@ $+

%.o: %.cu
	$(NVCC) -Iinclude $(NVCUFLAGS) -c -o $@ $+

clean:
	rm -f $(OBJFILES) $(LIBNAME)
