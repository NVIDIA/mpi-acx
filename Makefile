NVCC ?= nvcc
MPI_HOME ?= /usr
NVCC_GENCODE ?= -gencode=arch=compute_80,code=sm_80

MPI_ACX_PARTITIONED_SUPPORT ?= 1
MPI_ACX_MEMOPS_V2 ?= 0

NVCUFLAGS ?= -I$(MPI_HOME)/include -L$(MPI_HOME)/lib $(NVCC_GENCODE) -dc
NVCUFLAGS += -Xcompiler -Wall,-Wextra,-Wno-unused-function,-Wno-unused-parameter
NVCUFLAGS += -g
#NVCUFLAGS += -DDEBUG

SRCFILES = src/init.cpp \
           src/triggered.cpp \
           src/sendrecv.cu

ifeq ($(MPI_ACX_PARTITIONED_SUPPORT), 1)
NVCUFLAGS += -DUSE_MPI_PARTITIONED
SRCFILES += src/partitioned.cu
endif

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
