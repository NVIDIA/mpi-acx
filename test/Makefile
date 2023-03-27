NVCC ?= nvcc
MPI_HOME ?= /usr
NVCC_GENCODE ?= -gencode=arch=compute_80,code=sm_80
EXTRA_LDFLAGS ?=

MPI_ACX_PARTITIONED_SUPPORT ?= 1

NVCUFLAGS ?= -I$(MPI_HOME)/include -L$(MPI_HOME)/lib $(NVCC_GENCODE)
NVCUFLAGS += -Xcompiler -Wall,-Wextra,-Wno-unused-function,-Wno-unused-parameter -DDEBUG -g

NVLDFLAGS += -L.. -lpthread -lcuda -lcudart -lmpi -lmpi-acx $(EXTRA_LDFLAGS)

SRCFILES = src/ring.c \
           src/ring-all.c \
           src/ring-all-device.c \
           src/ring-all-graph.c \
           src/ring-all-graph-construction.c

ifeq ($(MPI_ACX_PARTITIONED_SUPPORT), 1)
SRCFILES += src/ring-partitioned.cu
endif

BINFILES  = $(patsubst %.c, %, $(filter %.c, $(SRCFILES)))
BINFILES += $(patsubst %.cu, %, $(filter %.cu, $(SRCFILES)))

OBJFILES  = $(patsubst %.cu, %.o, $(filter %.cu, $(SRCFILES)))

.PHONY: default clean

default: $(BINFILES)

%: %.c
	$(NVCC) -I../include $(NVCUFLAGS) -o $@ $+ $(NVLDFLAGS)

%.o: %.cu
	$(NVCC) -dc -I../include $(NVCUFLAGS) -o $@ $+ $(NVLDFLAGS)

%: %.o
	$(NVCC) -I../include $(NVCUFLAGS) -o $@ $+ $(NVLDFLAGS)

clean:
	rm -f $(BINFILES)
