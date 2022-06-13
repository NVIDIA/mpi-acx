# MPI Accelerator Extensions Prototype

This code provides a simple prototype for the proposed 
[stream and graph triggered MPI Extensions](https://github.com/mpiwg-hybrid/hybrid-issues/issues/5),
as well as
[kernel triggering for partitioned communication](https://github.com/mpiwg-hybrid/hybrid-issues/issues/4).
The prototype currently supports a hybrid MPI+CUDA programming model.

# Requirements

MPI-ACX requires CUDA 11.3 or later.

The MPI library must support the partitioned communication API introduced in
MPI 4.0. The MPI library must be initialized with support for the
`MPI_THREAD_MULTIPLE` threading model.  If communication will involve GPU
memory, the MPI library must also be CUDA aware.

# APIs Supported

```c++
/* ENQUEUED OPERATIONS: ******************************************************/

enum {
    MPIX_QUEUE_CUDA_STREAM,
    MPIX_QUEUE_CUDA_GRAPH
};

int MPIX_Isend_enqueue(const void *buf, int count, MPI_Datatype datatype, int dest,
                       int tag, MPI_Comm comm, MPIX_Request *request, int qtype, void *queue);

int MPIX_Irecv_enqueue(void *buf, int count, MPI_Datatype datatype, int source,
                       int tag, MPI_Comm comm, MPIX_Request *request, int qtype, void *queue);

int MPIX_Wait_enqueue(MPIX_Request *req, int qtype, void *queue);
int MPIX_Waitall_enqueue(int count, MPIX_Request *reqs, int qtype, void *queue);

/* PARTITIONED OPERATIONS: ***************************************************/

int MPIX_Psend_init(const void *buf, int partitions, MPI_Count count,
                    MPI_Datatype datatype, int dest, int tag, MPI_Comm comm,
                    MPI_Info info, MPIX_Request *request);

int MPIX_Precv_init(void *buf, int partitions, MPI_Count count,
                    MPI_Datatype datatype, int source, int tag, MPI_Comm comm,
                    MPI_Info info, MPIX_Request *request);

int MPIX_Prequest_create(MPIX_Request *request, MPIX_Prequest *prequest);
int MPIX_Prequest_free(MPIX_Prequest *request);

__host__ int MPIX_Pready(int partition, MPIX_Request request);
__host__ int MPIX_Parrived(MPIX_Request request, int partition, int *flag);

__device__ int MPIX_Pready(int partition, MPIX_Prequest request);
__device__ int MPIX_Parrived(MPIX_Prequest request, int partition, int *flag);

/* HELPER FUNCTIONS: *********************************************************/

int MPIX_Init(void);
int MPIX_Finalize(void);

int MPIX_Start(MPIX_Request *request);
int MPIX_Startall(int count, MPIX_Request *request);

int MPIX_Wait(MPIX_Request *req, MPI_Status *status);
int MPIX_Waitall(int count, MPIX_Request *reqs, MPI_Status *statuses);

int MPIX_Request_free(MPIX_Request *request);

```

For a complete listing, refer to `include/mpi-acx.h`.

# Building and Testing

The following optional Make variables can be set:

* NVCC - NVIDIA compiler command (e.g. nvcc)
* MPI_HOME - MPI library location (e.g. /usr)
* NVCC_GENCODE - NVCC Gencode options (e.g. -gencode=arch=compute_80,code=sm_80)

Building MPI-ACX:

```sh
$ make
```

Results in the static `libmpi-acx.a` library, which can be used together with
`include/mpi-acx.h`.

Similarly, tests can be compiled:

```sh
$ cd test
$ make
```

And executed by launching them as MPI programs, e.g.

```sh
$ mpiexec -np 2 src/ring
```

# How does it work?

This prototype spawns a thread on the CPU to proxy communication requests from
the GPU. The proxy thread directly calls the desired MPI function on behalf of
the GPU. The proxy maintains a set of flags in host pinned memory that are used
to coordinate with the GPU.  Flags are
assigned to operations when the operation is set up. A send or receive on a
stream uses one flag; however, a partitioned operation uses one flag per
partition. On-stream operations interact with flags either through CUDA stream
memory operations or CUDA kernels. In-graph operations interact with flags
through CUDA kernels.

# Relation to the MPI Standard

MPI-ACX is a prototype of APIs that are proposed for inclusion in the MPI
standard. Users should expect that if these APIs are integrated in MPI, the
`MPIX_Request` type would be replaced by the standard `MPI_Request` type and
that any functionality in the helper functions (e.g. `MPIX_Init` or
`MPIX_Start`) would be merged into these existing MPI functions (e.g.
`MPI_Init` or `MPI_Start`). That is, the existence of "MPIX" variants of
existing MPI objects and APIs is a consequence of prototyping and not a
proposed addition to the MPI standard.

# Environment variables

`MPIACX_NFLAGS` - Set the number of flags that the proxy thread should
allocate (default of 4096).

`MPIACX_DISABLE_MEMOPS` - Disable the use of CUDA stream memory operations
(memOps). MemOps must be enabled when the `nvidia` kernel module is loaded by
setting `NVreg_EnableStreamMemOPs=1`
([more details](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEMOP.html)).
The code automatically detects when memOp support is present, so there is no
need to set this environment variable if memOp support is not available.

# Caveats

It's possible to create a deadlock situation when using blocking operations
(e.g. wait on stream/graph or polling on Pready in a kernel) while
communication on GPU buffers is pending. This can happen when the MPI library
submits work to the CUDA layer to perform communication (e.g. cudaMemcpy);
however, this work is not able to make forward progress because of the blocking
stream/graph operation or blocked kernel. This can be avoided by ensuring
that the MPI library can use GPUDirect RDMA to access data in GPU buffers,
by requesting that the MPI library use GDRCopy for copy operations, or by
using host memory for communication buffers.

It's also possible to create a deadlock when the same kernel performs
`MPIX_Pready` operations and also polls on on an `MPIX_Parrived` operation. The
MPI standard doesn't require the `MPIX_Parried` operation to return `true`
until all partitions have been marked as ready. When the specified grid is
larger than can execute concurrently, polling threads can prevent pending
threads from executing and marking partitions as ready. This can be avoided by
using a cooperative kernel launch that limits grid size, or by marking ready
and waiting on partitions in separate kernels.

# Known Limitations

Below is a listing of current limitations:

* MPI-ACX supports a limited set of MPI operations: send/recv and psend/precv

* On-stream operations cannot be waited on from within a graph.

* In-graph operations must be completed by a wait operation performed within a graph.

* Partitioned operation start and wait operations currently must be performed on the CPU.
