/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef MPI_ACX_H
#define MPI_ACX_H

#include <mpi.h>
#include <cuda.h>
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef void * MPIX_Request;
typedef void * MPIX_Prequest;

#define MPIX_REQUEST_NULL  NULL
#define MPIX_PREQUEST_NULL NULL

int MPIX_Init(void);
int MPIX_Finalize(void);

/* ENQUEUED OPERATIONS: ******************************************************/

enum {
    MPIX_QUEUE_CUDA_STREAM,
    MPIX_QUEUE_CUDA_GRAPH
};

int MPIX_Isend_enqueue(const void *buf, int count, MPI_Datatype datatype, int dest,
                       int tag, MPI_Comm comm, MPIX_Request *request, int qtype, void *queue);

int MPIX_Irecv_enqueue(void *buf, int count, MPI_Datatype datatype, int source,
                       int tag, MPI_Comm comm, MPIX_Request *request, int qtype, void *queue);

int MPIX_Wait_enqueue(MPIX_Request *req, MPI_Status *status, int qtype, void *queue);
int MPIX_Waitall_enqueue(int count, MPIX_Request *reqs, MPI_Status *statuses, int qtype, void *queue);

/* PARTITIONED OPERATIONS: ***************************************************/

int MPIX_Psend_init(const void *buf, int partitions, MPI_Count count,
                    MPI_Datatype datatype, int dest, int tag, MPI_Comm comm,
                    MPI_Info info, MPIX_Request *request);

int MPIX_Precv_init(void *buf, int partitions, MPI_Count count,
                    MPI_Datatype datatype, int source, int tag, MPI_Comm comm,
                    MPI_Info info, MPIX_Request *request);

int MPIX_Prequest_create(MPIX_Request request, MPIX_Prequest *prequest);
int MPIX_Prequest_free(MPIX_Prequest *request);

/* HELPER FUNCTIONS FOR PARTITIONED OPERATIONS: ******************************/

int MPIX_Start(MPIX_Request *request);
int MPIX_Startall(int count, MPIX_Request *request);

int MPIX_Wait(MPIX_Request *req, MPI_Status *status);
int MPIX_Waitall(int count, MPIX_Request *reqs, MPI_Status *statuses);

int MPIX_Request_free(MPIX_Request *request);

#ifdef __cplusplus
}
#endif

/* DEVICE FUNCTIONS FOR PARTITIONED OPERATIONS: ******************************/

#ifdef __CUDACC__
// Request argument is declared void* to allow the same function signature for
// host and device functions. The expected type is:
//   __host__   -- MPIX_Request *request
//   __device__ -- MPIX_Prequest *request

__host__ __device__ int MPIX_Pready(int partition, void *request);
__host__ __device__ int MPIX_Parrived(void *request, int partition, int *flag);
#endif /* __CUDACC__ */

#endif
