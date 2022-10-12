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

#ifndef MPI_ACX_INTERNAL_H
#define MPI_ACX_INTERNAL_H

#include <mutex>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(stmt)                                                          \
    do {                                                                          \
        cudaError_t result = (stmt);                                              \
        if (cudaSuccess != result) {                                              \
            fprintf(stderr, "[%s:%d] cuda failed with %s \n", __FILE__, __LINE__, \
                    cudaGetErrorString(result));                                  \
            exit(-1);                                                             \
        }                                                                         \
    } while (0)

#define CUDA_CHECK_AND_JMP(ERRVAR, LABEL, stmt)                                   \
    do {                                                                          \
        cudaError_t result = (stmt);                                              \
        if (cudaSuccess != result) {                                              \
            fprintf(stderr, "[%s:%d] cuda failed with %s \n", __FILE__, __LINE__, \
                    cudaGetErrorString(result));                                  \
            ERRVAR = 1;                                                           \
            goto LABEL;                                                           \
        }                                                                         \
    } while (0)

#define CU_CHECK(stmt)                                                            \
    do {                                                                          \
        CUresult result = (stmt);                                                 \
        if (CUDA_SUCCESS != result) {                                             \
            const char *str;                                                      \
            CUresult str_result = cuGetErrorString(result, &str);                 \
            if (str_result != CUDA_SUCCESS)                                       \
                fprintf(stderr, "[%s:%d] cu failed with unknown error %d\n",      \
                        __FILE__, __LINE__, result);                              \
            else                                                                  \
                fprintf(stderr, "[%s:%d] cu failed with %s \n",                   \
                        __FILE__, __LINE__, str);                                 \
            exit(-1);                                                             \
        }                                                                         \
    } while (0)

#define CU_CHECK_AND_JMP(ERRVAL, LABEL, stmt)                                     \
    do {                                                                          \
        CUresult result = (stmt);                                                 \
        if (CUDA_SUCCESS != result) {                                             \
            const char *str;                                                      \
            CUresult str_result = cuGetErrorString(result, &str);                 \
            if (str_result != CUDA_SUCCESS)                                       \
                fprintf(stderr, "[%s:%d] cu failed with unknown error %d\n",      \
                        __FILE__, __LINE__, result);                              \
            else                                                                  \
                fprintf(stderr, "[%s:%d] cu failed with %s \n",                   \
                        __FILE__, __LINE__, str);                                 \
            (ERRVAL) = 1;                                                         \
            goto LABEL;                                                           \
        }                                                                         \
    } while (0)

#define ERRMSG(...)                                                               \
    do {                                                                          \
        char str[1000];                                                           \
        size_t off;                                                               \
        off  = snprintf(str, sizeof(str), "%s:%d Error in %s\n",                  \
                       __FILE__, __LINE__, __func__);                             \
        off += snprintf(str+off, sizeof(str)-off, __VA_ARGS__);                   \
        fprintf(stderr, "%s", str);                                               \
    } while (0)

#define MPI_CHECK(ERRVAR, LABEL, STMT)                                            \
    do {                                                                          \
        ERRVAR = STMT;                                                            \
        if (ERRVAR != MPI_SUCCESS) {                                              \
            char str[MPI_MAX_ERROR_STRING];                                       \
            int error_err, len;                                                   \
            error_err = MPI_Error_string(ERRVAR, str, &len);                      \
            fprintf(stderr, "[%s:%d] %s\n\nFailed with: %s\n",                    \
                        __FILE__, __LINE__, #STMT,                                \
                       (error_err == MPI_SUCCESS) ? str : "Unknown error");       \
            goto LABEL;                                                           \
        }                                                                         \
    } while (0)

#define NULL_CHECK_AND_JMP(ERRVAR, LABEL, PTR)                                    \
    do {                                                                          \
        if (NULL == (PTR)) {                                                      \
            ERRMSG("Memory allocation failed for %s\n", #PTR);                    \
            ERRVAR = 1;                                                           \
            goto LABEL;                                                           \
        }                                                                         \
    } while (0)




#ifdef DEBUG
#define DEBUGMSG(...) do { fprintf(stderr, __VA_ARGS__); } while(0)
#else
#define DEBUGMSG(...)
#endif

#ifdef DEBUG
#define DEBUGMSG_D(...) do { printf(__VA_ARGS__); } while(0)
#else
#define DEBUGMSG_D(...)
#endif

#define MPIACX_NFLAGS 4096

/**
  * Below is a description of the how ops are managed for each type of operation.
  *
  * Persistent Partitioned Operations
  * ---------------------------------
  *
  * One request object is allocated and N ops are allocated (one for each of N
  * partitions). This allows the proxy to track each partition's start/ready
  * state separately. The op state transitions for each partition are shown
  * below.
  *
  * 0. Available (Psend/Precv_init  -> Reserved)
  * 1. Reserved  (MPIX_Start        -> Issued)
  * 2. Issued    (MPIX_Pready       -> Pending)
  * 3. Pending   (Proxy             -> Completed)
  * 4. Completed (MPIX_Start        -> Issued)
  *              (MPIX_Request_free -> Available)
  *
  * On-Stream Send/Recv Operations
  * ------------------------------
  *
  * One request object is allocated and one op is allocated. Op state transitions
  * are shown below.
  *
  * 0. Available (Send/Recv Enqueue (STREAM) -> Issued)
  * 1. Issued    (Stream reaches operation   -> Pending)
  * 2. Pending   (Proxy                      -> Completed)
  * 3. Completed (Stream reaches wait        -> Cleanup)
  *              (MPIX_Wait Enqueue (STREAM) -> Available) (Wait call occurs when op in completed state)
  *              (MPIX_Wait                  -> Available)
  * 4. Cleanup   (Proxy                      -> Available)
  *
  * In-Graph Send/Recv Operations
  * -----------------------------
  *
  * One request object is allocated and one op is allocated. Both stream
  * capture and graph creation have the same behavior. Note that the returned
  * graph can be used in only one graph at a time. Op state transitions are
  * shown below.
  *
  * 0. Available (Send/Recv Enqueue (GRAPH) -> Issued)
  * 1. Issued    (Graph reaches operation   -> Pending)
  * 2. Pending   (Proxy                     -> Completed)
  * 3. Completed (Graph reaches wait        -> Completed)
  *              (Graph reaches send/recv   -> Pending)
  *              (Graph is freed            -> Available)
  **/

enum MPIACX_Request_kind {
    MPIACX_REQ_BASIC,
    MPIACX_REQ_PARTITIONED
};

enum MPIACX_Op_state {
    MPIACX_OP_STATE_AVAILABLE = 0,
    MPIACX_OP_STATE_RESERVED,
    MPIACX_OP_STATE_PENDING,
    MPIACX_OP_STATE_ISSUED,
    MPIACX_OP_STATE_COMPLETED,
    MPIACX_OP_STATE_CLEANUP
};

enum MPIACX_Op_kind {
    MPIACX_OP_ISEND,
    MPIACX_OP_IRECV,
    MPIACX_OP_PSEND,
    MPIACX_OP_PRECV
};

typedef struct MPIACX_Request {
    enum MPIACX_Request_kind kind;
    union {
        struct {
            size_t           flag_idx;
        } basic;
        struct {
            MPIACX_Op_kind   op;
            int              partitions;
            size_t          *flag_idx;
            size_t          *flag_idx_d;
            size_t           times_used;
            MPI_Request      request;
        } partitioned;
    };
} MPIACX_Request;

typedef struct {
    size_t           *idx;
    volatile int     *flags;
} MPIACX_Prequest;

typedef struct {
    enum MPIACX_Op_kind     kind;
    int                     in_graph;
    union {
        struct {
            void           *buf;
            int             count;
            MPI_Datatype    datatype;
            int             peer;
            int             tag;
            MPI_Comm        comm;
            MPI_Request     request;
            MPI_Status      status_save;
            MPI_Status     *status;    // Populated by wait on stream
            MPIACX_Request *ireq;      // and copied/freed by the proxy
        } sendrecv;
        struct {
            MPI_Request     request;
            int             partition;
        } partitioned;
    };
} MPIACX_Op;

typedef struct {
    int           use_memops;
    size_t        nflags;
    volatile int *flags;
    volatile int *flags_d;
    CUdeviceptr   flags_d_ptr;
    MPIACX_Op    *op;
} MPIACX_State;

extern const char *MPIACX_Op_state_str[];

extern MPIACX_State *mpiacx_state;
extern std::mutex    mpiacx_op_completion_mutex;

void mpiacx_triggered_slot_free(size_t idx);

int mpiacx_triggered_isend_init(const void *buf, int count, MPI_Datatype datatype,
                                int dest, int tag, MPI_Comm comm, size_t *idx);

int mpiacx_triggered_irecv_init(void *buf, int count, MPI_Datatype datatype,
                                int source, int tag, MPI_Comm comm, size_t *idx);

int mpiacx_triggered_pready_init(int partition, MPI_Request request, size_t *idx);
int mpiacx_triggered_parrived_init(int partition, MPI_Request request, size_t *idx);

#endif
