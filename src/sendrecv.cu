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

#include <mpi-acx.h>
#include <mpi-acx-internal.h>

#ifdef USE_MEMOPS_V2
#define wrapWriteValue cuStreamWriteValue32_v2
#define wrapWaitValue  cuStreamWaitValue32_v2
#define wrapBatchMemOp cuStreamBatchMemOp_v2
#else
#define wrapWriteValue cuStreamWriteValue32
#define wrapWaitValue  cuStreamWaitValue32
#define wrapBatchMemOp cuStreamBatchMemOp
#endif

__global__ void set(volatile int *ptr, int val) {
    *ptr = val;
    DEBUGMSG_D("Set  on %p completed\n", ptr);
}

__global__ void wait(volatile int *ptr, int val) {
    DEBUGMSG_D("Wait on %p starting\n", ptr);
    while (*ptr != val)
        ;
    DEBUGMSG_D("Wait on %p completed\n", ptr);
}

__global__ void wait_and_set(volatile int *ptr, int val, int newval) {
    DEBUGMSG_D("Wait on %p starting\n", ptr);
    while (*ptr != val)
        ;
    *ptr = newval;
    DEBUGMSG_D("Wait on %p completed\n", ptr);
}

__global__ void waitall(volatile int *ptr, int count, int val) {
    DEBUGMSG_D("Wait on %p starting\n", ptr);
    for (int i = 0; i < count; i++) {
        volatile int *p = ptr + i;
        while (*p != val)
            ;
    }
    DEBUGMSG_D("Wait on %p completed\n", ptr);
}

static inline int is_capturing(cudaStream_t stream) {
    cudaStreamCaptureStatus status;

    CUDA_CHECK(cudaStreamIsCapturing(stream, &status));

    return status != cudaStreamCaptureStatusNone;
}

static inline int try_complete_wait_op(int idx) {
    int completed = 0;

    mpiacx_op_completion_mutex.lock();

    // The proxy already completed the underlying MPI operation
    // Advance to cleanup and don't issue anything into the stream
    if (mpiacx_state->flags[idx] == MPIACX_OP_STATE_COMPLETED) {
        MPIACX_Op *op = &mpiacx_state->op[idx];

        if(op->sendrecv.status)
            memcpy(op->sendrecv.status, &op->sendrecv.status_save, sizeof(MPI_Status));

        if (op->kind == MPIACX_OP_ISEND || op->kind == MPIACX_OP_IRECV)
            mpiacx_state->flags[idx] = MPIACX_OP_STATE_CLEANUP;

        completed = 1;
    }

    mpiacx_op_completion_mutex.unlock();

    return completed;
}

static void cb_graph_cleanup(void *req_ptr) {
    MPIACX_Request *ireq = (MPIACX_Request *) req_ptr;

    if (ireq->kind == MPIACX_REQ_BASIC) {
        size_t idx        = ireq->basic.flag_idx;
        volatile int *ptr = &mpiacx_state->flags[idx];

        // Note, there is a race between the proxy thread completing the request
        // and this call. Only the proxy thread should complete the MPI request.
        // Wait for the proxy to signal completion.

        DEBUGMSG("Host wait on %p starting\n", ptr);
        while (*ptr != MPIACX_OP_STATE_COMPLETED)
            ;
        DEBUGMSG("Host wait on %p completed\n", ptr);

        mpiacx_triggered_slot_free(idx);
        free(ireq);
    }
    else
        printf("cb_graph_cleanup: Could not cleanup request kind %d", ireq->kind);
}

int MPIX_Isend_enqueue(const void *buf, int count, MPI_Datatype datatype, int dest,
                       int tag, MPI_Comm comm, MPIX_Request *request, int qtype, void *queue) {
    int err;
    size_t idx;
    cudaError_t cerr;
    MPIACX_Request *ireq = NULL;

    err = mpiacx_triggered_isend_init(buf, count, datatype, dest, tag, comm, &idx);
    if (err) goto out;

    ireq = (MPIACX_Request *) malloc(sizeof(MPIACX_Request));
    NULL_CHECK_AND_JMP(err, out, ireq);

    ireq->kind            = MPIACX_REQ_BASIC;
    ireq->basic.flag_idx  = idx;

    cerr = cudaGetLastError();
    if (cerr != cudaSuccess) {
        ERRMSG("CUDA Error: %s\n", cudaGetErrorName(cerr));
        err = 1;
        goto out;
    }

    if (qtype == MPIX_QUEUE_CUDA_STREAM) {
        cudaStream_t stream = *(cudaStream_t*)queue;
        const int capture_enabled = is_capturing(stream);

        // TODO: memOps currently not supported on graphs
        if (mpiacx_state->use_memops && !capture_enabled) {
            CU_CHECK_AND_JMP(err, out,
                    wrapWriteValue(stream, mpiacx_state->flags_d_ptr + (idx*sizeof(int)),
                                        MPIACX_OP_STATE_PENDING,
                                        CU_STREAM_WRITE_VALUE_DEFAULT));
        }
        else {
            set<<<1, 1, 0, stream>>>(&mpiacx_state->flags_d[idx], MPIACX_OP_STATE_PENDING);

            cerr = cudaGetLastError();
            if (cerr != cudaSuccess) {
                ERRMSG("CUDA Error: %s\n", cudaGetErrorName(cerr));
                err = 1;
                goto out;
            }
        }

        if (capture_enabled) {
            cudaGraph_t             capture_graph;
            cudaStreamCaptureStatus capture_status;
            cudaUserObject_t        cleanup_obj;

            CUDA_CHECK_AND_JMP(err, out,
                    cudaStreamGetCaptureInfo_v2(stream, &capture_status, NULL, &capture_graph, NULL, NULL));

            CUDA_CHECK_AND_JMP(err, out, cudaUserObjectCreate(&cleanup_obj, (void*) ireq, &cb_graph_cleanup, 1, cudaUserObjectNoDestructorSync));
            CUDA_CHECK_AND_JMP(err, out, cudaGraphRetainUserObject(capture_graph, cleanup_obj, 1, cudaGraphUserObjectMove));
        }
    }
    else if (qtype == MPIX_QUEUE_CUDA_GRAPH) {
        cudaGraph_t *graph = (cudaGraph_t*) queue;
        cudaGraphNode_t graph_node;
        cudaKernelNodeParams params;

        void *op_ptr           = (void*)&mpiacx_state->flags_d[idx];
        int   op_state         = MPIACX_OP_STATE_PENDING;
        void *kernel_params[2] = { &op_ptr, &op_state };

        params.func           = (void*) set;
        params.gridDim        = 1;
        params.blockDim       = 1;
        params.sharedMemBytes = 0;
        params.kernelParams   = kernel_params;
        params.extra          = NULL;

        CUDA_CHECK_AND_JMP(err, out, cudaGraphCreate(graph, 0));
        CUDA_CHECK_AND_JMP(err, out, cudaGraphAddKernelNode(&graph_node, *graph, NULL, 0, &params));

        cudaUserObject_t cleanup_obj;
        CUDA_CHECK_AND_JMP(err, out, cudaUserObjectCreate(&cleanup_obj, (void*) ireq, &cb_graph_cleanup, 1, cudaUserObjectNoDestructorSync));
        CUDA_CHECK_AND_JMP(err, out, cudaGraphRetainUserObject(*graph, cleanup_obj, 1, cudaGraphUserObjectMove));
    }
    else {
        ERRMSG("Invalid queue type (%d)\n", qtype);
        err = 1;
        goto out;
    }

out:
    if (err) {
        free(ireq);
        if (idx < mpiacx_state->nflags)
            mpiacx_triggered_slot_free(idx);
    }

    if (!err)
        *(MPIACX_Request **)request = ireq;

    // FIXME: Should we return err here or call the error handler on comm?

    return err;
}


int MPIX_Irecv_enqueue(void *buf, int count, MPI_Datatype datatype, int source,
                       int tag, MPI_Comm comm, MPIX_Request *request, int qtype, void *queue) {
    int err = 0;
    size_t idx;
    cudaError_t cerr;
    MPIACX_Request *ireq = NULL;

    err = mpiacx_triggered_irecv_init(buf, count, datatype, source, tag, comm, &idx);
    if (err) goto out;

    ireq = (MPIACX_Request *) malloc(sizeof(MPIACX_Request));
    NULL_CHECK_AND_JMP(err, out, ireq);

    ireq->kind            = MPIACX_REQ_BASIC;
    ireq->basic.flag_idx  = idx;

    cerr = cudaGetLastError();
    if (cerr != cudaSuccess) {
        ERRMSG("CUDA Error: %s\n", cudaGetErrorName(cerr));
        err = 1;
        goto out;
    }

    if (qtype == MPIX_QUEUE_CUDA_STREAM) {
        cudaStream_t stream = *(cudaStream_t*)queue;
        const int capture_enabled = is_capturing(stream);

        // TODO: memOps currently not supported on graphs
        if (mpiacx_state->use_memops && !capture_enabled) {
            CU_CHECK_AND_JMP(err, out,
                    wrapWriteValue(stream, mpiacx_state->flags_d_ptr + (idx*sizeof(int)),
                                        MPIACX_OP_STATE_PENDING,
                                        CU_STREAM_WRITE_VALUE_DEFAULT));
        }
        else {
            set<<<1, 1, 0, stream>>>(&mpiacx_state->flags_d[idx], MPIACX_OP_STATE_PENDING);

            cerr = cudaGetLastError();
            if (cerr != cudaSuccess) {
                ERRMSG("CUDA Error: %s\n", cudaGetErrorName(cerr));
                err = 1;
                goto out;
            }
        }

        if (capture_enabled) {
            cudaGraph_t             capture_graph;
            cudaStreamCaptureStatus capture_status;
            cudaUserObject_t        cleanup_obj;

            CUDA_CHECK_AND_JMP(err, out,
                    cudaStreamGetCaptureInfo_v2(stream, &capture_status, NULL, &capture_graph, NULL, NULL));

            CUDA_CHECK_AND_JMP(err, out, cudaUserObjectCreate(&cleanup_obj, (void*) ireq, &cb_graph_cleanup, 1, cudaUserObjectNoDestructorSync));
            CUDA_CHECK_AND_JMP(err, out, cudaGraphRetainUserObject(capture_graph, cleanup_obj, 1, cudaGraphUserObjectMove));
        }
    }
    else if (qtype == MPIX_QUEUE_CUDA_GRAPH) {
        cudaGraph_t *graph = (cudaGraph_t*) queue;
        cudaGraphNode_t graph_node;
        cudaKernelNodeParams params;

        void *op_ptr           = (void*)&mpiacx_state->flags_d[idx];
        int   op_state         = MPIACX_OP_STATE_PENDING;
        void *kernel_params[2] = { &op_ptr, &op_state };

        params.func           = (void*) set;
        params.gridDim        = 1;
        params.blockDim       = 1;
        params.sharedMemBytes = 0;
        params.kernelParams   = kernel_params;
        params.extra          = NULL;

        CUDA_CHECK_AND_JMP(err, out, cudaGraphCreate(graph, 0));
        CUDA_CHECK_AND_JMP(err, out, cudaGraphAddKernelNode(&graph_node, *graph, NULL, 0, &params));

        cudaUserObject_t cleanup_obj;
        CUDA_CHECK_AND_JMP(err, out, cudaUserObjectCreate(&cleanup_obj, (void*) ireq, &cb_graph_cleanup, 1, cudaUserObjectNoDestructorSync));
        CUDA_CHECK_AND_JMP(err, out, cudaGraphRetainUserObject(*graph, cleanup_obj, 1, cudaGraphUserObjectMove));
    }
    else {
        ERRMSG("Invalid queue type (%d)\n", qtype);
        err = 1;
        goto out;
    }

out:
    if (err) {
        free(ireq);
        if (idx < mpiacx_state->nflags)
            mpiacx_triggered_slot_free(idx);
    }
    else
        *(MPIACX_Request **)request = ireq;

    return err;
}


int MPIX_Wait_enqueue(MPIX_Request *req, MPI_Status *status, int qtype, void *queue) {
    MPIACX_Request *ireq = *(MPIACX_Request **) req;
    size_t idx;
    int err = 0;
    cudaError_t cerr;

    if (ireq->kind == MPIACX_REQ_PARTITIONED) {
        ERRMSG("MPIX_Wait_enqueue on partitioned requests is not yet supported\n");
        err = 1;
        goto out;
    }

    idx = ireq->basic.flag_idx;

    cerr = cudaGetLastError();
    if (cerr != cudaSuccess) {
        ERRMSG("CUDA Error: %s\n", cudaGetErrorName(cerr));
        err = 1;
        goto out;
    }

    if (status != MPI_STATUS_IGNORE)
        mpiacx_state->op[idx].sendrecv.status = status;
    else
        mpiacx_state->op[idx].sendrecv.status = NULL;

    mpiacx_state->op[idx].sendrecv.ireq = ireq;

    // TODO: Polling on host mapped memory will generate loads across PCIe.
    // This can be fixed by allocating device memory for flags_d and using
    // GDRCopy to set the flag from the proxy thread.

    if (qtype == MPIX_QUEUE_CUDA_STREAM) {
        cudaStream_t stream = *(cudaStream_t*)queue;
        const int capture_enabled = is_capturing(stream);

        // TODO: memOps currently not supported on graphs
        if (mpiacx_state->use_memops && !capture_enabled) {

            if (try_complete_wait_op(idx)) {
                goto cleanup;
            }

            CU_CHECK_AND_JMP(err, out, 
                    wrapWaitValue(stream, mpiacx_state->flags_d_ptr + (idx*sizeof(int)),
                                        MPIACX_OP_STATE_COMPLETED,
                                        CU_STREAM_WAIT_VALUE_EQ
#ifdef FLUSH_REMOTE_WRITES
                                        | CU_STREAM_WAIT_VALUE_FLUSH
#endif
                                        ));

            CU_CHECK_AND_JMP(err, out,
                    wrapWriteValue(stream, mpiacx_state->flags_d_ptr + (idx*sizeof(int)),
                                        MPIACX_OP_STATE_CLEANUP,
                                        CU_STREAM_WRITE_VALUE_DEFAULT));
        }
        else {
            if (!capture_enabled) {
                if (try_complete_wait_op(idx)) {
                    goto cleanup;
                }

                wait_and_set<<<1, 1, 0, stream>>>(mpiacx_state->flags_d + idx, MPIACX_OP_STATE_COMPLETED, MPIACX_OP_STATE_CLEANUP);
            } else
                wait<<<1, 1, 0, stream>>>(mpiacx_state->flags_d + idx, MPIACX_OP_STATE_COMPLETED);

            cerr = cudaGetLastError();
            if (cerr != cudaSuccess) {
                ERRMSG("CUDA Error: %s\n", cudaGetErrorName(cerr));
                err = 1;
                goto out;
            }
        }
    }
    else if (qtype == MPIX_QUEUE_CUDA_GRAPH) {
        cudaGraph_t *graph = (cudaGraph_t*) queue;
        cudaGraphNode_t graph_node;
        cudaKernelNodeParams params;

        void *op_ptr           = (void*)&mpiacx_state->flags_d[idx];
        int   op_state         = MPIACX_OP_STATE_PENDING;
        void *kernel_params[2] = { &op_ptr, &op_state };

        params.func           = (void*) wait;
        params.gridDim        = 1;
        params.blockDim       = 1;
        params.sharedMemBytes = 0;
        params.kernelParams   = kernel_params;
        params.extra          = NULL;

        CUDA_CHECK_AND_JMP(err, out, cudaGraphCreate(graph, 0));
        CUDA_CHECK_AND_JMP(err, out, cudaGraphAddKernelNode(&graph_node, *graph, NULL, 0, &params));
    }
    else {
        ERRMSG("Invalid queue type (%d)\n", qtype);
        err = 1;
        goto out;
    }

cleanup:
    // FIXME: Partitioned operations should not reset the request handle
    *req = MPIX_REQUEST_NULL;

out:
    return err;
}


int MPIX_Waitall_enqueue(int count, MPIX_Request *reqs, MPI_Status *statuses, int qtype, void *queue) {
    int err = 0;
    cudaError_t cerr;
    CUstreamBatchMemOpParams *params = NULL;

    cerr = cudaGetLastError();
    if (cerr != cudaSuccess) {
        ERRMSG("CUDA Error: %s\n", cudaGetErrorName(cerr));
        err = 1;
        goto out;
    }

    for (int i = 0; i < count; i++) {
        MPIACX_Request *ireq = (MPIACX_Request *) reqs[i];
        size_t idx;

        if (ireq->kind == MPIACX_REQ_PARTITIONED) {
            ERRMSG("MPIX_Wait_enqueue on partitioned requests is not yet supported\n");
            err = 1;
            goto out;
        }

        idx = ireq->basic.flag_idx;

        if (statuses != MPI_STATUSES_IGNORE)
            mpiacx_state->op[idx].sendrecv.status = &statuses[i];
        else
            mpiacx_state->op[idx].sendrecv.status = NULL;

        mpiacx_state->op[idx].sendrecv.ireq = ireq;
    }

    // TODO: Polling on host mapped memory will generate loads across PCIe.  If
    // it affects performance, we could allocate flags_d in device memory and
    // use GDRCopy to set the flag from the proxy thread.

    if (qtype == MPIX_QUEUE_CUDA_STREAM) {
        cudaStream_t stream = *(cudaStream_t*)queue;
        const int capture_enabled = is_capturing(stream);

        if (mpiacx_state->use_memops && !capture_enabled) {
            CUresult cu_ret = CUDA_SUCCESS;
            int num_memops = 0;

            params = (CUstreamBatchMemOpParams *) calloc(2 * count, sizeof(CUstreamBatchMemOpParams));
            NULL_CHECK_AND_JMP(err, out, params);

            for (int i = 0; i < count; i++) {
                size_t idx = ((MPIACX_Request *) reqs[i])->basic.flag_idx;

                if (!try_complete_wait_op(idx)) {
                    params[num_memops].waitValue.operation = CU_STREAM_MEM_OP_WAIT_VALUE_32;
                    params[num_memops].waitValue.address   = mpiacx_state->flags_d_ptr + (idx*sizeof(int));
                    params[num_memops].waitValue.value     = MPIACX_OP_STATE_COMPLETED;
                    params[num_memops].waitValue.flags     = CU_STREAM_WAIT_VALUE_EQ;
#ifdef FLUSH_REMOTE_WRITES
                    params[num_memops].waitValue.flags    |= CU_STREAM_WAIT_VALUE_FLUSH;
#endif

                    params[num_memops+1].writeValue.operation = CU_STREAM_MEM_OP_WRITE_VALUE_32;
                    params[num_memops+1].writeValue.address   = mpiacx_state->flags_d_ptr + (idx*sizeof(int));
                    params[num_memops+1].writeValue.value     = MPIACX_OP_STATE_CLEANUP;
                    params[num_memops+1].writeValue.flags     = CU_STREAM_WRITE_VALUE_DEFAULT; // TODO: It should be safe to set CU_STREAM_WRITE_VALUE_NO_MEMORY_BARRIER

                    num_memops += 2;
                }
            }

            if (num_memops > 0)
                cu_ret = wrapBatchMemOp(stream, 2 * count, params, 0);

            free(params);

            CU_CHECK_AND_JMP(err, out, cu_ret);
        }
        else {

            // TODO: Need to allocate an array to store the list of indices and
            // pass into the waitall kernel. Would also need to free this array
            // afterward. Could do this with a host callback, but it would be really
            // slow. Probably more efficient to use a memory pool and do garbage
            // collection.

            // waitall<<<1, 1, 0, stream>>>(mpiacx_state->flags_d + idx, count, MPIACX_OP_STATE_COMPLETED);

            for (int i = 0; i < count; i++) {
                size_t idx = ((MPIACX_Request *) reqs[i])->basic.flag_idx;

                if (!capture_enabled) {
                    if (!try_complete_wait_op(idx))
                        wait_and_set<<<1, 1, 0, stream>>>(mpiacx_state->flags_d + idx, MPIACX_OP_STATE_COMPLETED, MPIACX_OP_STATE_CLEANUP);
                }
                else {
                    wait<<<1, 1, 0, stream>>>(mpiacx_state->flags_d + idx, MPIACX_OP_STATE_COMPLETED);
                }
            }

            cerr = cudaGetLastError();
            if (cerr != cudaSuccess) {
                ERRMSG("CUDA Error: %s\n", cudaGetErrorName(cerr));
                err = 1;
                goto out;
            }
        }
    }
    else if (qtype == MPIX_QUEUE_CUDA_GRAPH) {
        cudaGraph_t *graph = (cudaGraph_t*) queue;
        cudaGraphNode_t graph_node;
        cudaKernelNodeParams params;
        int op_state = MPIACX_OP_STATE_COMPLETED;

        CUDA_CHECK_AND_JMP(err, out, cudaGraphCreate(graph, 0));

        for (int i = 0; i < count; i++) {
            size_t idx              = ((MPIACX_Request *) reqs[i])->basic.flag_idx;
            void  *op_ptr           = (void*)&mpiacx_state->flags_d[idx];
            void  *kernel_params[2] = { &op_ptr, &op_state };

            params.func           = (void*) wait;
            params.gridDim        = 1;
            params.blockDim       = 1;
            params.sharedMemBytes = 0;
            params.kernelParams   = kernel_params;
            params.extra          = NULL;

            CUDA_CHECK_AND_JMP(err, out, cudaGraphAddKernelNode(&graph_node, *graph, NULL, 0, &params));
        }
    }
    else {
        ERRMSG("Invalid queue type (%d)\n", qtype);
        err = 1;
        goto out;
    }

    // FIXME: Partitioned operations should not reset the request handle
    for (int i = 0; i < count; i++)
        reqs[i] = MPIX_REQUEST_NULL;

out:
    return err;
}


int MPIX_Wait(MPIX_Request *req, MPI_Status *status) {
    MPIACX_Request *ireq = *(MPIACX_Request **) req;

    if (ireq->kind == MPIACX_REQ_BASIC) {
        size_t idx        = ireq->basic.flag_idx;
        volatile int *ptr = (volatile int *) &mpiacx_state->flags_d[idx];
        MPIACX_Op *op      = &mpiacx_state->op[idx];

        // Note, there is a race between the proxy thread completing the request
        // and this call. Only the proxy thread should complete the MPI request.
        // Wait for the proxy to signal completion.

        DEBUGMSG("Host wait on %p starting\n", ptr);
        while (*ptr != MPIACX_OP_STATE_COMPLETED)
            ;
        DEBUGMSG("Host wait on %p completed\n", ptr);

        if (status != MPI_STATUS_IGNORE)
            memcpy(status, &op->sendrecv.status_save, sizeof(MPI_Status));

        mpiacx_triggered_slot_free(idx);

        free(*req);
        *req = MPIX_REQUEST_NULL;
    }
    else if (ireq->kind == MPIACX_REQ_PARTITIONED) {
#ifdef USE_MPI_PARTITIONED
        size_t *idx = ireq->partitioned.flag_idx;

        // Wait for proxy to finish processing pready/parrived calls

        for (int i = 0; i < ireq->partitioned.partitions; i++) {
            volatile int *ptr = (volatile int *) &mpiacx_state->flags_d[idx[i]];

            DEBUGMSG("Host wait on %s %p starting\n", ireq->partitioned.op == MPIACX_OP_PSEND ? "psend" : "precv", ptr);
            while (*ptr != MPIACX_OP_STATE_COMPLETED)
                ;
            DEBUGMSG("Host wait on %s %p completed\n", ireq->partitioned.op == MPIACX_OP_PSEND ? "psend" : "precv", ptr);

            *ptr = MPIACX_OP_STATE_RESERVED;
        }

        // Note, the proxy thread does not wait/test this request, so it should
        // be completed here.

        MPI_Wait(&ireq->partitioned.request, status);
#else
        ERRMSG("Host wait on partitioned op, but partitioned support is disabled\n");
        return 1;
#endif
    }
    else {
        ERRMSG("Invalid request kind (%d)\n", ireq->kind);
        return 1;
    }

    return 0;
}


int MPIX_Waitall(int count, MPIX_Request *reqs, MPI_Status *statuses) {
    int err = 0;

    for (int i = 0; i < count; i++) {
        err = MPIX_Wait(&reqs[i], (statuses != MPI_STATUSES_IGNORE) ? &statuses[i] : MPI_STATUS_IGNORE);
        if (err) break;
    }

    return err;
}


int MPIX_Request_free(MPIX_Request *request) {
    int err = 0;
    MPIACX_Request *ireq = *(MPIACX_Request **) request;

    if (ireq == NULL) {
        ERRMSG("NULL Request\n");
        err = 1;
        goto out;
    }

    // Note, request must be an inactive operation. This function does not support
    // freeing of active requests.

    if (ireq->kind == MPIACX_REQ_PARTITIONED) {
        MPI_Request_free(&ireq->partitioned.request);

        for (int i = 0; i < ireq->partitioned.partitions; i++)
            mpiacx_triggered_slot_free(ireq->partitioned.flag_idx[i]);

        cudaFree(ireq->partitioned.flag_idx_d);
        free(ireq->partitioned.flag_idx);

        free(*request);
        *request = MPIX_REQUEST_NULL;
    }

out:
    return err;
}
