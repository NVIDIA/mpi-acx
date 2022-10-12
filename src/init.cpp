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

#include <atomic>
#include <mutex>
#include <thread>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <mpi-acx.h>
#include <mpi-acx-internal.h>

const char *MPIACX_Op_state_str[] = {
    "MPIACX_OP_STATE_AVAILABLE",
    "MPIACX_OP_STATE_RESERVED",
    "MPIACX_OP_STATE_PENDING",
    "MPIACX_OP_STATE_ISSUED",
    "MPIACX_OP_STATE_COMPLETED",
    "MPIACX_OP_STATE_CLEANUP"
};

MPIACX_State *mpiacx_state;

static std::thread *    progress_thread;
static std::atomic<int> progress_thread_exit;
       std::mutex       mpiacx_op_completion_mutex;

static void progress_thread_fn(void) {

    for (;;) {
        if (progress_thread_exit.load(std::memory_order_acquire))
            break;

        for (size_t i = 0; i < mpiacx_state->nflags; i++) {
            MPIACX_Op *op = &mpiacx_state->op[i];

            if (mpiacx_state->flags[i] == MPIACX_OP_STATE_PENDING) {
                switch (op->kind) {
                    case MPIACX_OP_ISEND:
                        // FIXME: Assuming MPI_ERRORS_ARE_FATAL/ABORT.
                        // Would be good to also support MPI_ERRORS_RETURN.
                        MPI_Isend(op->sendrecv.buf, op->sendrecv.count, op->sendrecv.datatype, op->sendrecv.peer,
                                  op->sendrecv.tag, op->sendrecv.comm, &op->sendrecv.request);
                        DEBUGMSG("Proxy [%zu]: Advanced Isend to issued\n", i);
                        mpiacx_state->flags[i] = MPIACX_OP_STATE_ISSUED;
                        break;

                    case MPIACX_OP_IRECV:
                        MPI_Irecv(op->sendrecv.buf, op->sendrecv.count, op->sendrecv.datatype, op->sendrecv.peer,
                                  op->sendrecv.tag, op->sendrecv.comm, &op->sendrecv.request);
                        DEBUGMSG("Proxy [%zu]: Advanced Irecv to issued\n", i);
                        mpiacx_state->flags[i] = MPIACX_OP_STATE_ISSUED;
                        break;

                    case MPIACX_OP_PSEND:
#ifdef USE_MPI_PARTITIONED
                        MPI_Pready(op->partitioned.partition, op->partitioned.request);
                        DEBUGMSG("Proxy [%zu]: Advanced Pready to issued\n", i);
                        mpiacx_state->flags[i] = MPIACX_OP_STATE_COMPLETED;
#else
                        ERRMSG("Proxy encountered a partitioned send, but partitioned support is disabled\n");
#endif
                        break;

                    default:
                        ERRMSG("Invalid pending op (%d)\n", op->kind);
                        MPI_Abort(MPI_COMM_WORLD, 10);
                }

            }

            if (mpiacx_state->flags[i] == MPIACX_OP_STATE_ISSUED) {
                int flag;

                assert(op->kind != MPIACX_OP_PSEND);

                if (op->kind == MPIACX_OP_PRECV) {
#ifdef USE_MPI_PARTITIONED
                    MPI_Parrived(op->partitioned.request, op->partitioned.partition, &flag);

                    if (flag) {
                        DEBUGMSG("Proxy [%zu]: Parrived partition %d advanced to completed\n", i, op->partitioned.partition);
                        mpiacx_state->flags[i] = MPIACX_OP_STATE_COMPLETED;
                    }
#else
                        ERRMSG("Proxy encountered a partitioned recv, but partitioned support is disabled\n");
#endif
                }
                else {

                    // There is a race between enqueueing a wait operation on
                    // the request and the request being completed by the
                    // proxy. Need to add a mutex around this case and Wait
                    // operations to eliminate this race.

                    mpiacx_op_completion_mutex.lock();

                    MPI_Test(&op->sendrecv.request, &flag, &op->sendrecv.status_save);

                    if (flag) {
                        DEBUGMSG("Proxy [%zu]: Other op advanced to completed\n", i);

                        // If an on-stream wait was already posted, copy status to
                        // the user's status object. Otherwise, status gets copied
                        // in the call to MPIX_Wait*

                        if (op->sendrecv.ireq != NULL && op->sendrecv.status != NULL)
                            memcpy(op->sendrecv.status, &op->sendrecv.status_save, sizeof(MPI_Status));

                        mpiacx_state->flags[i] = MPIACX_OP_STATE_COMPLETED;
                    }

                    mpiacx_op_completion_mutex.unlock();
                }

                if (mpiacx_state->flags[i] == MPIACX_OP_STATE_CLEANUP) {
                    // The operation was completed on a stream and needs to be cleaned up
                    // In-graph and persistent operations should not enter this state
                    assert(op->kind == MPIACX_OP_ISEND || op->kind == MPIACX_OP_IRECV);

                    free(op->sendrecv.ireq);
                    mpiacx_triggered_slot_free(i);
                }
            }
        }
    }
}


int MPIX_Init(void) {
    int initialized, finalized, thread_level;
    int err = 0;
    int device;
    int can_memops, can_flush = 1;
    const char *nflags_env_str = getenv("MPIACX_NFLAGS");

    MPI_Initialized(&initialized);
    MPI_Finalized(&finalized);

    if (!initialized || finalized) {
        ERRMSG("MPI library is not initialized (%d) or is finalized (%d)\n",
               initialized, finalized);
        err = 1;
        goto out;
    }

    MPI_Query_thread(&thread_level);

    if (thread_level < MPI_THREAD_MULTIPLE) {
        ERRMSG("MPI library thread level (%d) is less than MPI_THREAD_MULTIPLE\n",
               thread_level);
        err = 1;
        goto out;
    }

    mpiacx_state = (MPIACX_State *) calloc(1, sizeof(MPIACX_State));
    NULL_CHECK_AND_JMP(err, out, mpiacx_state);

    if (!getenv("MPIACX_DISABLE_MEMOPS")) {
        CUDA_CHECK(cudaGetDevice(&device));
#ifdef USE_MEMOPS_V2
        can_memops = 1;
#else
        CU_CHECK(cuDeviceGetAttribute(&can_memops, CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS, device));
#endif
#ifdef FLUSH_REMOTE_WRITES
        CU_CHECK(cuDeviceGetAttribute(&can_flush, CU_DEVICE_ATTRIBUTE_CAN_FLUSH_REMOTE_WRITES, device));
#endif
        mpiacx_state->use_memops = can_memops && can_flush;

        if (!mpiacx_state->use_memops)
            printf("Warning: Memops not supported, falling back to kernels\n");
    } else {
        printf("Warning: Memops disabled, falling back to kernels\n");
        mpiacx_state->use_memops = 0;
    }

    if (nflags_env_str) {
        long nflags_env = atol(nflags_env_str);
        if (nflags_env <= 0) {
            ERRMSG("Invalid value in MPIACX_NFLAGS (%s)\n", nflags_env_str);
            err = 1;
            goto out;
        }

        mpiacx_state->nflags = nflags_env;
    }
    else
        mpiacx_state->nflags = MPIACX_NFLAGS;

    DEBUGMSG("Allocating %zu flags\n", mpiacx_state->nflags);

    CUDA_CHECK(cudaHostAlloc((void **)&mpiacx_state->flags,
                             sizeof(int) * mpiacx_state->nflags, cudaHostAllocMapped));

    CUDA_CHECK(cudaHostGetDevicePointer((void**)&mpiacx_state->flags_d,
                                        (void *)mpiacx_state->flags, 0));

    CU_CHECK_AND_JMP(err, out,
            cuMemHostGetDevicePointer(&mpiacx_state->flags_d_ptr,
                                      (void*) mpiacx_state->flags_d, 0));

    for (size_t i = 0; i < mpiacx_state->nflags; i++)
        mpiacx_state->flags[i] = MPIACX_OP_STATE_AVAILABLE;

    mpiacx_state->op = (MPIACX_Op *) calloc(mpiacx_state->nflags, sizeof(MPIACX_Op));
    NULL_CHECK_AND_JMP(err, out, mpiacx_state->op);

    progress_thread_exit.store(0, std::memory_order_seq_cst);

    progress_thread = new std::thread(progress_thread_fn);

out:
    if (err) {
        if (NULL != mpiacx_state) {
            if (NULL != mpiacx_state->flags)
                cudaFreeHost((void*)mpiacx_state->flags);

            free(mpiacx_state->op);
            free(mpiacx_state);
            mpiacx_state = NULL;
        }
    }

    return err;
}

int MPIX_Finalize(void) {
    int err = 0;

    progress_thread_exit.store(1, std::memory_order_release);

    progress_thread->join();

    for (size_t i = 0; i < mpiacx_state->nflags; i++) {
        if (mpiacx_state->flags[i] != MPIACX_OP_STATE_AVAILABLE &&
            mpiacx_state->flags[i] != MPIACX_OP_STATE_CLEANUP)
            printf("Warning: flags[%zu] = %s\n", i, MPIACX_Op_state_str[mpiacx_state->flags[i]]);
    }


    cudaFreeHost((void*)mpiacx_state->flags);
    free(mpiacx_state->op);
    free(mpiacx_state);
    mpiacx_state = NULL;

    return err;
}
