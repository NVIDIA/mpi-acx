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

#include <stdio.h>
#include <assert.h>
#include <mpi-acx.h>
#include <mpi-acx-internal.h>

int MPIX_Psend_init(const void *buf, int partitions, MPI_Count count,
                    MPI_Datatype datatype, int dest, int tag, MPI_Comm comm,
                    MPI_Info info, MPIX_Request *request) {
    int err = 0;
    MPIACX_Request *ireq = NULL;

    assert(partitions > 0);

    ireq = (MPIACX_Request *) malloc(sizeof(MPIACX_Request));
    NULL_CHECK_AND_JMP(err, out, ireq);

    ireq->kind = MPIACX_REQ_PARTITIONED;
    ireq->partitioned.op         = MPIACX_OP_PSEND;
    ireq->partitioned.partitions = partitions;
    ireq->partitioned.times_used = 0;
    ireq->partitioned.flag_idx_d = NULL;

    ireq->partitioned.flag_idx = (size_t *) malloc(ireq->partitioned.partitions * sizeof(size_t));
    NULL_CHECK_AND_JMP(err, out, ireq->partitioned.flag_idx);

    // Note: casting away const on the source buffer to work around a bug in MPICH v4.0
    MPI_CHECK(err, out,
              MPI_Psend_init((void*)buf, partitions, count, datatype, dest, tag, comm,
                             info, &ireq->partitioned.request));

    for (int i = 0; i < ireq->partitioned.partitions; i++) {
        int err = mpiacx_triggered_pready_init(i, ireq->partitioned.request,
                                               ireq->partitioned.flag_idx + i);
        if (err) {
            ERRMSG("Error allocating trigger slot (%d)\n", i);
            goto out;
        }
    }

    *request = ireq;

out:
    if (err) {
        if (ireq) free(ireq->partitioned.flag_idx);
        free(ireq);
    }

    return err;
}

int MPIX_Precv_init(void *buf, int partitions, MPI_Count count,
                    MPI_Datatype datatype, int source, int tag, MPI_Comm comm,
                    MPI_Info info, MPIX_Request *request) {
    int err = 0;
    MPIACX_Request *ireq = NULL;

    assert(partitions > 0);

    ireq = (MPIACX_Request *) malloc(sizeof(MPIACX_Request));
    NULL_CHECK_AND_JMP(err, out, ireq);

    ireq->kind = MPIACX_REQ_PARTITIONED;
    ireq->partitioned.op         = MPIACX_OP_PRECV;
    ireq->partitioned.partitions = partitions;
    ireq->partitioned.times_used = 0;
    ireq->partitioned.flag_idx_d = NULL;

    ireq->partitioned.flag_idx = (size_t *) malloc(ireq->partitioned.partitions * sizeof(size_t));
    NULL_CHECK_AND_JMP(err, out, ireq->partitioned.flag_idx);

    MPI_CHECK(err, out,
              MPI_Precv_init(buf, partitions, count, datatype, source, tag, comm,
                             info, &ireq->partitioned.request));

    for (int i = 0; i < ireq->partitioned.partitions; i++) {
        int err = mpiacx_triggered_parrived_init(i, ireq->partitioned.request,
                                                 ireq->partitioned.flag_idx + i);
        if (err) {
            ERRMSG("Error allocating trigger slot (%d)\n", i);
            goto out;
        }
    }

    *request = ireq;

out:
    if (err) {
        if (ireq) free(ireq->partitioned.flag_idx);
        free(ireq);
    }

    return err;
}

int MPIX_Start(MPIX_Request *request) {
    int err = 0;
    MPIACX_Request *ireq = (*(MPIACX_Request**) request);

    assert(ireq->kind == MPIACX_REQ_PARTITIONED);

    MPI_CHECK(err, out, MPI_Start(&ireq->partitioned.request));

    if (ireq->partitioned.op == MPIACX_OP_PRECV) {
        for (int i = 0; i < ireq->partitioned.partitions; i++)
            mpiacx_state->flags[ireq->partitioned.flag_idx[i]] = MPIACX_OP_STATE_ISSUED;
    }

    ireq->partitioned.times_used++;

out:
    if (err) {
        free(ireq->partitioned.flag_idx);
        cudaFree(ireq->partitioned.flag_idx_d);
    }

    return err;
}


int MPIX_Startall(int count, MPIX_Request *request) {
    int err = 0;

    for (int i = 0; i < count && err == 0; i++)
        err = MPIX_Start(&request[i]);

    return err;
}


int MPIX_Prequest_create(MPIX_Request request, MPIX_Prequest *prequest) {
    int err = 0;
    MPIACX_Request *ireq = (MPIACX_Request*) request;
    MPIACX_Prequest preq_h, *preq_d = NULL;

    assert(ireq->kind == MPIACX_REQ_PARTITIONED);

    CUDA_CHECK_AND_JMP(err, out, cudaMalloc(&preq_d, sizeof(MPIACX_Prequest)));

    if (NULL == ireq->partitioned.flag_idx_d) {
        CUDA_CHECK_AND_JMP(err, out, cudaMalloc(&ireq->partitioned.flag_idx_d,
                                                ireq->partitioned.partitions * sizeof(size_t)));
        CUDA_CHECK_AND_JMP(err, out, cudaMemcpy(ireq->partitioned.flag_idx_d,
                                                ireq->partitioned.flag_idx,
                                                ireq->partitioned.partitions * sizeof(size_t),
                                                cudaMemcpyHostToDevice));
    }

    preq_h.idx   = ireq->partitioned.flag_idx_d;
    preq_h.flags = mpiacx_state->flags_d;

    CUDA_CHECK_AND_JMP(err, out, cudaMemcpy(preq_d, &preq_h, sizeof(MPIACX_Prequest),
                                            cudaMemcpyHostToDevice));

    *prequest = preq_d;

out:
    if (err) cudaFree(preq_d);
    return err;
}


int MPIX_Prequest_free(MPIX_Prequest *request) {
    cudaFree(*request);
    *request = MPIX_PREQUEST_NULL;

    return 0;
}


__host__ __device__ int MPIX_Pready(int partition, void *request) {
#ifdef __CUDA_ARCH__
    MPIACX_Prequest *preq = (MPIACX_Prequest *) request;

    preq->flags[preq->idx[partition]] = MPIACX_OP_STATE_PENDING;
#else
    MPIACX_Request *ireq = (*(MPIACX_Request**) request);

    mpiacx_state->flags[ireq->partitioned.flag_idx[partition]] = MPIACX_OP_STATE_PENDING;
#endif

    return 0;
}


__host__ __device__ int MPIX_Parrived(void *request, int partition, int *flag) {
    int state;

#ifdef __CUDA_ARCH__
    MPIACX_Prequest *preq = (MPIACX_Prequest *) request;

    state = preq->flags[preq->idx[partition]];
#else
    MPIACX_Request *ireq = (*(MPIACX_Request**) request);

    state = mpiacx_state->flags[ireq->partitioned.flag_idx[partition]];
#endif

    *flag = (state == MPIACX_OP_STATE_COMPLETED);

    return 0;
}
