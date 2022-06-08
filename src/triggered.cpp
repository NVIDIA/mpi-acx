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

#include <string.h>
#include <mpi-acx.h>
#include <mpi-acx-internal.h>

static int mpiacx_triggered_slot_allocate(size_t *idx) {
    int err = 0, idx_found = 0;

    for (size_t i = 0; i < mpiacx_state->nflags; i++) {

        // FIXME: This is only safe when there is a single thread issuing
        // on-stream operations. To make this thread safe, we would need an
        // additional state to mark the flag as allocated and to use atomic CAS
        // to allocate the slot

        if (mpiacx_state->flags[i] == MPIACX_OP_STATE_AVAILABLE) {
            *idx = i;
            idx_found = 1;
            break;
        }
    }

    if (!idx_found) {
        ERRMSG("No available operation slots\n");
        err = 1;
        goto out;
    }

    mpiacx_state->flags[*idx] = MPIACX_OP_STATE_RESERVED;

out:
    return err;
}

void mpiacx_triggered_slot_free(size_t idx) {
    memset(&mpiacx_state->op[idx], 0, sizeof(MPIACX_Op));
    mpiacx_state->flags[idx] = MPIACX_OP_STATE_AVAILABLE;
}

int mpiacx_triggered_isend_init(const void *buf, int count, MPI_Datatype datatype,
                                int dest, int tag, MPI_Comm comm, size_t *idx) {
    int err = 0;
    MPIACX_Op *op = NULL;

    err = mpiacx_triggered_slot_allocate(idx);
    if (err) goto out;

    op = &mpiacx_state->op[*idx];

    op->kind              = MPIACX_OP_ISEND;
    op->sendrecv.buf      = (void *) buf;
    op->sendrecv.count    = count;
    op->sendrecv.datatype = datatype;
    op->sendrecv.peer     = dest;
    op->sendrecv.tag      = tag;
    op->sendrecv.comm     = comm;
    op->sendrecv.request  = MPI_REQUEST_NULL;

out:
    return err;
}


int mpiacx_triggered_irecv_init(void *buf, int count, MPI_Datatype datatype,
                                int source, int tag, MPI_Comm comm, size_t *idx) {
    int err = 0;
    MPIACX_Op *op = NULL;

    err = mpiacx_triggered_slot_allocate(idx);
    if (err) goto out;

    op = &mpiacx_state->op[*idx];

    op->kind              = MPIACX_OP_IRECV;
    op->sendrecv.buf      = buf;
    op->sendrecv.count    = count;
    op->sendrecv.datatype = datatype;
    op->sendrecv.peer     = source;
    op->sendrecv.tag      = tag;
    op->sendrecv.comm     = comm;
    op->sendrecv.request  = MPI_REQUEST_NULL;

out:
    return err;
}


int mpiacx_triggered_pready_init(int partition, MPI_Request request, size_t *idx) {
    int err = 0;
    MPIACX_Op *op = NULL;

    err = mpiacx_triggered_slot_allocate(idx);
    if (err) goto out;

    op = &mpiacx_state->op[*idx];

    op->kind                  = MPIACX_OP_PSEND;
    op->partitioned.partition = partition;
    op->partitioned.request   = request;

out:
    return err;
}


int mpiacx_triggered_parrived_init(int partition, MPI_Request request, size_t *idx) {
    int err = 0;
    MPIACX_Op *op = NULL;

    err = mpiacx_triggered_slot_allocate(idx);
    if (err) goto out;

    op = &mpiacx_state->op[*idx];

    op->kind                  = MPIACX_OP_PRECV;
    op->partitioned.partition = partition;
    op->partitioned.request   = request;

out:
    return err;
}
