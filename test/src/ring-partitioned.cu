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
#include <mpi.h>
#include <mpi-acx.h>

#define NUM_PARTITIONS 10
#define NUM_ITER 10

__global__ void mark_ready(MPIX_Prequest preq) {
    MPIX_Pready(threadIdx.x, preq);
}

__global__ void wait_until_arrived(MPIX_Prequest preq) {
    int flag;
    do {
        MPIX_Parrived(preq, threadIdx.x, &flag);
    } while (!flag);
}

int main(int argc, char **argv) {
    int provided, ret, ndevices;
    cudaError_t curet;
    int world_rank, world_size;
    int errs = 0;

    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);

    if (provided < MPI_THREAD_MULTIPLE)
        MPI_Abort(MPI_COMM_WORLD, 1);

    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    curet = cudaGetDeviceCount(&ndevices);
    if (curet != cudaSuccess) {
        printf("Failed to get number of CUDA devices\n");
        MPI_Abort(MPI_COMM_WORLD, 2);
    }

    curet = cudaSetDevice(world_rank % ndevices);
    if (curet != cudaSuccess) {
        printf("Failed to set CUDA device\n");
        MPI_Abort(MPI_COMM_WORLD, 2);
    }

    ret = MPIX_Init();
    if (ret) {
        printf("Failed to initialize MPI on-stream support\n");
        MPI_Abort(MPI_COMM_WORLD, 2);
    }

    int send[NUM_PARTITIONS], recv[NUM_PARTITIONS];
    MPI_Status status[2];
    MPIX_Request req[2];
    MPIX_Prequest preq[2];

    for (int i = 0; i < NUM_PARTITIONS; i++) {
        send[i] = (world_rank + i) % world_size;
        recv[i] = -1;
    }

    MPIX_Psend_init(&send, NUM_PARTITIONS, 1, MPI_INT,
                    (world_rank + 1) % world_size,
                    0, MPI_COMM_WORLD, MPI_INFO_NULL, &req[0]);
    MPIX_Precv_init(&recv, NUM_PARTITIONS, 1, MPI_INT,
                    (world_rank + world_size - 1) % world_size,
                    0, MPI_COMM_WORLD, MPI_INFO_NULL, &req[1]);

    MPIX_Prequest_create(req[0], &preq[0]);
    MPIX_Prequest_create(req[1], &preq[1]);

    for (int iter = 0; iter < NUM_ITER; iter++) {
        if (world_rank == 0) printf("********** ITERATION %d **********\n", iter);

        MPIX_Startall(2, req);

        mark_ready<<<1, NUM_PARTITIONS, 0, 0>>>(preq[0]);
        wait_until_arrived<<<1, NUM_PARTITIONS, 0, 0>>>(preq[1]);

        curet = cudaStreamSynchronize(0);
        if (curet != cudaSuccess) {
            printf("Failed to sync CUDA stream: %s\n", cudaGetErrorName(curet));
            MPI_Abort(MPI_COMM_WORLD, 2);
        }

        MPIX_Waitall(2, req, status);

        for (int i = 0; i < NUM_PARTITIONS; i++) {
            int expected = (world_rank + world_size - 1 + i) % world_size + iter;
            if (recv[i] != expected) {
                printf("[%2d] Error: Partition %d, Sent %d, received %d, expected %d\n",
                        world_rank, i, send[i], recv[i], expected);
                ++errs;
            }
        }

        for (int i = 0; i < NUM_PARTITIONS; i++) send[i]++;
    }

    MPIX_Prequest_free(&preq[0]);
    MPIX_Prequest_free(&preq[1]);
    MPIX_Request_free(&req[0]);
    MPIX_Request_free(&req[1]);

    MPI_Allreduce(MPI_IN_PLACE, &errs, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

    MPIX_Finalize();
    MPI_Finalize();

    return errs != 0;
}
