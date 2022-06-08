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

#define CUDA_CHECK(cmd) \
    do { \
        cudaError_t ret = cmd; \
         \
        if (ret != cudaSuccess) { \
            printf("[%s:%d] CUDA Call failed (%d): %s\n", __FILE__, __LINE__, ret, #cmd); \
            MPI_Abort(MPI_COMM_WORLD, 1); \
        } \
    } while (0)

int main(int argc, char **argv) {
    int provided, ret, ndevices;
    int world_rank, world_size;
    int errs = 0;

    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);

    if (provided < MPI_THREAD_MULTIPLE)
        MPI_Abort(MPI_COMM_WORLD, 1);

    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    CUDA_CHECK(cudaGetDeviceCount(&ndevices));
    CUDA_CHECK(cudaSetDevice(world_rank % ndevices));

    ret = MPIX_Init_stream();
    if (ret) {
        printf("Failed to initialize MPI on-stream support\n");
        MPI_Abort(MPI_COMM_WORLD, 2);
    }

    printf("Hello from %d of %d\n", world_rank, world_size);

    int send = world_rank + 1, recv = -1;
    MPIX_Request req[2];
    cudaGraph_t send_graph, recv_graph, wait_graph, graph;
    cudaGraphNode_t send_node, recv_node, wait_node;

    MPIX_Isend_enqueue(&send, 1, MPI_INT, (world_rank + 1) % world_size,
                       0, MPI_COMM_WORLD, &req[0], MPIX_QUEUE_CUDA_GRAPH, &send_graph);
    MPIX_Irecv_enqueue(&recv, 1, MPI_INT, (world_rank + world_size - 1) % world_size,
                       0, MPI_COMM_WORLD, &req[1], MPIX_QUEUE_CUDA_GRAPH, &recv_graph);

    MPIX_Waitall_enqueue(2, req, MPI_STATUSES_IGNORE, MPIX_QUEUE_CUDA_GRAPH, &wait_graph);

    CUDA_CHECK(cudaGraphCreate(&graph, 0));
    CUDA_CHECK(cudaGraphAddChildGraphNode(&send_node, graph, NULL, 0, send_graph));
    CUDA_CHECK(cudaGraphAddChildGraphNode(&recv_node, graph, &send_node, 1, recv_graph));
    CUDA_CHECK(cudaGraphAddChildGraphNode(&wait_node, graph, &recv_node, 1, wait_graph));

    cudaGraphExec_t graph_exec;
    cudaGraphNode_t error_node;
    char error_msg[256];

    CUDA_CHECK(cudaGraphInstantiate(&graph_exec, graph, &error_node, error_msg, 256));

    // Send my value around the ring until I get it back
    for (int i = 0; i < world_size; i++) {
        if (world_rank == 0) printf("Iteration %d\n", i);
        CUDA_CHECK(cudaGraphLaunch(graph_exec, 0));
        CUDA_CHECK(cudaMemcpyAsync(&send, &recv, sizeof(int), cudaMemcpyHostToHost, 0));
    }

    CUDA_CHECK(cudaStreamSynchronize(0));
    CUDA_CHECK(cudaGraphExecDestroy(graph_exec));
    CUDA_CHECK(cudaGraphDestroy(graph));
    CUDA_CHECK(cudaGraphDestroy(send_graph));
    CUDA_CHECK(cudaGraphDestroy(recv_graph));
    CUDA_CHECK(cudaGraphDestroy(wait_graph));

    int expected = world_rank + 1;
    if (recv != expected) {
        printf("[%2d] Error: received %d, expected %d\n",
                world_rank, recv, expected);
        ++errs;
    }

    MPI_Allreduce(MPI_IN_PLACE, &errs, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

    MPIX_Finalize_stream();
    MPI_Finalize();

    return errs != 0;
}
