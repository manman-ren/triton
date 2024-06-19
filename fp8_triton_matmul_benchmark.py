import itertools
from enum import Enum

import torch

import triton.language as tl
import triton.ops as tt_ops
from triton.ops.matmulTma import matmulTma as triton_op_matmulTma
import triton.testing as tt_testing
from triton.runtime.jit import reinterpret as tl_reinterpret

fp8_fast_accum = True
allow_tf32 = torch.backends.cuda.matmul.allow_tf32


def triton_mm(tensor_a, tensor_b):
    # parameters: a, b, acc_dtype, allow_tf32, fp8_fast_accum, output_dtype
    #return tt_ops.matmul(tensor_a, tensor_b, None, None, fp8_fast_accum, None)
    return triton_op_matmulTma(tensor_a, tensor_b, None, None, fp8_fast_accum, None)


def triton_forward(input_, weight, unused_output):
    return triton_mm(input_, weight)


def triton_backward_input(unused_input, weight_t, output):
    # weight is pre-transposed
    return triton_mm(output, weight_t)


def triton_backward_weight(input_t, unused_weight, output):
    # input is pre-transposed
    return triton_mm(input_t, output)


def product_dict(**kwargs):
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in itertools.product(*vals):
        yield dict(zip(keys, instance))


class Direction(Enum):
    FWD = "forward"
    BWD_IN = "backward_input"
    BWD_W = "backward_weight"


triton_funcs = {
    Direction.FWD: triton_forward,
    Direction.BWD_IN: triton_backward_input,
    Direction.BWD_W: triton_backward_weight,
}


def convert_fp8_type(tensor):
    # Based on https://github.com/openai/triton/blob/768fc1fcd98/python/triton/language/semantic.py#L1295
    # tl.float8e4nv is supported only for CUDA arch >= 90
    # tl.float8e4b15 is supported only for CUDA arch < 90
    # tl.float8e5 is supported for all
    # Forward: A and B are 4 bit exponent
    # Backward: A is 4 bit, B is 5 bit
    if tensor.dtype == torch.float8_e4m3fn:
        # TODO: add support for arch < 90
        return tl_reinterpret(tensor, dtype=tl.float8e4nv)
    elif tensor.dtype == torch.float8_e5m2:
        return tl_reinterpret(tensor, dtype=tl.float8e5)
    return tensor


def run_bench(shape, direction):
    tt_fn = triton_funcs[direction]
    m, n, k = shape

    # CUDA's FP8 only supports NT GEMMs (the second matrix is transposed) Thus,
    # we tranpose the second matrix to make sure that we use the native CUDA FP8
    weight_shape = (n, k) if direction == direction.FWD else (k, n)
    output_shape = (m, n) if direction == direction.BWD_IN else (n, m)

    base_dtype = torch.float
    input_ = torch.randn((m, k), device="cuda", dtype=base_dtype)
    weight = torch.randn(weight_shape, device="cuda", dtype=base_dtype)
    output = torch.randn(output_shape, device="cuda", dtype=base_dtype)

    # Weight is transposed for both FWD and BWD_IN
    weight = weight.t()
    if direction == direction.BWD_W:
        # Pre-transpose input and make sure that it is contiguous
        input_ = input_.t().contiguous()
        output = output.t()

    # Cast tensors to torch.float8_e4m3fn
    dtype = torch.float8_e4m3fn
    input_ = input_.to(dtype)
    weight = weight.to(dtype)
    output = output.to(dtype)

    # Cast tensors to tl.float8e4nv
    input_ = convert_fp8_type(input_)
    weight = convert_fp8_type(weight)
    output = convert_fp8_type(output)

    # Run benchmark and compute Tflops
    ms = tt_testing.do_bench(lambda: tt_fn(input_, weight, output), )
    tflops = (2 * m * n * k) / 1e12
    sec = ms / 1e3
    perf_str = f"{tflops / sec:.4f}"

    print(
        f"shape {str(shape):<25} direction {direction:<20} tflops {perf_str:<8}",
        flush=True,
    )


if __name__ == "__main__":
    cases = list(
        product_dict(
            shape=[
                (256, 256, 256),
                (512, 512, 512),
                (1024, 1024, 1024),
                (2048, 2048, 2048),
                (4096, 4096, 4096),
                (8192, 8192, 4096),
                #(10240, 10240, 10240),
                #(16384, 16384, 16384),
                #(20480, 20480, 20480),
            ],
            direction=[
                Direction.FWD,
                #Direction.BWD_IN,
                #Direction.BWD_W,
            ],
        ))

    for case in cases:
        run_bench(**case)
