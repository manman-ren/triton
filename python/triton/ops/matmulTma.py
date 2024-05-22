import torch
import triton
import numpy as np

from .. import Config, cdiv, jit
from .. import language as tl

_ordered_datatypes = [torch.int8, torch.float16, torch.bfloat16, torch.float32]


def upcast_if_fp8(a):
    if "fp8" in str(a):
        return torch.float16
    return a


def get_higher_dtype(a, b):
    a = upcast_if_fp8(a)
    b = upcast_if_fp8(b)
    if a is b:
        return a

    assert a in _ordered_datatypes
    assert b in _ordered_datatypes

    for d in _ordered_datatypes:
        if a is d:
            return b
        if b is d:
            return a


def init_to_zero(name):
    return lambda nargs: nargs[name].zero_()


def get_configs_io_bound():
    configs = []
    for num_stages in [2, 3, 4, 5, 6]:
        for block_m in [16, 32]:
            for block_k in [32, 64]:
                for block_n in [32, 64, 128, 256]:
                    num_warps = 2 if block_n <= 64 else 4
                    configs.append(
                        Config({'BLOCK_M': block_m, 'BLOCK_N': block_n, 'BLOCK_K': block_k, 'SPLIT_K': 1},
                               num_stages=num_stages, num_warps=num_warps))
                    # split_k (can't support splitk with tma store.
                    #for split_k in [2, 4, 8, 16]:
                    #    configs.append(
                    #        Config({'BLOCK_M': block_m, 'BLOCK_N': block_n, 'BLOCK_K': block_k, 'SPLIT_K': split_k},
                    #               num_stages=num_stages, num_warps=num_warps, pre_hook=init_to_zero('C')))
    return configs


'''
@autotune(
    configs=[
        # basic configs for compute-bound matmuls
        Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'SPLIT_K': 1}, num_stages=3, num_warps=8),
        Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32, 'SPLIT_K': 1}, num_stages=3, num_warps=8),
        Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 32, 'SPLIT_K': 1}, num_stages=4, num_warps=4),
        Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32, 'SPLIT_K': 1}, num_stages=4, num_warps=4),
        Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'SPLIT_K': 1}, num_stages=4, num_warps=4),
        Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'SPLIT_K': 1}, num_stages=4, num_warps=4),
        Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'SPLIT_K': 1}, num_stages=4, num_warps=4),
        Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32, 'SPLIT_K': 1}, num_stages=4, num_warps=4),
        Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32, 'SPLIT_K': 1}, num_stages=5, num_warps=2),
        # good for int8
        Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 128, 'SPLIT_K': 1}, num_stages=3, num_warps=8),
        Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 128, 'SPLIT_K': 1}, num_stages=3, num_warps=8),
        Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 128, 'SPLIT_K': 1}, num_stages=4, num_warps=4),
        Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 128, 'SPLIT_K': 1}, num_stages=4, num_warps=4),
        Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128, 'SPLIT_K': 1}, num_stages=4, num_warps=4),
        Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64, 'SPLIT_K': 1}, num_stages=4, num_warps=4),
        Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64, 'SPLIT_K': 1}, num_stages=4, num_warps=4),
        Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 64, 'SPLIT_K': 1}, num_stages=4, num_warps=4),
        Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 64, 'SPLIT_K': 1}, num_stages=5, num_warps=2),
    ], #+ get_configs_io_bound(),
    key=['M', 'N', 'K'],
)
@heuristics({
    'EVEN_K': lambda args: args['K'] % (args['BLOCK_K'] * args['SPLIT_K']) == 0,
})
'''


@jit
def _kernel(A, B, C, a_desc_ptr, b_desc_ptr, c_desc_ptr, M, N, K,  #
            acc_dtype: tl.constexpr,  #
            input_precision: tl.constexpr,  #
            fp8_fast_accum: tl.constexpr,  #
            BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,  #
            GROUP_M: tl.constexpr, SPLIT_K: tl.constexpr, AB_DTYPE: tl.constexpr):
    # matrix multiplication
    pid = tl.program_id(0)
    #pid_z = tl.program_id(1)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)
    # do matrix multiplication
    offs_am = pid_m * BLOCK_M
    offs_bn = pid_n * BLOCK_N
    offs_k = 0
    #rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    #rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    #ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    #rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    # pointers
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=acc_dtype)
    for k in range(0, tl.cdiv(K, BLOCK_K * SPLIT_K)):
        a = tl._experimental_descriptor_load(a_desc_ptr, [offs_am, offs_k], [BLOCK_M, BLOCK_K], A.dtype.element_ty)
        b = tl._experimental_descriptor_load(b_desc_ptr, [offs_bn, offs_k], [BLOCK_N, BLOCK_K],
                                             B.dtype.element_ty)  #tl.float8e4nv) #INPUT_DTYPE)
        b = tl.trans(b)
        if AB_DTYPE is not None:
            a = a.to(AB_DTYPE)
            b = b.to(AB_DTYPE)
        if fp8_fast_accum:
            acc = tl.dot(a, b, acc, out_dtype=acc_dtype, input_precision=input_precision)
        else:
            acc += tl.dot(a, b, out_dtype=acc_dtype, input_precision=input_precision)
        offs_k += BLOCK_K
    acc = acc.to(C.dtype.element_ty)
    # rematerialize rm and rn to save registers
    #rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    #rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    #mask = (rm < M)[:, None] & (rn < N)[None, :]
    # handles write-back with reduction-splitting
    if SPLIT_K == 1:
        tl._experimental_descriptor_store(c_desc_ptr, acc, [offs_am, offs_bn])
    #else:
    #    tl.atomic_add(C, acc, mask=mask)


class _matmulTma(torch.autograd.Function):
    kernel = _kernel

    _locks = {}

    @staticmethod
    def _call(a, b, acc_dtype, input_precision, fp8_fast_accum, output_dtype):
        device = a.device
        # handle non-contiguous inputs if necessary
        if a.stride(0) > 1 and a.stride(1) > 1:
            a = a.contiguous()
        if b.stride(0) > 1 and b.stride(1) > 1:
            b = b.contiguous()
        # checks constraints
        assert a.shape[1] == b.shape[0], "incompatible dimensions"
        M, K = a.shape
        _, N = b.shape
        print("M N K", M, N, K, b.stride(0), b.stride(1))

        # common type between a and b
        ab_dtype = get_higher_dtype(a.dtype, b.dtype)

        # allocates output
        if (output_dtype is None):
            output_dtype = ab_dtype

        c = torch.empty((M, N), device=device, dtype=output_dtype)

        TMA_SIZE = 128
        desc_a = np.empty(TMA_SIZE, dtype=np.int8)
        desc_b = np.empty(TMA_SIZE, dtype=np.int8)
        desc_c = np.empty(TMA_SIZE, dtype=np.int8)
        #'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32
        BLOCK_M, BLOCK_N, BLOCK_K = 128, 256, 64
        triton.runtime.driver.active.utils.fill_2d_tma_descriptor(a.data_ptr(), M, K, BLOCK_M, BLOCK_K,
                                                                  a.element_size(), desc_a)
        triton.runtime.driver.active.utils.fill_2d_tma_descriptor(b.data_ptr(), N, K, BLOCK_N, BLOCK_K,
                                                                  b.element_size(), desc_b)
        triton.runtime.driver.active.utils.fill_2d_tma_descriptor(c.data_ptr(), M, N, BLOCK_M, BLOCK_N,
                                                                  c.element_size(), desc_c)
        desc_a = torch.tensor(desc_a, device=a.device)
        desc_b = torch.tensor(desc_b, device=a.device)
        desc_c = torch.tensor(desc_c, device=a.device)

        # Allowed types for acc_type given the types of a and b.
        supported_acc_dtypes = {
            torch.float16: (torch.float32, torch.float16), torch.bfloat16: (torch.float32, torch.bfloat16),
            torch.float32: (torch.float32, ), torch.int8: (torch.int32, )
        }

        if acc_dtype is None:
            acc_dtype = supported_acc_dtypes[ab_dtype][0]
        else:
            assert isinstance(acc_dtype, torch.dtype), "acc_dtype must be a torch.dtype"
            assert acc_dtype in supported_acc_dtypes[a.dtype], "acc_dtype not compatible with the type of a"
            assert acc_dtype in supported_acc_dtypes[b.dtype], "acc_dtype not compatible with the type of b"

        def to_tl_type(ty):
            return getattr(tl, str(ty).split(".")[-1])

        def grid(META):
            nonlocal desc_a
            nonlocal desc_b
            nonlocal desc_c
            triton.runtime.driver.active.utils.fill_2d_tma_descriptor(a.data_ptr(), M, K, META['BLOCK_M'],
                                                                      META['BLOCK_K'], a.element_size(), desc_a)
            # 2nd input is pre-transposed, so load as N, K and BLOCK_N BLOCK_K
            triton.runtime.driver.active.utils.fill_2d_tma_descriptor(b.data_ptr(), N, K, META['BLOCK_N'],
                                                                      META['BLOCK_K'], b.element_size(), desc_b)
            triton.runtime.driver.active.utils.fill_2d_tma_descriptor(c.data_ptr(), M, N, META['BLOCK_M'],
                                                                      META['BLOCK_N'], c.element_size(), desc_c)
            desc_a = torch.tensor(desc_a, device=a.device)
            desc_b = torch.tensor(desc_b, device=a.device)
            desc_c = torch.tensor(desc_c, device=a.device)
            return (cdiv(M, META['BLOCK_M']) * cdiv(N, META['BLOCK_N']), META['SPLIT_K'])

        acc_dtype = to_tl_type(acc_dtype)
        ab_dtype = to_tl_type(ab_dtype)
        output_dtype = to_tl_type(output_dtype)
        #a_dtype = to_tl_type(a.dtype)
        print("acc_dtype", str(acc_dtype), "ab_dtype", str(ab_dtype), "output_dtype", str(output_dtype), "a_dtype",
              str(a.dtype))
        print("strdies", a.stride(0), a.stride(1), b.stride(0), b.stride(1),  #
              c.stride(0), c.stride(1))

        # Tensor cores support input with mixed float8 types.
        if a.dtype in [tl.float8e4nv, tl.float8e5] and b.dtype in [tl.float8e4nv, tl.float8e5]:
            ab_dtype = None
        # launch kernel
        #grid = lambda META: (cdiv(M, META['BLOCK_M']) * cdiv(N, META['BLOCK_N']), META['SPLIT_K'])
        _kernel[cdiv(M, BLOCK_M) * cdiv(N, BLOCK_N), 1, 1](
            a, b, c, desc_a, desc_b, desc_c, M, N, K,  #
            acc_dtype=acc_dtype,  #
            input_precision=input_precision,  #
            fp8_fast_accum=fp8_fast_accum,  #
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, GROUP_M=8, SPLIT_K=1, AB_DTYPE=ab_dtype)

        # Print autotune result, i.e, best config
        #print(kernel_info.metadata)
        #print("num_stages=", kernel_info.metadata.num_stages)
        #print("num_warps=", kernel_info.metadata.num_warps)
        #print(kernel_info.src.constants)

        # Dump all IR for NVIDIA GPUs.
        #for ir in ["ttir", "ttgir", "llir", "ptx"]:
        #    with open(f"{ir}.txt", "w") as f:
        #        f.write(kernel_info.asm[ir])
        return c

    @staticmethod
    def forward(ctx, a, b, acc_dtype=None, input_precision=None, fp8_fast_accum=True, output_dtype=None):
        return _matmulTma._call(a, b, acc_dtype=acc_dtype, input_precision=input_precision,
                                fp8_fast_accum=fp8_fast_accum, output_dtype=output_dtype)


matmulTma = _matmulTma.apply
