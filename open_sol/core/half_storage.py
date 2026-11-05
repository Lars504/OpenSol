"""CUDA source snippets of the stepper precisions (solver_plan S5): fp16 field storage with float32 arithmetic by
the PTX cvt instructions (no cuda_fp16.h needed by NVRTC); arrays are cupy float16, kernels see unsigned short. No
cupy import: engines paste these into their RawKernel / ElementwiseKernel sources."""
import numpy as np

# cvt.f32.f16 / cvt.rn.f16.f32 on a 16-bit register: what cuda_fp16's __half2float / __float2half_rn compile to
HALF_CONVERSIONS = r'''
__device__ __forceinline__ float fp16_bits_to_float(unsigned short value)
{
    float result;
    asm("cvt.f32.f16 %0, %1;" : "=f"(result) : "h"(value));
    return result;
}

__device__ __forceinline__ unsigned short float_to_fp16_bits(float value)
{
    unsigned short result;
    asm("cvt.rn.f16.f32 %0, %1;" : "=h"(result) : "f"(value));
    return result;
}
'''

# ElementwiseKernel preamble: stored_load(x) -> float and stored_set(x, value) overloaded on float (single) and unsigned
# short (fp16 bits, half_bits of a float16 array), so one kernel with a type placeholder serves both precisions.
STORED_ACCESS = HALF_CONVERSIONS + r'''
__device__ __forceinline__ float stored_load(float value) { return value; }
__device__ __forceinline__ float stored_load(unsigned short value) { return fp16_bits_to_float(value); }
__device__ __forceinline__ void stored_set(float& target, float value) { target = value; }
__device__ __forceinline__ void stored_set(unsigned short& target, float value)
{
    target = float_to_fp16_bits(value);
}
'''

STORAGE_PRECISIONS = ('single', 'half_storage')


def half_bits(array):
    """uint16 bit view of a float16 array for the kernels above, else the array itself. cupy's own float16 kernels
    (casts, copyto, arithmetic into float16) need cuda_fp16's __half constructors, unresolved on some CUDA / cupy pairs."""
    return array.view(np.uint16) if array.dtype == np.float16 else array


def storage_preamble(precision, type_name, load, store, scale=1.0) -> str:
    """CUDA typedef `type_name` and macros load(v) -> float, store(x) -> stored value for a stepper precision; scale
    (a power of two: exact) multiplies the stored fp16 value (stored = x * scale) to keep small values normal."""
    if precision == 'single':
        return f'typedef float {type_name};\n#define {load}(value) (value)\n#define {store}(value) (value)\n'
    if precision != 'half_storage':
        raise ValueError(f"storage precision must be one of {', '.join(STORAGE_PRECISIONS)}, got {precision!r}")
    if scale == 1.0:
        return (HALF_CONVERSIONS + f'typedef unsigned short {type_name};\n'
                f'#define {load}(value) fp16_bits_to_float(value)\n'
                f'#define {store}(value) float_to_fp16_bits(value)\n')
    return (HALF_CONVERSIONS + f'typedef unsigned short {type_name};\n'
            f'#define {load}(value) (fp16_bits_to_float(value) * {float(1.0 / scale)!r}f)\n'
            f'#define {store}(value) float_to_fp16_bits((value) * {float(scale)!r}f)\n')
