from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="mygemm",
    version="0.0.1",
    ext_modules=[
        CUDAExtension(
            name="mygemm_cuda",  # the built .so/.pyd
            sources=["csrc/mygemm.cpp", "csrc/mygemm_kernels.cu"],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "--use_fast_math"]
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    packages=["mygemm"],
)
