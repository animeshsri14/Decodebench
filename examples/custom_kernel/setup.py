from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="custom_kernel",
    ext_modules=[
        CUDAExtension(
            name="custom_kernel",
            sources=["kernel.cu", "bindings.cpp"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
