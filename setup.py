from setuptools import setup, find_packages

setup(
    name="ivfpqOur",
    version="0.1.0",
    description="Custom ANN method based on HKMeans + Monotonic Neighbors + NNDescent",
    author="your_name",
    author_email="your_email@example.com",
    url="https://github.com/soreak/ivfpqOur",
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=[
        "numpy",
        "scipy",
        "numba",
        "pynndescent",
    ],
    include_package_data=True,
    zip_safe=False,
)
