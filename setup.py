import setuptools


def install_requires(filename):
    with open(filename, "r") as f:
        requirements = f.read().splitlines()
    return requirements


setuptools.setup(
    name="deep_table",
    version="0.1.0",
    packages=setuptools.find_packages(),
    include_package_data=True,
    install_requires=install_requires("requirements.txt"),
)
