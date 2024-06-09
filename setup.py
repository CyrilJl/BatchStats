from setuptools import find_packages, setup

# Read version from the __init__.py file
with open("batchstats/__init__.py", "r") as f:
    for line in f:
        if line.startswith("__version__"):
            version = line.strip().split()[-1][1:-1]
            break

setup(name='batchstats',
      version=version,
      author='Cyril Joly',
      description='Efficient batch statistics computation library for Python.',
      long_description=open('README.md').read(),
      long_description_content_type='text/markdown',
      url='https://github.com/CyrilJl/BatchStats',
      packages=find_packages(),
      install_requires=['numpy'])
