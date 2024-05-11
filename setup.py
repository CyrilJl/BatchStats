from setuptools import find_packages, setup

setup(
    name='batchstats',
    version='0.1',
    author='Cyril Joly',
    description='Description of your package',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/CyrilJl/BatchStats',
    packages=find_packages(),
    install_requires=['numpy'],
)
