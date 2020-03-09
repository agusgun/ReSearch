import setuptools

with open('README.md', 'r') as f:
    long_description = f.read()

setuptools.setup(
    name='hydraml',
    version='0.0.1',
    author='HydraML Team',
    description='HydraML package to be used in machine learning project',
    long_description=long_description,
    url='https://github.com/agusgun/ReSearch',
    packages=setuptools.find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)