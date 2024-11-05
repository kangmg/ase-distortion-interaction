from setuptools import setup, find_namespace_packages

setup(
    name='ase-dias',
    version='0.0.1',
    author='Kang mingi',
    author_email='kangmg@korea.ac.kr',
    description='An ASE-compatible plugin for distortion interaction analysis.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',  
    url='https://github.com/kangmg/ase-distortion-interaction',
    keywords=['chemistry','computational chemistry','machine learning'],
    include_package_data=True,
    packages=find_namespace_packages(), 
    install_requires=[
        'matplotlib',
        'numpy',
        'requests',
        'ase>=3.22.1',
        'torch>=2.2.1',
    ],
    classifiers=[ 
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Chemistry'
    ],
    python_requires='>=3.10.0',
)
