from setuptools import setup, find_packages

setup(
    name='conformal_tree',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
        'scikit-learn'
    ],
    extras_require={'benchmarks': ['numpy==2.5.4', 'pandas==2.2.3',
                                   'scikit-learn==1.7.2']},
    python_requires='>=3.9',
    author=r"Jungeum Kim and Sean O'Hagan",
    url='https://github.com/JungeumKim/conformal_tree',
)
