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
    author=r"Jungeum Kim and Sean O'Hagan",
    url='https://github.com/JungeumKim/conformal_tree',
)
