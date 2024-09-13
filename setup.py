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
    author_email='to be filled in',
    description='to be filled in',
    url='to be filled in (github repo)',  # Optional, if you host it on GitHub
)