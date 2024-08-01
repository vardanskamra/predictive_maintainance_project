from setuptools import setup, find_packages

setup(
    name='predictive_maintenance_project',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'numpy',
        'scikit-learn',
        'tensorflow',
        'matplotlib',
        'seaborn',
        'pickle'
    ],
    entry_points={
        'console_scripts': [
            
        ],
    },
)
