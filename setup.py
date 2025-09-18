from setuptools import setup, find_packages

def read_requirements():
    """Reads the requirements.txt file and returns a list of dependencies."""
    with open('requirements.txt', 'r') as f:
        return [line.strip() for line in f.readlines() if line.strip()]

setup(
    name='manufacturing_equipment_failure_prediction',
    version='0.1.0',
    packages=find_packages(),
    description="""A machine learning model for predicting manufacturing equipment failure 
    using an Artificial Neural Network (ANN).""",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Mackintosh Odika',
    author_email='odikamackintosh@gmail.com',
    url='https://github.com/Mackintosh007/manufacturing-equipment-failure-prediction-end-to-end-deployment',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    install_requires=read_requirements(),
    python_requires='>=3.8',
)
