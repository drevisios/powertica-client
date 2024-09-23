from setuptools import setup, find_packages

setup(
    name='powertica_client',
    version='0.1.3',
    packages=find_packages(),
    description='A client for Powertica tools',
    install_requires=[
        'azure-ai-ml',
        'azure-identity',
        'pandas',
        'pyarrow',
        'scikit-learn',
        'cloudpickle',
        'scikeras',
        'tensorflow'
    ],
    author='Dionysis Revisios',
    author_email='d.revisios@gmail.com',
    url='https://github.com/yourusername/my_private_package',
)