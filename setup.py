from setuptools import setup, find_packages

# Read the content of your requirements.txt file
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='ppcax',  # This will be the name of your package
    version='0.1.0',  # Start with a small version
    author='Ahmed Nabil Atwa',
    author_email='n.is.ccit@gmail.com',
    description='Stochastic dimensionality reduction for finance data using PPCA in JAX framework',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/AI-Ahmed/ppcax',
    packages=find_packages(where='src'),  # Finds packages in 'src' to include
    package_dir={'': 'src'},  # Considers 'src' as the base directory for packages
    install_requires=requirements,
    include_package_data=True,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Financial and Insurance Industry',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'License :: OSI Approved :: Apache License',  # If you have a different license, put that here
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
    python_requires='>=3.9',  # Specify the Python version support
    entry_points={
        'console_scripts': [
            # Here, you can define command-line scripts that you want to make available to users.
            # The syntax is 'script-name = module:function'.
            # 'script-name' is what the user will type to run the script.
            # 'module:function' points to a Python function that will get executed when the script is run.
        ],
    },
)
