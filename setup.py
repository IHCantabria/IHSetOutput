from setuptools import setup, find_packages

setup(
    name='IHSetOutput',
    version='0.6.0',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'numpy',
        'xarray',
        'datetime',
        'scipy',
        'IHSetUtils @ git+https://github.com/IHCantabria/IHSetUtils.git'
    ],
    author='Lucas de Freitas Pereira',
    author_email='lucas.defreitas@unican.es',
    description='IH-SET output module',
    url='https://github.com/IHCantabria/IHSetOutput',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
)