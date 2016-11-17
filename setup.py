from distutils.core import setup

setup(
    name='mlp',
    version='0.2',
    description='multi-layer perceptron',
    author='masaponto',
    author_email='masaponto@gmail.com',
    url='masaponto.github.io',
    install_requires=['scikit-learn',
                      'scipy', 'numpy'],
    py_modules=["mlp"],
    package_dir={'': 'src'}
)
