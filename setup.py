from distutils.core import setup

setup(
    name='mlp',
    version='0.1',
    description='multi-layer perceptron',
    author='masaponto',
    author_email='masaponto@gmail.com',
    url='masaponto.github.io',
    install_requires=['scikit-learn==0.17.1', 'scipy==0.17.0', 'numpy==1.10.4'],
    py_modules = ["mlp.mlp"],
    package_dir = {'': 'src'}
)
