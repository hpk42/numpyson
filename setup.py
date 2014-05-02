
from setuptools import setup

setup(name="numpyson",
      version="0.1",
      py_modules=['numpyson', 'test_numpyson'],
      install_requires=["numpy", "pandas", "jsonpickle>=0.7"],
)
