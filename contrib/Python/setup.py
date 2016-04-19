from distutils.core import setup

setup(name='CNTK',
      version='1.5',
      description='CNTK',
      author='Microsoft',
      author_email='deepblis@microsoft.com',
      url='cntk.ai',
      packages=['cntk', 'cntk.examples', 'cntk.tests', 'cntk.utils', 'cntk.ops'],
      package_data={'cntk': ['templates/*.cntk']}
     )
