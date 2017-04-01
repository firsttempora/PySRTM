from distutils.core import setup
versionstr = '0.1'
setup(
    name='srtm',
    packages=['srtm'], # this must be the same as the name above for PyPI
    version=versionstr,
    description='A package for working with Shuttle Radar Topography Mission data',
    author='firsttempora',
    author_email='first.tempora@gmail.com',
    url='https://github.com/firsttempora/PySRTM', # use the URL to the github repo
    download_url='https://github.com/firsttempora/JLLUtils/tarball/{0}'.format(versionstr), # version must be a git tag
    keywords=['srtm', 'shuttle radar topography mission', 'topography'],
    classifiers=[],
)