import pathlib
import setuptools

setuptools.setup(
    name="stresslog",
    version="1.6.6",
    description="Library for stress calculations from well logs",
    long_description=pathlib.Path("README.md").read_text(),
    long_description_content_type="text/markdown",
    url="https://www.rocklab.in/stresslog",
    author="Arkadeep Ghosh",
    author_email="arkadeep_ghosh@rocklab.in",
    license="AGPL-3.0",
    project_urls={
        "Documentation":"https://stresslog.readthedocs.io/en/stable/",
        "Source":"https://github.com/GeoArkadeep/Stresslog"
    },
    classifiers=[
        "Development Status :: 6 - Mature",
        "License :: OSI Approved :: GNU Affero General Public License v3",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Scientific/Engineering",
    ],
    python_requires=">=3.7,<=3.13",
    install_requires=["pandas>=2.0","numpy","scipy","welly","pint","matplotlib", "plotly","dlisio"],
    packages=setuptools.find_packages(),
    include_package_data=True,    
)

