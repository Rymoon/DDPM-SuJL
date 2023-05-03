from setuptools import find_packages, setup
from pathlib import Path
import re

name = "MyDDPM"
version = "0.1"

p = Path(Path(__file__).parent,name,"__init__.py")
s_ver = f"__version__='{version}'\n"
if not p.exists():
    with open(p,"x") as f:
        f.write(s_ver)
else:
    with open(p,"r") as f:
        s=  f.read()

    o = re.search("__version__[\w\W]*?\n",s)
    if o is not None:
        s =re.sub("__version__[\w\W]*?\n",s_ver,s)
    else:
        s+=f"\n{s_ver}"
    
    with open(p,"w") as f:
        f.write(s)

setup(
    name=name,
    version=version,
    author='Yumeng Ren',
    author_email='tjrym@outlook.com',
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    install_requires=[
    ],
)