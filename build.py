#   -*- coding: utf-8 -*-
from pybuilder.core import use_plugin, init

use_plugin("python.core")
use_plugin("python.unittest")
use_plugin("python.flake8")
#use_plugin("python.coverage")
use_plugin("python.distutils")


name = "titanic"
version = "0.1.0"
#default_task = ["install_dependencies", "publish"]
default_task = "publish"


@init
def set_properties(project):
    # Set Python version 
    project.set_property("python_version", "3.8")

    # Install dependencies
    project.depends_on_requirements("requirements.txt")
