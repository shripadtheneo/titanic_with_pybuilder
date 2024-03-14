#   -*- coding: utf-8 -*-
"""
This module contains the build script for the Titanic project.
"""

from pybuilder.core import use_plugin, init

use_plugin("python.core")
use_plugin("python.unittest")
use_plugin("python.flake8")
use_plugin("python.coverage")
use_plugin("python.distutils")
use_plugin("python.install_dependencies")
use_plugin("pypi:pybuilder_pytest")

name = "titanic"
version = "0.1.0"
default_task = "publish"


@init
def set_properties(project):
    """
    Set properties for the project.

    Args:
        project: The PyBuilder project object.
    """
    # Set Python version
    project.set_property("python_version", "3.11.8")

    # Install dependencies
    project.build_depends_on_requirements("requirements.txt")

    project.get_property("pytest_extra_args").append("-x")

    # Set coverage_break_build to False
    project.set_property("coverage_break_build", False)
