=======
mesonic
=======

.. image:: https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white
   :target: https://github.com/pre-commit/pre-commit
   :alt: pre-commit

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black
    :alt: black

.. image:: https://github.com/dreinsch/mesonic/actions/workflows/main.yml/badge.svg?branch=main
    :target: https://github.com/dreinsch/mesonic/actions/workflows/main.yml
    :alt: github-actions

a Sonification Framework


Installation
============

You can install mesonic with the default backend `sc3nb`_ using pip

    pip install mesonic[sc3nb]

Note that to run the notebooks mesonic also requires a backend like `SuperCollider`_ and/or `pya`_ to be installed.

Developer Installation
======================

To clone and install mesonic with dependencies required for the notebooks run::

    git clone https://github.com/interactive-sonification/mesonic.git
    cd mesonic
    pip install -e ".[notebooks]"

Running Tests & Building
========================

We use `tox`_ for the automation of tasks like testing and building.

Please refer to the tox.ini file for more details


Making Changes & Contributing
=============================

This project uses `pre-commit`_, please make sure to install it before making any
changes::

    pip install pre-commit
    cd mesonic
    pre-commit install


update the hooks to the latest version::

    pre-commit autoupdate

.. _sc3nb: https://github.com/interactive-sonification/sc3nb
.. _SuperCollider: https://github.com/supercollider/supercollider
.. _tox: https://github.com/tox-dev/tox
.. _pre-commit: https://pre-commit.com/
.. _pya: https://github.com/interactive-sonification/pya#installation
