RouteHunter
=======================

Predict whether a chemistry paper describes a **multi-step synthesis route**,
using only its **title and abstract**.

This model is the screening component of **RouteHunter**, a system that
monitors major chemistry journals, pulls new paper titles/abstracts, and
flags papers likely to contain a multi-step synthesis route.


Motivation
----------

Multi-step synthesis routes are valuable for downstream applications
(route mining, retrosynthesis benchmarking, literature review automation),
but they make up only a minority of papers, even in synthesis-focused
journals. Manually screening full texts across many journals doesn't scale.

This project asks: can we predict route-containing papers from title and
abstract alone, cheaply and at scale, so that full-text extraction effort is
spent only on likely candidates?


Installation
------------

.. code-block:: bash

    pip install routehunter


Usage
-----

Train:

.. code-block:: python

    pass
