RouteHunter
===========

.. image:: https://img.shields.io/pypi/v/routehunter.svg
   :target: https://pypi.org/project/routehunter/
   :alt: PyPI version

.. image:: https://img.shields.io/badge/python-3.12%2B-blue.svg
   :target: https://pypi.org/project/routehunter/
   :alt: Python versions

.. image:: https://img.shields.io/badge/license-MIT-green.svg
   :target: https://github.com/KagakuLab/RouteHunter/blob/main/LICENSE
   :alt: License

.. image:: https://img.shields.io/badge/demo-Streamlit-ff4b4b.svg
   :target: https://routehunter.streamlit.app/
   :alt: Live demo

A system for the collection and distribution of reference information on chemical synthesis routes.

RouteHunter maintains a growing collection of digitized target molecules, each linked to the paper
that reports a multi-step synthesis route to it. This dataset supports benchmarking
**computer-aided synthesis planning (CASP)** tools by how many targets they can solve and, for
solved targets, comparing a tool's predicted route against the one published in the original paper.

Beyond the dataset itself, RouteHunter lets you:

- **search** for a molecule to find whether a route to it has already been published, or solved by
  a CASP tool,
- **predict** a molecule's solvability by open-source CASP tools, and
- **monitor** newly published papers likely to report new synthesis routes.

`Try the live demo → <https://routehunter.streamlit.app/>`_

Installation
------------

.. code-block:: bash

   pip install routehunter

Requires Python 3.12 or later.

Quickstart
----------

RouteHunter's underlying dataset is hosted on `Hugging Face
<https://huggingface.co/datasets/KagakuLab/RouteHunterData>`_. Download it once, then build the app
from it:

.. code-block:: python

   from routehunter.utils import download_app_data
   from routehunter import RouteHunterApp

   app_data_dir = download_app_data(to=".")
   app = RouteHunterApp.from_data_dir(app_data_dir)

   print(app.review())

For examples of the Search, Predict, and Monitor services, see
`RouteHunterApp.ipynb <https://github.com/KagakuLab/RouteHunter/blob/main/RouteHunterApp.ipynb>`_.

Contributing
------------

RouteHunter is a static, curated dataset - there's currently no way to add or edit data files
directly. All data is updated manually by an administrator after the submitted data has been
validated.

Contributions of any kind are welcome:

- **Scientists**: propose new papers and their target molecules for the dataset, or flag targets
  with no known route that you'd like tested against open-source CASP tools.
- **Developers**: propose having your CASP tool integrated into RouteHunter, for use in route
  search and solvability prediction.

Send contributions to dvzankov@gmail.com, and please include the name you'd like registered as the
contributor on your records.