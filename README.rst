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

Search for a molecule
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   result = app.search("C#CCOC1=C(C=C(C(=C1)N2C(=O)N3CCCCC3=N2)Cl)Cl")

   print(result.found)          # True if a paper or a CASP tool already covers this molecule
   print(result.paper_message)  # e.g. "Found 1 paper(s) reporting a route for this molecule"
   print(result.paper_report)   # pandas.DataFrame: journal, title, year, doi
   print(result.tool_message)   # e.g. "Found 1 tool(s) predicted routes for this molecule"
   print(result.tool_report)    # pandas.DataFrame: tool, result, route

Predict solvability
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   result = app.predict("CC(C)Cc1ccc(cc1)C(C)C(=O)O")
   print(result.to_dataframe())  # predicted solvability probability per CASP tool, with a link to each tool

Monitor recent papers
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   papers = app.monitor(year_min=2023, year_max=2025)
   print(papers)  # pandas.DataFrame, ranked by predicted probability of reporting a synthesis route

CASP tools currently covered
-----------------------------

- `AiZynthFinder <https://github.com/MolecularAI/aizynthfinder>`_
- `SynPlanner <https://github.com/Laboratoire-de-Chemoinformatique/SynPlanner>`_

Project layout
--------------

.. code-block::

   routehunter/        the published package (pip install routehunter) -- everything above lives here
   routehunter_build/  internal tooling for building and maintaining the dataset (not published to PyPI)
   streamlit/          the Streamlit app behind the live demo

Contributing
------------

RouteHunter is a static, curated dataset -- there's currently no way to add or edit data files
directly. All data is updated manually by an administrator after the submitted data has been
validated.

Contributions of any kind are welcome:

- **Scientists**: propose new papers and their target molecules for the dataset, or flag targets
  with no known route that you'd like tested against open-source CASP tools.
- **Developers**: propose having your CASP tool integrated into RouteHunter, for use in route
  search and solvability prediction.

Send contributions to dvzankov@gmail.com, and please include the name you'd like registered as the
contributor on your records.

License
-------

MIT -- see `LICENSE <https://github.com/KagakuLab/RouteHunter/blob/main/LICENSE>`_.