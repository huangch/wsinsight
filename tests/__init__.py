"""Marks the test tree as a package.

``tests/regression/*`` import their fixtures as ``tests.regression.conftest``,
which needs this file: without it the name ``tests`` resolves to the top-level
``tests`` package that ``caio`` and ``torchstain`` install into site-packages,
and collection fails with ``No module named 'tests.regression'``.
"""
