"""Shared code of the wheat rust spectral sufficiency study.

io_data      paths, data access, and the conventions of the RGB, MS, and HS views
features     engineered features of every view
common       outer folds, cached feature tables, and metrics
bands        band selection and generic band-set features
learners     the FLAML and TabPFN learners
experiments  the registry of every reported configuration
runner       experiment units and their tidy result tables
stats        paired comparisons on repeated cross-validation
"""
