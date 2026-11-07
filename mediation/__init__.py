"""Causal mediation analysis of attention heads and MLPs in language models.

Mediators are the per-head inputs to the attention output projection and the MLP outputs.
:mod:`mediation.patching` replaces them, :mod:`mediation.estimands` turns patched runs into
NIE, PIE, INT and cross-interactions, and :mod:`mediation.tasks` supplies the prompt pairs
and metrics.
"""
