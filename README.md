# Asymmetric Student-t model for neuropil decontamination

This repository provides an experimental algorithm to remove neuropil signal in
calcium signal acquired with 2-photon imaging.

To get started, just download or `git clone` this repository and add the `src`
folder to your MATLAB path.

Decontamination is done by the `fit_ast_model` function. You should provide 2
traces (cell ROI and neuropil ROI) and the number of pixels corresponding to
each ROI. Use `help fit_ast_model` or `doc fit_ast_model` in MATLAB to get a
description of all available options.

A description of the probabilistic model is also available
[online](https://basellasermouse.github.io/ast_model/model.html).
