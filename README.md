# Uncertainty-aware principal component analysis

![Build Status](https://github.com/grtlr/uapca/workflows/build/badge.svg)
![npm](https://img.shields.io/npm/v/uapca)
![GitHub](https://img.shields.io/github/license/grtlr/uapca)

This is an implementation of uncertainty-aware principal component analysis, which generalizes PCA to work on probability distributions.

![Teaser](https://raw.githubusercontent.com/grtlr/uapca/master/teaser.gif)

You can find a preprint of our paper at [arXiv:1905.01127](https://arxiv.org/abs/1905.01127) or on my [personal website](https://www.jgoertler.com).
We also extracted means and covariances from the [*student grade* dataset](https://raw.githubusercontent.com/grtlr/uapca/master/data/student_grades.json).

## Development

The dependencies can be install using `yarn`:

```bash
yarn install
```

Builds can be prepared using:

```bash
yarn run build
yarn run dev # watches for changes
```

Run tests:

```bash
yarn run test
```
    
To perform linter checks you there is:

```bash
yarn run lint
yarn run lint-fix # tries to fix some of the warnings
```

## Citation

To cite this work, you can use the following BibTex entry:

```bibtex
@article{UaPCA:2020,
  author    = {Jochen Görtler and Thilo Spinner and Dirk Streeb and Daniel Weiskopf and Oliver Deussen},
  title     = {Uncertainty-Aware Principal Component Analysis},
  journal   = {IEEE Transactions on Visualization and Computer Graphics},
  year      = {2020},
  pages     = {to appear}
}
```
