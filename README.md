# weisfeiler-and-manifold-techniques-svm


## Table of Contents

- [Description](#Description)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)


## Description

Read this [article](https://www.dsi.unive.it/~atorsell/AI/graph/Unfolding.pdf) presenting a way to improve the disciminative power of graph kernels.

Choose one [graph kernel](https://www.dsi.unive.it/~atorsell/AI/graph/kernels.pdf) among

* Shortest-path Kernel
* Graphlet Kernel
* Random Walk Kernel
* Weisfeiler-Lehman Kernel
* Choose one manifold learning technique among

Isomap
Diffusion Maps
Laplacian Eigenmaps
Local Linear Embedding
Compare the performance of an SVM trained on the given kernel, with or without the manifold learning step, on the following datasets:

* [PPI]
* [Shock]

The zip files contain csv files representing the adjacecy matrices of the graphs and of the lavels. the files graphxxx.csv contain the adjaccency matrices, one per file, while the file labels.csv contains all the labels.


In this case I decided to use LocallyLinearEmbedding and Isomap, and to implement the Weisfeiler Lehman graph kernel.

## Installation

A python 3 interpreter is required, along with the following modules:
```bash
pip install numpy
pip install matplotlib
pip install scikit-learn
pip install pandas
```

## Usage

An IDE is required to run the main Jupyter Notebook.

## Contributing

```bash
git clone https://github.com/jgurakuqi/mitsuba-snapshot-tool.git
```

## License

MIT License

Copyright (c) 2021 Elsa Sejdi

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS," WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
