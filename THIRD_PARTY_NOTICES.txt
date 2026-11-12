# TH-Bench spacing normalization

`src/humanization/datasets/mgtbench.py::process_spaces` reproduces the spacing-normalization routine
from TH-Bench's `th_bench/dataloader.py`, commit
`5c1d99cefee49208206b6fa75d3a628ac9935f86`, derived from MGTBench.
The upstream software license follows. This license does not establish the
license of the underlying essays, Reuters articles or WritingPrompts texts;
their source terms remain separate. Source files are under `data/raw/`;
selected processed text pairs and per-text scores are under `data/processed/`.
Including these files does not change the terms of their source corpora.

MIT License

Copyright (c) 2023 Xinlei He

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


# Resources packaged with the desktop project

The three MGTBench CSVs are the unchanged copies supplied by the author. Official OpAI Parquet files and their source metadata are under data/raw/opai. CMUdict is packaged under data/resources/nltk_data/corpora/cmudict with its original README and dictionary notices. Model weights are not redistributed in the delivery ZIP.
