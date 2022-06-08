To further justify Concerto’s ability to avoid over-correction, we design a controlled experiment using a simulated dataset.
We remove Group2 cells from all 6 batches except Batch1 to construct partially overlapping dataset before integration.
If Group2 cells can be separated from other cell types, it indicates that Concerto can avoid over-correction. 

1. Data preparation
- Download the simulated dataset from https://doi.org/10.6084/m9.figshare.20025374.v1
 -Then save expBatch1_woGroup2.loom in ./data

2. Unsupervised training model and plot UMAP of cell embeddings
- Run tutorial_overcorrect.ipynb
