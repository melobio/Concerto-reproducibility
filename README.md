# Concerto demo
This repository contains code and tutorials for the following tasks: Transfering task, Multimodal unsupervised pretraining task, 
Batch correction task and Covid19 task.
## Getting started
In the first step, you will need to download datasets to run each notebook and reproduce the result. 
The download link for datasets is shown in a folder named "data" in the following tasks directory.
## Transfering task
- Note book path: transfer/tutorial_transfer.ipynb
- [Readme for the transfer tutorial](transfer/README.txt)
- Description: We use human pancreas dataset to do query-reference mapping (HP->inDrop).
## Multimodal unsupervised pretraining task
- Note book path (Clustering): Multimodal_integration/tutorial_multimodal_cluster.ipynb
- Note book path (Plot attention weight): Multimodal_integration/tutorial_multimodal_print_attention.ipynb
- [Readme for the multimodal unsupervised pretraining tutorial](Multimodal_unsupervised_pretraining/README.txt)
- Description: We use PBMC160K multimodal dataset to unsupervised pretrain the multimodal model and perform clustering and plotting attention weight.
## Batch correction task
- Note book path: Batch_correction/tutorial_overcorrect.ipynb
- [Readme for the batch corrtection tutorial](Batch_correction/README.txt)
- Description: To further justify Concerto’s ability to avoid over-correction, we design a controlled experiment using a simulated dataset and we unsupervised pretrain model and plot UMAP of cell embeddings.
## Covid19 task
- Note book path: Covid19_task/Covid19_demo.ipynb
- [Readme for the Covid19 tutorial](Covid19_task/README.md)
- Description: Query-reference mapping process and downstream analysis for COVID-19 patients.
