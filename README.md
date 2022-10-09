# Concerto
This repository contains code and tutorials for the following tasks: query-to-reference task, multimodal self-supervised pretraining task, 
batch correction task and Covid19 task.
## Getting started
In the first step, you will need to download datasets to run each notebook and reproduce the result. 
The download links for datasets are shown in a folder named "data" in the following tasks directories.
## Basic Usage (details see notebooks in each task)
```python
from concerto_function5_3 import *

# preprocessing
adata = preprocessing_rna(adata, batch_key='tech')

# make tfrecord
ref_tf_path = concerto_make_tfrecord(adata_ref,tf_path = save_path + 'tfrecord/ref_tf/',batch_col_name = 'tech')
query_tf_path = concerto_make_tfrecord(adata_query,tf_path = save_path + 'tfrecord/query_tf/',batch_col_name = 'tech')

# integrate
concerto_train_ref_query(ref_tf_path,query_tf_path,weight_path)

# query to reference mapping
ref_embedding,query_embedding,ref_id,query_id = concerto_test_ref_query(weight_path,ref_tf_path,query_tf_path)

# NNvoting according to cells' embedding
query_neighbor,query_prob = knn_classifier(ref_embedding,query_embedding,adata_ref,ref_id,column_name='celltype',k=5)
```

## Query-to-reference task (Fig4)
- Notebook path: transfer/tutorial_transfer.ipynb
- [Readme for the transfer tutorial](transfer/README.txt)
- Description: Human pancreas dataset is used to perform query-to-reference mapping (HP->inDrop).
## Multimodal self-supervised pretraining task (Fig3)
- Cell embeddings are generated by self-supervised pretraining and used for clustering: Multimodal_pretraining/tutorial_multimodal_cluster.ipynb
- Plot attention weight of self-supervised pretraining: Multimodal_pretraining/tutorial_multimodal_print_attention.ipynb
- [Readme for the multimodal self-supervised pretraining tutorial](Multimodal_pretraining/README.txt)
- Description: We use PBMC160K multimodal dataset to self-supervised pretrain a multimodal model, which can be used to perform clustering and extract attention weight.
## Batch correction task (Fig3, extend Fig 1)
- Notebook path: Batch_correction/tutorial_overcorrect.ipynb
- [Readme for the batch corrtection tutorial](Batch_correction/README.txt)
- Description: To justify Concerto’s ability to avoid over-correction, we design a controlled experiment using a simulated dataset.
## Covid19 task (Fig5)
- Note book path: Covid19_task/Covid19_demo.ipynb
- [Readme for the Covid19 tutorial](Covid19_task/README.md)
- Description: Mapping Covid-19 cells against the integrated reference from 10X and DNBelab-C4 data.
## Cross tissue annotation (Fig2g）
- Note book path: Cross_tissue_annotation/tutorial_inter.ipynb
- [Readme for the cross-tissue annotation tutorial](Cross_tissue_annotation/README.txt)
- Description: Cross-tissue prediction on the TMS dataset.
## Dataset used in this study with corresponding links（Supplementary Table 1&7）
- The download link of dataset is shown in [SupplementaryTable1&7_links.xlsx](SupplementaryTable1&7_links.xlsx)
