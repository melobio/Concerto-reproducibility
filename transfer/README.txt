
1. Data preparation
- Download  the human pancreas dataset  form GSE81076, GSE85241, E-MTAB-5061 and GSE84133.
 -Then merge the above 4 dataset and save it to ./data/hp_rmSchwann_commonCelltype.loom

1.Create tf-record file
python 00_create_tfrecord.py \
  --data ./data/hp_rmSchwann_commonCelltype.loom \
  --output ./tfrecord \
  --batch tech \
  --label celltype

2. Unsupervised training model
python 01_unsupervised_train_test.py \
  --data ./tfrecord \
  --save ./weight \
  --result ./result 

