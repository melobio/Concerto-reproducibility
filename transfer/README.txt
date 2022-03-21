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

