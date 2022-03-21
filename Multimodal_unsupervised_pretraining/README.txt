##If you don't want to train the model, you can just load our pre-trained weight and test it directly and step 3 is not required.
1. Data preparation
- Download PBMC160K multimodal dataset from https://doi.org/10.6084/m9.figshare.19390574.v1


2.Create tf-record file
python 00_create_tfrecord.py \
  --RNA-data ./data/multi_RNA_l2.loom \
  --Protein-data ./data/multi_protein_l2.loom \
  --output ./tfrecord \
  --batch batch \
  --label cell_type


3. Unsupervised multimodal training
python 01_multi_unsupervised_train_test.py \
  --task train \
  --RNA-data ./tfrecord/RNA \
  --Protein-data ./tfrecord/Protein \
  --save ./weight \
  --result ./result 

4. Test multimodal model
python 01_multi_unsupervised_train_test.py \
  --task test \
  --RNA-data ./tfrecord/RNA \
  --Protein-data ./tfrecord/Protein \
  --save ./weight \
  --result ./result 
