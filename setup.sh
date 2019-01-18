#!/bin/sh

PROJECT_DIR="$(cd "." && pwd -P)"

if [[ $1 == cityscapes ]]; then
	echo "Use cityscapes dataset"
	export DATASET="/cluster/scratch/$USER/cityscapes"
	export DATALIST="$PROJECT_DIR/dataset_cityscapes/train_fine.txt"
	export VALDATALIST="$PROJECT_DIR/dataset_cityscapes/val_fine.txt"
	export NR_VAL=500
	export BATCH_SIZE=3
	export IMG_SIZE=719
	export NUM_CLASSES=19
	export LEARNING_RATE=0.0001
	export NUM_ITER=40
else
	echo "Use VOC2012 dataset"
	export DATASET="$PROJECT_DIR/../VOC2012"
	export DATALIST="$PROJECT_DIR/dataset/train.txt"
	export VALDATALIST="$PROJECT_DIR/dataset/val.txt"
	export NR_VAL=1449
	export BATCH_SIZE=10
	export IMG_SIZE=321
	export NUM_CLASSES=21
	export LEARNING_RATE=0.002
	export NUM_ITER=10
fi





