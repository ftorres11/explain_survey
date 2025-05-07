#!/bin/bash

# Evaluating for R50
# IBA 
#python interpretable_evaluation.py --batch_size 16 --model resnet50 --method IBA  --path_data Evaluation/resnet50/subset_info.json --root_data ../../ILSVRC_2012/val
#python interpretable_evaluation.py --batch_size 16 --model resnet50 --method IBA --lab predicted --path_data Evaluation/resnet50/subset_info.json --root_data ../../ILSVRC_2012/val
#python interpretable_evaluation.py --batch_size 16 --model resnet50 --method IBA  --lab least --path_data Evaluation/resnet50/subset_info.json --root_data ../../ILSVRC_2012/val

# LIME 
python interpretable_evaluation.py --batch_size 16 --model resnet50 --method LIME  --path_data Evaluation/resnet50/subset_info.json --root_data ../../ILSVRC_2012/val
#python interpretable_evaluation.py --batch_size 16 --model resnet50 --method LIME --lab predicted --path_data Evaluation/resnet50/subset_info.json --root_data ../../ILSVRC_2012/val
#python interpretable_evaluation.py --batch_size 16 --model resnet50 --method LIME  --lab least --path_data Evaluation/resnet50/subset_info.json --root_data ../../ILSVRC_2012/val

# RISE
#python interpretable_evaluation.py --batch_size 16 --model resnet50 --method RISE  --path_data Evaluation/resnet50/subset_info.json --root_data ../../ILSVRC_2012/val
#python interpretable_evaluation.py --batch_size 16 --model resnet50 --method RISE --lab predicted --path_data Evaluation/resnet50/subset_info.json --root_data ../../ILSVRC_2012/val
#python interpretable_evaluation.py --batch_size 16 --model resnet50 --method RISE  --lab least --path_data Evaluation/resnet50/subset_info.json --root_data ../../ILSVRC_2012/val

# Fake-CAM
#python interpretable_evaluation.py --batch_size 16 --model resnet50 --method fakecam  --path_data Evaluation/resnet50/subset_info.json --root_data ../../ILSVRC_2012/val
#python interpretable_evaluation.py --batch_size 16 --model resnet50 --method fakecam --lab predicted --path_data Evaluation/resnet50/subset_info.json --root_data ../../ILSVRC_2012/val
#python interpretable_evaluation.py --batch_size 16 --model resnet50 --method fakecam  --lab least --path_data Evaluation/resnet50/subset_info.json --root_data ../../ILSVRC_2012/val

# Grad-CAM
#python interpretable_evaluation.py --batch_size 16 --model resnet50 --method gradcam  --path_data Evaluation/resnet50/subset_info.json --root_data ../../ILSVRC_2012/val
#python interpretable_evaluation.py --batch_size 16 --model resnet50 --method gradcam --lab predicted --path_data Evaluation/resnet50/subset_info.json --root_data ../../ILSVRC_2012/val
#python interpretable_evaluation.py --batch_size 16 --model resnet50 --method gradcam  --lab least --path_data Evaluation/resnet50/subset_info.json --root_data ../../ILSVRC_2012/val

# Grad-CAM++
python interpretable_evaluation.py --batch_size 16 --model resnet50 --method gradcampp  --path_data Evaluation/resnet50/subset_info.json --root_data ../../ILSVRC_2012/val
#python interpretable_evaluation.py --batch_size 16 --model resnet50 --method gradcampp --lab predicted --path_data Evaluation/resnet50/subset_info.json --root_data ../../ILSVRC_2012/val
#python interpretable_evaluation.py --batch_size 16 --model resnet50 --method gradcampp  --lab least --path_data Evaluation/resnet50/subset_info.json --root_data ../../ILSVRC_2012/val

# Score-CAM
#python interpretable_evaluation.py --batch_size 16 --model resnet50 --method scorecam  --path_data Evaluation/resnet50/subset_info.json --root_data ../../ILSVRC_2012/val
#python interpretable_evaluation.py --batch_size 16 --model resnet50 --method scorecam --lab predicted --path_data Evaluation/resnet50/subset_info.json --root_data ../../ILSVRC_2012/val
#python interpretable_evaluation.py --batch_size 16 --model resnet50 --method scorecam  --lab least --path_data Evaluation/resnet50/subset_info.json --root_data ../../ILSVRC_2012/val

# Backprop-Outlier
#python interpretable_evaluation.py --batch_size 16 --model resnet50 --method backprop_outlier  --path_data Evaluation/resnet50/subset_info.json --root_data ../../ILSVRC_2012/val
#python interpretable_evaluation.py --batch_size 16 --model resnet50 --method backprop_outlier --lab predicted --path_data Evaluation/resnet50/subset_info.json --root_data ../../ILSVRC_2012/val
#python interpretable_evaluation.py --batch_size 16 --model resnet50 --method backprop_outlier  --lab least --path_data Evaluation/resnet50/subset_info.json --root_data ../../ILSVRC_2012/val

# Guided Backprop-Outlier
#python interpretable_evaluation.py --batch_size 16 --model resnet50 --method guidedbackprop_outlier  --path_data Evaluation/resnet50/subset_info.json --root_data ../../ILSVRC_2012/val
#python interpretable_evaluation.py --batch_size 16 --model resnet50 --method guidedbackprop_outlier --lab predicted --path_data Evaluation/resnet50/subset_info.json --root_data ../../ILSVRC_2012/val
#python interpretable_evaluation.py --batch_size 16 --model resnet50 --method guidedbackprop_outlier  --lab least --path_data Evaluation/resnet50/subset_info.json --root_data ../../ILSVRC_2012/val

# Backprop-Smooth
#python interpretable_evaluation.py --batch_size 16 --model resnet50 --method backprop_smooth  --path_data Evaluation/resnet50/subset_info.json --root_data ../../ILSVRC_2012/val
#python interpretable_evaluation.py --batch_size 16 --model resnet50 --method backprop_smooth --lab predicted --path_data Evaluation/resnet50/subset_info.json --root_data ../../ILSVRC_2012/val
#python interpretable_evaluation.py --batch_size 16 --model resnet50 --method backprop_smooth  --lab least --path_data Evaluation/resnet50/subset_info.json --root_data ../../ILSVRC_2012/val

# Guided Backprop-Smooth
#python interpretable_evaluation.py --batch_size 16 --model resnet50 --method guidedbackprop_smooth  --path_data Evaluation/resnet50/subset_info.json --root_data ../../ILSVRC_2012/val
#python interpretable_evaluation.py --batch_size 16 --model resnet50 --method guidedbackprop_smooth --lab predicted --path_data Evaluation/resnet50/subset_info.json --root_data ../../ILSVRC_2012/val
#python interpretable_evaluation.py --batch_size 16 --model resnet50 --method guidedbackprop_smooth  --lab least --path_data Evaluation/resnet50/subset_info.json --root_data ../../ILSVRC_2012/val

# Integrated Gradients-Smooth
#python interpretable_evaluation.py --batch_size 16 --model resnet50 --method int_grad_smooth  --path_data Evaluation/resnet50/subset_info.json --root_data ../../ILSVRC_2012/val
#python interpretable_evaluation.py --batch_size 16 --model resnet50 --method int_grad_smooth --lab predicted --path_data Evaluation/resnet50/subset_info.json --root_data ../../ILSVRC_2012/val
#python interpretable_evaluation.py --batch_size 16 --model resnet50 --method int_grad_smooth  --lab least --path_data Evaluation/resnet50/subset_info.json --root_data ../../ILSVRC_2012/val

echo "All Done"!
wait
