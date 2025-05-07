#!/bin/bash

# Evaluating for VGG16
# IBA 
#python interpretable_evaluation.py --batch_size 16 --model vgg16_bn --method IBA  --path_data Evaluation/vgg16_bn/subset_info.json --root_data ../../ILSVRC_2012/val --path_saliency SaliencyMaps/ImageNet/VGG16_BN
#python interpretable_evaluation.py --batch_size 16 --model vgg16_bn --method IBA --lab predicted --path_data Evaluation/vgg16_bn/subset_info.json --root_data ../../ILSVRC_2012/val --path_saliency SaliencyMaps/ImageNet/VGG16_BN
#python interpretable_evaluation.py --batch_size 16 --model vgg16_bn --method IBA  --lab least --path_data Evaluation/vgg16_bn/subset_info.json --root_data ../../ILSVRC_2012/val --path_saliency SaliencyMaps/ImageNet/VGG16_BN

# LIME 
#python interpretable_evaluation.py --batch_size 16 --model vgg16_bn --method LIME  --path_data Evaluation/vgg16_bn/subset_info.json --root_data ../../ILSVRC_2012/val --path_saliency SaliencyMaps/ImageNet/VGG16_BN
#python interpretable_evaluation.py --batch_size 16 --model vgg16_bn --method LIME --lab predicted --path_data Evaluation/vgg16_bn/subset_info.json --root_data ../../ILSVRC_2012/val --path_saliency SaliencyMaps/ImageNet/VGG16_BN
#python interpretable_evaluation.py --batch_size 16 --model vgg16_bn --method LIME  --lab least --path_data Evaluation/vgg16_bn/subset_info.json --root_data ../../ILSVRC_2012/val --path_saliency SaliencyMaps/ImageNet/VGG16_BN

# RISE
python interpretable_evaluation.py --batch_size 16 --model vgg16_bn --method RISE  --path_data Evaluation/vgg16_bn/subset_info.json --root_data ../../ILSVRC_2012/val --path_saliency SaliencyMaps/ImageNet/VGG16_BN
#python interpretable_evaluation.py --batch_size 16 --model vgg16_bn --method RISE --lab predicted --path_data Evaluation/vgg16_bn/subset_info.json --root_data ../../ILSVRC_2012/val --path_saliency SaliencyMaps/ImageNet/VGG16_BN
#python interpretable_evaluation.py --batch_size 16 --model vgg16_bn --method RISE  --lab least --path_data Evaluation/vgg16_bn/subset_info.json --root_data ../../ILSVRC_2012/val --path_saliency SaliencyMaps/ImageNet/VGG16_BN

# Fake-CAM
#python interpretable_evaluation.py --batch_size 16 --model vgg16_bn --method fakecam  --path_data Evaluation/vgg16_bn/subset_info.json --root_data ../../ILSVRC_2012/val --path_saliency SaliencyMaps/ImageNet/VGG16_BN
#python interpretable_evaluation.py --batch_size 16 --model vgg16_bn --method fakecam --lab predicted --path_data Evaluation/vgg16_bn/subset_info.json --root_data ../../ILSVRC_2012/val --path_saliency SaliencyMaps/ImageNet/VGG16_BN
#python interpretable_evaluation.py --batch_size 16 --model vgg16_bn --method fakecam  --lab least --path_data Evaluation/vgg16_bn/subset_info.json --root_data ../../ILSVRC_2012/val --path_saliency SaliencyMaps/ImageNet/VGG16_BN

# Grad-CAM
python interpretable_evaluation.py --batch_size 16 --model vgg16_bn --method gradcam  --path_data Evaluation/vgg16_bn/subset_info.json --root_data ../../ILSVRC_2012/val --path_saliency SaliencyMaps/ImageNet/VGG16_BN
#python interpretable_evaluation.py --batch_size 16 --model vgg16_bn --method gradcam --lab predicted --path_data Evaluation/vgg16_bn/subset_info.json --root_data ../../ILSVRC_2012/val --path_saliency SaliencyMaps/ImageNet/VGG16_BN
#python interpretable_evaluation.py --batch_size 16 --model vgg16_bn --method gradcam  --lab least --path_data Evaluation/vgg16_bn/subset_info.json --root_data ../../ILSVRC_2012/val --path_saliency SaliencyMaps/ImageNet/VGG16_BN

# Grad-CAM++
#python interpretable_evaluation.py --batch_size 16 --model vgg16_bn --method gradcampp  --path_data Evaluation/vgg16_bn/subset_info.json --root_data ../../ILSVRC_2012/val --path_saliency SaliencyMaps/ImageNet/VGG16_BN
#python interpretable_evaluation.py --batch_size 16 --model vgg16_bn --method gradcampp --lab predicted --path_data Evaluation/vgg16_bn/subset_info.json --root_data ../../ILSVRC_2012/val --path_saliency SaliencyMaps/ImageNet/VGG16_BN
#python interpretable_evaluation.py --batch_size 16 --model vgg16_bn --method gradcampp  --lab least --path_data Evaluation/vgg16_bn/subset_info.json --root_data ../../ILSVRC_2012/val --path_saliency SaliencyMaps/ImageNet/VGG16_BN

# Score-CAM
#python interpretable_evaluation.py --batch_size 16 --model vgg16_bn --method scorecam  --path_data Evaluation/vgg16_bn/subset_info.json --root_data ../../ILSVRC_2012/val --path_saliency SaliencyMaps/ImageNet/VGG16_BN
#python interpretable_evaluation.py --batch_size 16 --model vgg16_bn --method scorecam --lab predicted --path_data Evaluation/vgg16_bn/subset_info.json --root_data ../../ILSVRC_2012/val --path_saliency SaliencyMaps/ImageNet/VGG16_BN
#python interpretable_evaluation.py --batch_size 16 --model vgg16_bn --method scorecam  --lab least --path_data Evaluation/vgg16_bn/subset_info.json --root_data ../../ILSVRC_2012/val --path_saliency SaliencyMaps/ImageNet/VGG16_BN

# Backprop-Outlier
#python interpretable_evaluation.py --batch_size 16 --model vgg16_bn --method backprop_outlier  --path_data Evaluation/vgg16_bn/subset_info.json --root_data ../../ILSVRC_2012/val --path_saliency SaliencyMaps/ImageNet/VGG16_BN
#python interpretable_evaluation.py --batch_size 16 --model vgg16_bn --method backprop_outlier --lab predicted --path_data Evaluation/vgg16_bn/subset_info.json --root_data ../../ILSVRC_2012/val --path_saliency SaliencyMaps/ImageNet/VGG16_BN
#python interpretable_evaluation.py --batch_size 16 --model vgg16_bn --method backprop_outlier  --lab least --path_data Evaluation/vgg16_bn/subset_info.json --root_data ../../ILSVRC_2012/val --path_saliency SaliencyMaps/ImageNet/VGG16_BN

# Guided Backprop-Outlier
#python interpretable_evaluation.py --batch_size 16 --model vgg16_bn --method guidedbackprop_outlier  --path_data Evaluation/vgg16_bn/subset_info.json --root_data ../../ILSVRC_2012/val --path_saliency SaliencyMaps/ImageNet/VGG16_BN
#python interpretable_evaluation.py --batch_size 16 --model vgg16_bn --method guidedbackprop_outlier --lab predicted --path_data Evaluation/vgg16_bn/subset_info.json --root_data ../../ILSVRC_2012/val --path_saliency SaliencyMaps/ImageNet/VGG16_BN
#python interpretable_evaluation.py --batch_size 16 --model vgg16_bn --method guidedbackprop_outlier  --lab least --path_data Evaluation/vgg16_bn/subset_info.json --root_data ../../ILSVRC_2012/val --path_saliency SaliencyMaps/ImageNet/VGG16_BN

# Backprop-Smooth
#python interpretable_evaluation.py --batch_size 16 --model vgg16_bn --method backprop_smooth  --path_data Evaluation/vgg16_bn/subset_info.json --root_data ../../ILSVRC_2012/val --path_saliency SaliencyMaps/ImageNet/VGG16_BN
#python interpretable_evaluation.py --batch_size 16 --model vgg16_bn --method backprop_smooth --lab predicted --path_data Evaluation/vgg16_bn/subset_info.json --root_data ../../ILSVRC_2012/val --path_saliency SaliencyMaps/ImageNet/VGG16_BN
#python interpretable_evaluation.py --batch_size 16 --model vgg16_bn --method backprop_smooth  --lab least --path_data Evaluation/vgg16_bn/subset_info.json --root_data ../../ILSVRC_2012/val --path_saliency SaliencyMaps/ImageNet/VGG16_BN

# Guided Backprop-Smooth
#python interpretable_evaluation.py --batch_size 16 --model vgg16_bn --method guidedbackprop_smooth  --path_data Evaluation/vgg16_bn/subset_info.json --root_data ../../ILSVRC_2012/val --path_saliency SaliencyMaps/ImageNet/VGG16_BN
#python interpretable_evaluation.py --batch_size 16 --model vgg16_bn --method guidedbackprop_smooth --lab predicted --path_data Evaluation/vgg16_bn/subset_info.json --root_data ../../ILSVRC_2012/val --path_saliency SaliencyMaps/ImageNet/VGG16_BN
#python interpretable_evaluation.py --batch_size 16 --model vgg16_bn --method guidedbackprop_smooth  --lab least --path_data Evaluation/vgg16_bn/subset_info.json --root_data ../../ILSVRC_2012/val --path_saliency SaliencyMaps/ImageNet/VGG16_BN

# Integrated Gradients-Smooth
#python interpretable_evaluation.py --batch_size 16 --model vgg16_bn --method int_grad_smooth  --path_data Evaluation/vgg16_bn/subset_info.json --root_data ../../ILSVRC_2012/val --path_saliency SaliencyMaps/ImageNet/VGG16_BN
#python interpretable_evaluation.py --batch_size 16 --model vgg16_bn --method int_grad_smooth --lab predicted --path_data Evaluation/vgg16_bn/subset_info.json --root_data ../../ILSVRC_2012/val --path_saliency SaliencyMaps/ImageNet/VGG16_BN
#python interpretable_evaluation.py --batch_size 16 --model vgg16_bn --method int_grad_smooth  --lab least --path_data Evaluation/vgg16_bn/subset_info.json --root_data ../../ILSVRC_2012/val --path_saliency SaliencyMaps/ImageNet/VGG16_BN

echo "All Done"!
wait
