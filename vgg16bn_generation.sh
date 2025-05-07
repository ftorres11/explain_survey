#!/bin/bash

# Generating for VGG16_BN
# IBA
#python IBA_generation.py --model vgg16_bn --store_dir SaliencyMaps/ImageNet/VGG16_BN/   --path_data data/200classes_5inst.csv --batch_size=1
#python IBA_generation.py --model vgg16_bn --store_dir SaliencyMaps/ImageNet/VGG16_BN/   --path_data data/200classes_5inst.csv --batch_size=1 --lab predicted
#python IBA_generation.py --model vgg16_bn --store_dir SaliencyMaps/ImageNet/VGG16_BN/   --path_data data/200classes_5inst.csv --batch_size=1 --lab least

# LIME
#python LIME_generation.py --model vgg16_bn --store_dir SaliencyMaps/ImageNet/VGG16_BN/   --path_data data/200classes_5inst.csv --root_data ../../ILSVRC_2012/val
#python LIME_generation.py --model vgg16_bn --store_dir SaliencyMaps/ImageNet/VGG16_BN/   --path_data data/200classes_5inst.csv --lab predicted
#python LIME_generation.py --model vgg16_bn --store_dir SaliencyMaps/ImageNet/VGG16_BN/   --path_data data/200classes_5inst.csv --lab least

# LRP
#python LRP_generation.py --model vgg16_bn --store_dir SaliencyMaps/ImageNet/VGG16_BN/   --path_data data/200classes_5inst.csv --root_data ../../ILSVRC_2012/val
#python LRPv2_generation.py --model vgg16_bn --store_dir SaliencyMaps/ImageNet/VGG16_BN/   --path_data data/200classes_5inst.csv --root_data ../../ILSVRC_2012/val

#python LRP_generation.py --model vgg16_bn --store_dir SaliencyMaps/ImageNet/VGG16_BN/   --path_data data/200classes_5inst.csv --lab predicted
#python LRP_generation.py --model vgg16_bn --store_dir SaliencyMaps/ImageNet/VGG16_BN/   --path_data data/200classes_5inst.csv --lab least

# RISE 
#python RISE_generation.py --batch_size 1 --store_dir SaliencyMaps/ImageNet/VGG16_BN/   --path_data data/200classes_5inst.csv --root_data ../../ILSVRC_2012/val
#srun python RISE_generation.py --batch_size 5 --model vgg16_bn --store_dir SaliencyMaps/ImageNet/VGG16_BN/   --path_data data/200classes_5inst.csv --lab predicted
#srun python RISE_generation.py --batch_size 5 --model vgg16_bn --store_dir SaliencyMaps/ImageNet/VGG16_BN/   --path_data data/200classes_5inst.csv --lab least

# Grad-CAM
#python cam_generation.py --batch_size 10 --model vgg16_bn --method gradcam --store_dir SaliencyMaps/ImageNet/VGG16_BN/   --path_data data/200classes_5inst.csv --root_data ../../ILSVRC_2012/val
#srun python cam_generation.py --batch_size 10 --model vgg16_bn --method gradcam --store_dir SaliencyMaps/ImageNet/VGG16_BN/  --lab predicted --path_data data/200classes_5inst.csv

#srun python cam_generation.py --batch_size 10 --model vgg16_bn --method gradcam --store_dir SaliencyMaps/ImageNet/VGG16_BN/  --lab least --path_data data/200classes_5inst.csv


# Grad-CAM++
#python cam_generation.py --batch_size 10 --model vgg16_bn --method gradcampp --store_dir SaliencyMaps/ImageNet/VGG16_BN/   --path_data data/200classes_5inst.csv --root_data ../../ILSVRC_2012/val
#srun python cam_generation.py --batch_size 1 --model vgg16_bn --method gradcampp --store_dir SaliencyMaps/ImageNet/VGG16_BN/  --lab predicted
#srun python cam_generation.py --batch_size 1 --model vgg16_bn --method gradcampp --store_dir SaliencyMaps/ImageNet/VGG16_BN/  --lab least

# Score-CAM
#python cam_generation.py --batch_size 4 --model vgg16_bn --method scorecam --store_dir SaliencyMaps/ImageNet/VGG16_BN/   --path_data data/200classes_5inst.csv --root_data ../../ILSVRC_2012/val
#srun python cam_generation.py --batch_size 1 --model vgg16_bn --method scorecam --store_dir SaliencyMaps/ImageNet/VGG16_BN/  --lab predicted
#srun python cam_generation.py --batch_size 1 --model vgg16_bn --method scorecam --store_dir SaliencyMaps/ImageNet/VGG16_BN/  --lab least

# Backprop - Jacbogil
#srun python gradient_generation.py --batch_size 1 --model vgg16_bn --method backprop --store_dir SaliencyMaps/ImageNet/VGG16_BN/  
#srun python gradient_generation.py --batch_size 1 --model vgg16_bn --method backprop --store_dir SaliencyMaps/ImageNet/VGG16_BN/  --lab predicted
#srun python gradient_generation.py --batch_size 1 --model vgg16_bn --method backprop --store_dir SaliencyMaps/ImageNet/VGG16_BN/  --lab least


# SHAP - Jacobgil
python  SHAP_generation.py --batch_size 5 --model vgg16_bn --store_dir SaliencyMaps/ImageNet/VGG16_BN/   --path_data data/200classes_5inst.csv --root_data ../../ILSVRC_2012/val
#srun python gradient_generation.py --batch_size 1 --model vgg16_bn --method guidedbackprop --store_dir SaliencyMaps/ImageNet/VGG16_BN/  --lab predicted
#srun python gradient_generation.py --batch_size 1 --model vgg16_bn --method guidedbackprop --store_dir SaliencyMaps/ImageNet/VGG16_BN/  --lab least

# Guided Backprop - Jacobgil
#python gradient_generation.py --batch_size 1 --model vgg16_bn --method guidedbackprop  --store_dir SaliencyMaps/ImageNet/VGG16_BN/   --path_data data/200classes_5inst.csv --root_data ../../ILSVRC_2012/val
#srun python gradient_generation.py --batch_size 1 --model vgg16_bn --method guidedbackprop --store_dir SaliencyMaps/ImageNet/VGG16_BN/  --lab predicted
#srun python gradient_generation.py --batch_size 1 --model vgg16_bn --method guidedbackprop --store_dir SaliencyMaps/ImageNet/VGG16_BN/  --lab least

# Backprop - Smoothgrad denorm
#python gradient_generation.py --batch_size 1 --model vgg16_bn --method backprop --store_dir SaliencyMaps/ImageNet/VGG16_BN/  --norm smooth
#srun python gradient_generation.py --batch_size 1 --model vgg16_bn --method backprop --store_dir SaliencyMaps/ImageNet/VGG16_BN/  --lab predicted --norm smooth
#srun python gradient_generation.py --batch_size 1 --model vgg16_bn --method backprop --store_dir SaliencyMaps/ImageNet/VGG16_BN/  --lab least --norm smooth

# Guided Backprop - Smoothgrad denorm
#python gradient_generation.py --batch_size 1 --model vgg16_bn --method guidedbackprop --norm smooth --store_dir SaliencyMaps/ImageNet/VGG16_BN/   --path_data data/200classes_5inst.csv --root_data ../../ILSVRC_2012/val
#srun python gradient_generation.py --batch_size 1 --model vgg16_bn --method guidedbackprop --store_dir SaliencyMaps/ImageNet/VGG16_BN/  --lab predicted --norm smooth
#srun python gradient_generation.py --batch_size 1 --model vgg16_bn --method guidedbackprop --store_dir SaliencyMaps/ImageNet/VGG16_BN/  --lab least --norm smooth

# Integrated Gradients-smooth
#python integrated_grad_generation.py --batch_size 1 --model vgg16_bn --method int_grad --store_dir SaliencyMaps/ImageNet/VGG16_BN/  --norm smooth
#python integrated_grad_generation.py --batch_size 1 --model vgg16_bn --method int_grad --store_dir SaliencyMaps/ImageNet/VGG16_BN/  --lab predicted --norm smooth
#python integrated_grad_generation.py --batch_size 1 --model vgg16_bn --method int_grad --store_dir SaliencyMaps/ImageNet/VGG16_BN/  --lab least --norm smooth

# Integrated Gradients-Jacobgil
#python integrated_grad_generation.py --batch_size 1 --model vgg16_bn --method int_grad --store_dir SaliencyMaps/ImageNet/VGG16_BN/  
#python integrated_grad_generation.py --batch_size 1 --model vgg16_bn --method int_grad --store_dir SaliencyMaps/ImageNet/VGG16_BN/  --lab predicted 
#python integrated_grad_generation.py --batch_size 1 --model vgg16_bn --method int_grad --store_dir SaliencyMaps/ImageNet/VGG16_BN/  --lab least 

echo "All Done"!
wait
