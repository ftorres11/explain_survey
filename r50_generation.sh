#!/bin/bash

# Generating for ResNet50
# IBA
#python IBA_generation.py --model resnet50 --store_dir SaliencyMaps/ImageNet/ResNet50/   --path_data data/200classes_5inst.csv --batch_size=1
#python IBA_generation.py --model resnet50 --store_dir SaliencyMaps/ImageNet/ResNet50/   --path_data data/200classes_5inst.csv --batch_size=1 --lab predicted
#python IBA_generation.py --model resnet50 --store_dir SaliencyMaps/ImageNet/ResNet50/   --path_data data/200classes_5inst.csv --batch_size=1 --lab least

# LIME
#python LIME_generation.py --model resnet50 --store_dir SaliencyMaps/ImageNet/ResNet50/   --path_data data/200classes_5inst.csv --root_data ../../ILSVRC_2012/val
#python LIME_generation.py --model resnet50 --store_dir SaliencyMaps/ImageNet/ResNet50/   --path_data data/200classes_5inst.csv --lab predicted
#python LIME_generation.py --model resnet50 --store_dir SaliencyMaps/ImageNet/ResNet50/   --path_data data/200classes_5inst.csv --lab least

# LRP
#python LRP_generation.py --model resnet50 --store_dir SaliencyMaps/ImageNet/ResNet50/   --path_data data/200classes_5inst.csv --root_data ../../ILSVRC_2012/val
#python LRPv2_generation.py --model resnet50 --store_dir SaliencyMaps/ImageNet/ResNet50/   --path_data data/200classes_5inst.csv --root_data ../../ILSVRC_2012/val

#python LRP_generation.py --model resnet50 --store_dir SaliencyMaps/ImageNet/ResNet50/   --path_data data/200classes_5inst.csv --lab predicted
#python LRP_generation.py --model resnet50 --store_dir SaliencyMaps/ImageNet/ResNet50/   --path_data data/200classes_5inst.csv --lab least

# RISE 
#python -m pdb RISE_generation.py --batch_size 2 --store_dir SaliencyMaps/ImageNet/ResNet50/   --path_data data/200classes_5inst.csv --root_data ../../ILSVRC_2012/val
#srun python RISE_generation.py --batch_size 5 --model resnet50 --store_dir SaliencyMaps/ImageNet/ResNet50/   --path_data data/200classes_5inst.csv --lab predicted
#srun python RISE_generation.py --batch_size 5 --model resnet50 --store_dir SaliencyMaps/ImageNet/ResNet50/   --path_data data/200classes_5inst.csv --lab least

# Grad-CAM
#python -m pdb cam_generation.py --batch_size 10 --model resnet50 --method gradcam --store_dir SaliencyMaps/ImageNet/ResNet50/   --path_data data/200classes_5inst.csv --root_data ../../ILSVRC_2012/val
#srun python cam_generation.py --batch_size 10 --model resnet50 --method gradcam --store_dir SaliencyMaps/ImageNet/ResNet50/  --lab predicted --path_data data/200classes_5inst.csv

#srun python cam_generation.py --batch_size 10 --model resnet50 --method gradcam --store_dir SaliencyMaps/ImageNet/ResNet50/  --lab least --path_data data/200classes_5inst.csv
# Grad-CAM++
#python cam_generation.py --batch_size 10 --model resnet50 --method gradcampp --store_dir SaliencyMaps/ImageNet/ResNet50/   --path_data data/200classes_5inst.csv --root_data ../../ILSVRC_2012/val
#srun python cam_generation.py --batch_size 1 --model resnet50 --method gradcampp --store_dir SaliencyMaps/ImageNet/ResNet50/  --lab predicted
#srun python cam_generation.py --batch_size 1 --model resnet50 --method gradcampp --store_dir SaliencyMaps/ImageNet/ResNet50/  --lab least

# Score-CAM
#python cam_generation.py --batch_size 4 --model resnet50 --method scorecam --store_dir SaliencyMaps/ImageNet/ResNet50/   --path_data data/200classes_5inst.csv --root_data ../../ILSVRC_2012/val
#srun python cam_generation.py --batch_size 1 --model resnet50 --method scorecam --store_dir SaliencyMaps/ImageNet/ResNet50/  --lab predicted
#srun python cam_generation.py --batch_size 1 --model resnet50 --method scorecam --store_dir SaliencyMaps/ImageNet/ResNet50/  --lab least

# Backprop - Jacbogil
#srun python gradient_generation.py --batch_size 1 --model resnet50 --method backprop --store_dir SaliencyMaps/ImageNet/ResNet50/  
#srun python gradient_generation.py --batch_size 1 --model resnet50 --method backprop --store_dir SaliencyMaps/ImageNet/ResNet50/  --lab predicted
#srun python gradient_generation.py --batch_size 1 --model resnet50 --method backprop --store_dir SaliencyMaps/ImageNet/ResNet50/  --lab least


# SHAP - Jacobgil
python -m pdb SHAP_generation.py --batch_size 3 --model resnet50 --store_dir SaliencyMaps/ImageNet/ResNet50/   --path_data data/200classes_5inst.csv --root_data ../../ILSVRC_2012/val
#srun python gradient_generation.py --batch_size 1 --model resnet50 --method guidedbackprop --store_dir SaliencyMaps/ImageNet/ResNet50/  --lab predicted
#srun python gradient_generation.py --batch_size 1 --model resnet50 --method guidedbackprop --store_dir SaliencyMaps/ImageNet/ResNet50/  --lab least

# Guided Backprop - Jacobgil
#python gradient_generation.py --batch_size 1 --model resnet50 --method guidedbackprop  --store_dir SaliencyMaps/ImageNet/ResNet50/   --path_data data/200classes_5inst.csv --root_data ../../ILSVRC_2012/val
#srun python gradient_generation.py --batch_size 1 --model resnet50 --method guidedbackprop --store_dir SaliencyMaps/ImageNet/ResNet50/  --lab predicted
#srun python gradient_generation.py --batch_size 1 --model resnet50 --method guidedbackprop --store_dir SaliencyMaps/ImageNet/ResNet50/  --lab least

# Backprop - Smoothgrad denorm
#python gradient_generation.py --batch_size 1 --model resnet50 --method backprop --store_dir SaliencyMaps/ImageNet/ResNet50/  --norm smooth
#srun python gradient_generation.py --batch_size 1 --model resnet50 --method backprop --store_dir SaliencyMaps/ImageNet/ResNet50/  --lab predicted --norm smooth
#srun python gradient_generation.py --batch_size 1 --model resnet50 --method backprop --store_dir SaliencyMaps/ImageNet/ResNet50/  --lab least --norm smooth

# Guided Backprop - Smoothgrad denorm
#python gradient_generation.py --batch_size 1 --model resnet50 --method guidedbackprop --norm smooth --store_dir SaliencyMaps/ImageNet/ResNet50/   --path_data data/200classes_5inst.csv --root_data ../../ILSVRC_2012/val
#srun python gradient_generation.py --batch_size 1 --model resnet50 --method guidedbackprop --store_dir SaliencyMaps/ImageNet/ResNet50/  --lab predicted --norm smooth
#srun python gradient_generation.py --batch_size 1 --model resnet50 --method guidedbackprop --store_dir SaliencyMaps/ImageNet/ResNet50/  --lab least --norm smooth

# Integrated Gradients-smooth
#python integrated_grad_generation.py --batch_size 1 --model resnet50 --method int_grad --store_dir SaliencyMaps/ImageNet/ResNet50/  --norm smooth
#python integrated_grad_generation.py --batch_size 1 --model resnet50 --method int_grad --store_dir SaliencyMaps/ImageNet/ResNet50/  --lab predicted --norm smooth
#python integrated_grad_generation.py --batch_size 1 --model resnet50 --method int_grad --store_dir SaliencyMaps/ImageNet/ResNet50/  --lab least --norm smooth

# Integrated Gradients-Jacobgil
#python integrated_grad_generation.py --batch_size 1 --model resnet50 --method int_grad --store_dir SaliencyMaps/ImageNet/ResNet50/  
#python integrated_grad_generation.py --batch_size 1 --model resnet50 --method int_grad --store_dir SaliencyMaps/ImageNet/ResNet50/  --lab predicted 
#python integrated_grad_generation.py --batch_size 1 --model resnet50 --method int_grad --store_dir SaliencyMaps/ImageNet/ResNet50/  --lab least 

echo "All Done"!
wait
