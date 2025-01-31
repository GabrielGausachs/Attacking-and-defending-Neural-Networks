# Attacking-and-defending-Neural-Networks
This repository demonstrates the implementation of adversarial attack on a pretrained neural network for a classification task, followed by the application of a defense mechanism to enhance the network’s robustness. For the attack, we use the Iterative Fast Gradient Sign Method (I-FGSM), an improvement of the Fast Gradient Sign Method (FGSM), which creates adversarial examples by making small adjustments to images. I-FGSM refines this by applying FGSM iteratively with smaller steps, making the attack stronger while keeping changes less noticeable. To defend against these attacks, we applied the Class Label Guided Denoiser (CGD), a model that reduces adversarial noise using a Denoising U- Net (DUNET) structure. The CGD uses the target model’s classification loss to guide the removal of adversarial noise, improving the model’s ability to resist adversarial interference. The goal is to evaluate the effectiveness of the defense in improving the model’s robustness against adversarial attacks.

This is the [Report](https://github.com/GabrielGausachs/Attacking-and-defending-Neural-Networks/blob/main/Report.pdf) of the project.

## Env Setup

This project is developed under python3.8/Anaconda 3. Please check requirements.txt for required packages.

## Dataset

For this project we use the ImageNet dataset, especifically the validation set of ImageNet Large Scale Visual Recognition Challenge 2012 (ILSVRC2012). You can request the dataset in: https://image-net.org/index.php

## Getting Started
To train a model, execute:

`python3 main.py`

In the `Utils/config.py` file you will find the training configuration, paths, parameters and whether you want to do an attack or a defense.

## Citation
- Olga Russakovsky*, Jia Deng*, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang, Andrej Karpathy, Aditya Khosla, Michael Bernstein, Alexander C. Berg and Li Fei-Fei. (* = equal contribution) ImageNet Large Scale Visual Recognition Challenge. arXiv:1409.0575, 2014.
- Fangzhou Liao, Ming Liang, Yinpeng Dong, Tianyu Pang, Xiaolin Hu, and Jun Zhu. Defense against adversarial attacks using high-level representation guided denoiser. In 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1778–1787, 2018.

**This project was proudly created by [@Robert Grgac](https://github.com/Freelancer-cmd), [@Octaviana Cheteles](https://github.com/octaviana3), [@Rebecca Dallabetta](https://github.com/rebedallabetta), and [@Gabriel Gausachs](https://github.com/GabrielGausachs). Huge thanks to everyone involved!**


