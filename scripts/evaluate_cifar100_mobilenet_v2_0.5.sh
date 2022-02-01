#!/bin/sh
python train.py --model ai87netmobilenetv2cifar100_m0_5 --dataset CIFAR100 --evaluate --device MAX78000 --exp-load-weights-from ../ai8x-synthesis/trained/ai87-cifar100-mobilenet-v2-0.5-qat8-q.pth.tar -8 --use-bias "$@"
