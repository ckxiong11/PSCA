# -*- coding: utf-8 -*-
"""
@Auth ： 挂科边缘
@File ：trian.py
@IDE ：PyCharm
@Motto:学习新思想，争做新青年
@Email ：179958974@qq.com
"""
import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    # model.load('yolo11n.pt') # 加载预训练权重,改进或者做对比实验时候不建议打开，因为用预训练模型整体精度没有很明显的提升
    model = YOLO(model='yolov8s.yaml')
    model.train(data=r'F:/ckx/ckx/model1/PSCA-YOLO/cfg/datasets/fractured.yaml',
                imgsz=640,
                epochs=100,
                optimizer='SGD',
                batch=4,
                workers=4,
                device='0',
                )