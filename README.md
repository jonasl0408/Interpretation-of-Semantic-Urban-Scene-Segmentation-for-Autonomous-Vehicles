#  Interpretation-of-Semantic-Urban-Scene-Segmentation-for-Autonomous-Vehicles
# Prerequisites
    * Keras ( recommended version : 2.4.3 )
    * OpenCV for Python
    * Tensorflow ( recommended version : 2.4.1 )
# Download the sample prepared dataset
Download and extract the following:
https://www.kaggle.com/datasets/dansbecker/cityscapes-image-pairs
# Usage
* [Code for Seg-GradCAM method](./seggradcam/seggradcam.py#L118)
* [Code for Seg-GradCAM++ method](./seggradcamplusplus/seggradcam.py#L118)
* [Code for Seg-VanillaGrad method](./Seg-VanillaGrad%26Seg-SmoothGrad/saliency/core/base.py#L234)
* [Code for Seg-SmoothGrad method](./Seg-VanillaGrad%26Seg-SmoothGrad/saliency/core/base.py#L443)
* [Notebook for training 4 segmentation models](./image-segmentation-cityscapes_resnet50.ipynb)
* [Applying Seg-Grad-CAM on pretrained model](./pretrained_models.ipynb) 
# Credits:
[keras_segmentation](https://github.com/divamgupta/image-segmentation-keras)\
[SegGradCAM](https://github.com/kiraving/SegGradCAM)\
[VanillaGrad_SmoothGrad](http://github.com/pair-code/saliency/blob/master)\
[GradCAM++,GradCAM](https://github.com/jacobgil/pytorch-grad-cam)\
[Attention_Unet](https://github.com/lixiaolei1982/Keras-Implementation-of-U-Net-R2U-Net-Attention-U-Net-Attention-R2U-Net.-)\
[DANet](https://github.com/junfu1115/DANet)
# Citation:
@misc{jonasl0408,
  title={Interpretation of Semantic Urban Scene Segmentation for Autonomous Vehicles
},
  author={Jixiang Lei and contributors},
  year={2022},
  publisher={GitHub},
  howpublished={\url{https://github.com/jonasl0408/Interpretation-of-Semantic-Urban-Scene-Segmentation-for-Autonomous-Vehicles}},
}
