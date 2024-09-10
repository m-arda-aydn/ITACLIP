# ITACLIP

## Dependencies
Our code is built on top of [MMSegmentation](https://github.com/open-mmlab/mmsegmentation). Please follow the [instructions](https://mmsegmentation.readthedocs.io/en/main/get_started.html) to install MMSegmentation. We used ```Python=3.9.17```, ```torch=2.0.1```,  ```mmcv=2.1.0```, and ```mmseg=1.2.2``` in our experiments. 

## Datasets
We support four segmentation benchmarks: COCO-Stuff, COCO-Object, Pascal Context, and Pascal VOC. For the dataset preparation, please follow the [MMSeg Dataset Preparation document](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/user_guides/2_dataset_prepare.md). The COCO-Object dataset can be derived from COCO-Stuff by running the following command

```python datasets/cvt_coco_object.py PATH_TO_COCO_STUFF164K -o PATH_TO_COCO164K```
