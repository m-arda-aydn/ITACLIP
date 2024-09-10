# ITACLIP

## Dependencies
Our code is built on top of [MMSegmentation](https://github.com/open-mmlab/mmsegmentation). Please follow the [instructions](https://mmsegmentation.readthedocs.io/en/main/get_started.html) to install MMSegmentation. We used ```Python=3.9.17```, ```torch=2.0.1```,  ```mmcv=2.1.0```, and ```mmseg=1.2.2``` in our experiments. 

## Datasets
We support four segmentation benchmarks: COCO-Stuff, COCO-Object, Pascal Context, and Pascal VOC. For the dataset preparation, please follow the [MMSeg Dataset Preparation document](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/user_guides/2_dataset_prepare.md). The COCO-Object dataset can be derived from COCO-Stuff by running the following command

```
python datasets/cvt_coco_object.py PATH_TO_COCO_STUFF164K -o PATH_TO_COCO164K
```

Additional datasets can be seamlessly integrated following the same dataset preparation document. Additional datasets can be seamlessly integrated following the same dataset preparation document. Please modify the dataset (```data_root```) and class name (```name_path```) paths in the config files. 

## LLaMa Generated Texts
For reproducibility, we provide the LLM-generated auxiliary texts. Please update the auxiliary path (```auxiliary_text_path```) in the config files. 
## Evaluation
To evaluate ITACLIP on a dataset, run the following command
```
python eval.py --config ./configs/cfg_{dataset_name}.py
```
## Demo
To evaluate ITACLIP on a single image, run the ```demo.ipynb``` Jupyter Notebook
## Results
With the default configurations, you should achieve the following results (mIoU).

| Dataset               | mIoU  |
| --------------------- | ----- |
| COCO-Stuff            | 27.0  |
| COCO-Object           | 37.7  |
| PASCAL VOC            | 67.9  |
| PASCAL Context        | 37.5  |



