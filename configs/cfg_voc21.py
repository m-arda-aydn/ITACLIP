_base_ = './base_config.py'

dataset_name = 'voc21'
# model settings
model = dict(
    type='ITACLIP_Segmentor',
    model_name = 'ViT-B/16',
    img_engineering = True,
    dataset_name = dataset_name,
    auxiliary_text_path = f'/ITACLIP/llama_generated_texts/{dataset_name}_definitions.txt',
    slide_stride = 28,
    attn_self = True,
    def_coefficient = 0.05,
    img_eng_coefficient = 0.7,
    pamr_steps = 10,
    device = 'cuda:0',
    name_path=f'/ITACLIP/configs/cls_{dataset_name}.txt',
    logit_scale = 60,
    prob_thd = 0.1,
    area_thd = 0.1
)

# dataset settings
dataset_type = 'PascalVOCDataset'
data_root = ' '

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(2048, 336), keep_ratio=True),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='JPEGImages', seg_map_path='SegmentationClass'),
        ann_file='ImageSets/Segmentation/val.txt',
        pipeline=test_pipeline))
