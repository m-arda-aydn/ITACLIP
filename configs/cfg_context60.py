_base_ = './base_config.py'

dataset_name = 'context60'
# model settings
model = dict(
    type='ITACLIP_Segmentor',
    model_name = 'ViT-B/16',
    img_engineering = True,
    dataset_name = dataset_name,
    auxiliary_text_path = f'./ITACLIP/llama_generated_texts/{dataset_name}_definitions.txt',
    slide_stride = 28, 
    attn_self = True,
    def_coefficient = 0.15,
    img_eng_coefficient = 0.75,
    pamr_steps = 10,
    device = 'cuda:0',
    name_path=f'./ITACLIP/configs/cls_{dataset_name}.txt',
    logit_scale = 55,
    prob_thd = 0.1
)

# dataset settings
dataset_type = 'PascalContext60Dataset'
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
            img_path='JPEGImages', seg_map_path='SegmentationClassContext'),
        ann_file='ImageSets/SegmentationContext/val.txt',
        pipeline=test_pipeline))
