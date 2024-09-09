_base_ = './base_config.py'

dataset_name = 'coco_stuff'
# model settings
model = dict(
    type='ITACLIP_Segmentor',
    model_name = 'ViT-B/16',
    img_engineering = True,
    auxiliary_text_path = f'/llama_generated_texts/{dataset_name}_definitions.txt',
    dataset_name = dataset_name,
    slide_stride = 28,
    attn_self = True,
    def_coefficient = 0.2,
    img_eng_coefficient = 0.75,
    width_chunk_size = 75, # This variable helps reduce GPU memory consumption when the text expansion technique is applied. The default values are optimized for a 24 GB GPU.
    pamr_steps = 10,
    device = 'cuda:0',
    name_path=f'/cls_{dataset_name}.txt',
    logit_scale = 40,
)

# dataset settings
dataset_type = 'COCOStuffDataset'
data_root = ' '

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(2048, 448), keep_ratio=True),
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
            img_path='images/val2017', seg_map_path='annotations/val2017'),
        pipeline=test_pipeline))

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=2000),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook', interval=1))
