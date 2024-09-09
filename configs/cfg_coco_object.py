_base_ = './base_config.py'

dataset_name = 'coco_object'
# model settings
model = dict(
    type='ITACLIP_Segmentor',
    model_name = 'ViT-B/16',
    img_engineering = True,
    dataset_name = dataset_name,
    auxiliary_text_path = f'/ITACLIP/llama_generated_texts/{dataset_name}_synonyms.txt',
    slide_stride = 28,
    attn_self = True,
    def_coefficient = 0.1,
    img_eng_coefficient = 0.75,
    pamr_steps = 10,
    width_chunk_size = 150, # This variable helps reduce GPU memory consumption when the text expansion technique is applied. The default values are optimized for a 24 GB GPU.
    device = 'cuda:0',
    name_path=f'/ITACLIP/configs/cls_{dataset_name}.txt',
    logit_scale=50,
    prob_thd=0.1
)

# dataset settings
dataset_type = 'COCOObjectDataset'
data_root = ' '

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(2048, 336), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
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
        reduce_zero_label=False,
        data_prefix=dict(
            img_path='images/val2017', seg_map_path='annotations/val2017'),
        pipeline=test_pipeline))
