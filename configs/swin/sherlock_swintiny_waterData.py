_base_ = [
    '../_base_/models/sherlock_upernet_swin.py', 
    #'../_base_/datasets/sherlock_water_256x256.py',
    '../_base_/datasets/sherlock_water_512x512.py',
    '../_base_/default_runtime.py', 
    '../_base_/schedules/sherlock_schedule_300epo.py'
]
crop_size = (512,512)
data_preprocessor = dict(size=crop_size)
checkpoint_file = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_tiny_patch4_window7_224_20220317-1cdeb081.pth'  # noqa
model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file),
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        use_abs_pos_embed=False,
        drop_path_rate=0.3,
        patch_norm=True),
    decode_head=dict(in_channels=[96, 192, 384, 768], num_classes=2),
    #auxiliary_head=dict(in_channels=384, num_classes=2)
    )

# AdamW optimizer, no weight decay for position embedding & layer norm
# in backbone
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))

param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=True, begin=0, end=20),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=20,
        end=300,
        by_epoch=True,
    )
]

# By default, models are trained on 8 GPUs with 2 images per GPU
train_dataloader = dict()
val_dataloader = dict()
