_base_ = [
    '../_base_/models/twins_pcpvt-s_fpn.py', 
    '../_base_/datasets/sherlock_beijingbuilding_256.py',
    '../_base_/default_runtime.py', 
    '../_base_/schedules/sherlock_schedule_100k.py'
]
crop_size = (256,256)
data_preprocessor = dict(size=crop_size)
checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/twins/alt_gvt_small_20220308-7e1c3695.pth'  # noqa

model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='SVT',
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint),
        embed_dims=[64, 128, 256, 512],
        num_heads=[2, 4, 8, 16],
        mlp_ratios=[4, 4, 4, 4],
        depths=[2, 2, 10, 4],
        windiow_sizes=[7, 7, 7, 7],
        norm_after_stage=True),
    neck=dict(in_channels=[64, 128, 256, 512], out_channels=256, num_outs=4),
    decode_head=dict(num_classes=2),
)

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0001, weight_decay=0.0001),
    clip_grad=None)
