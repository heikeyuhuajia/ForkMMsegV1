_base_ = './sherlock_deeplabv3plus_r50_loveda-512x512.py'
model = dict(
    pretrained='open-mmlab://resnet18_v1c',
    backbone=dict(depth=18),
    decode_head=dict(
        c1_in_channels=64,
        c1_channels=12,
        in_channels=512,
        channels=128,
    ),
    #auxiliary_head=dict(in_channels=256, channels=64)
    )
