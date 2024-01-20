_base_ = [
    '../_base_/models/upernet_convnext.py', 
    '../_base_/datasets/sherlock_beijingbuilding_256.py',
    '../_base_/sherlock_default_runtime.py', 
    '../_base_/schedules/sherlock_schedule_100k.py'
]
crop_size = (256,256)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    decode_head=dict(in_channels=[128, 256, 512, 1024], num_classes=2),
    #auxiliary_head=dict(in_channels=512, num_classes=150),
    test_cfg=dict(mode='whole'),
)

optim_wrapper = dict(
    _delete_=True,
    type='AmpOptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.05),
    paramwise_cfg={
        'decay_rate': 0.9,
        'decay_type': 'stage_wise',
        'num_layers': 12
    },
    constructor='LearningRateDecayOptimizerConstructor',
    loss_scale='dynamic')

param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type='PolyLR',
        power=1.0,
        begin=1500,
        end=100000,
        eta_min=0.0,
        by_epoch=False,
    )
]

# By default, models are trained on 8 GPUs with 2 images per GPU
train_dataloader = dict(batch_size=8)
val_dataloader = dict(batch_size=1)
test_dataloader = val_dataloader
