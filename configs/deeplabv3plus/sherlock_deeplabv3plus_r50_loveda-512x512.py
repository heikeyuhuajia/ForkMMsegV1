_base_ = [
    '../_base_/models/deeplabv3plus_r50-d8.py', 
    '../_base_/datasets/sherlock_beijingbuilding_256.py',
    '../_base_/default_runtime.py', 
    '../_base_/schedules/sherlock_schedule_100k.py'
]
crop_size = (256,256)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    decode_head=dict(num_classes=2),
    #auxiliary_head=dict(num_classes=2)
    )
