_base_ = [
    '../_base_/models/sherlock_fcn_unet.py', 
    '../_base_/datasets/sherlock_beijingbuilding_256.py',
    '../_base_/default_runtime.py', 
    '../_base_/schedules/sherlock_schedule_200k.py'
]
crop_size = (256, 256)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    )
