_base_ = [
    '../_base_/models/twins_pcpvt-s_fpn.py', 
    '../_base_/datasets/sherlock_beijingbuilding_256.py',
    '../_base_/default_runtime.py', 
    '../_base_/schedules/sherlock_schedule_100k.py'
]
crop_size = (256,256)
data_preprocessor = dict(size=crop_size)
model = dict(data_preprocessor=data_preprocessor)
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0001, weight_decay=0.0001),
    clip_grad=None)
