_base_ = [
    '../_base_/models/sherlock_uranet.py', 
    '../_base_/datasets/sherlock_water_360x640.py',
    '../_base_/default_runtime.py', 
    '../_base_/schedules/sherlock_schedule_200k.py'
    #'../_base_/Runtime_trainWater.py',
    #'../_base_/schedules/Schedule_300ep_uranet.py',
]
crop_size = (360,640)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    )

