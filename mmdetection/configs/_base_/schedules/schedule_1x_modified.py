##############Warm-up 방식의 스케줄러임 ###############
# training schedule for 1x
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=10, val_interval=2)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# learning rate with warm-up
param_scheduler = [
    dict(
        type='LinearLR', 
        start_factor=0.001,  # 초기 학습률 비율
        by_epoch=False, 
        begin=0, 
        end=200  # warm-up 단계 종료 (iteration 기준)
    ),
    dict(
        type='MultiStepLR',
        begin=0,
        end=3,
        by_epoch=True,
        milestones=[8, 11],
        gamma=0.1)
]

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=1e-3, weight_decay=0.0001),
    clip_grad=dict(max_norm=1.0, norm_type=2),
     accumulation_steps=4)

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (2 samples per GPU).
auto_scale_lr = dict(enable=True, base_batch_size=4)