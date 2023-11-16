train_pipeline_enhanced = [
    dict(
        type='LoadImageFromFile',
        file_client_args=dict(backend='disk'),
        color_type='color_ignore_orientation'),
    dict(
        type='LoadOCRAnnotations',
        with_polygon=True,
        with_bbox=True,
        with_label=True),
    dict(type='FixInvalidPolygon', min_poly_points=4),
    dict(
        type='RandomResize',
        scale=(800, 800),
        ratio_range=(0.75, 2.5),
        keep_ratio=True),
    dict(
        type='TextDetRandomCropFlip',
        crop_ratio=0.5,
        iter_num=1,
        min_area_ratio=0.2),
    dict(
        type='RandomApply',
        transforms=[dict(type='RandomCrop', min_side_ratio=0.3)],
        prob=0.8),
    dict(
        type='RandomApply',
        transforms=[
            dict(
                type='RandomRotate',
                max_angle=35,
                pad_with_fixed_color=True,
                use_canvas=True)
        ],
        prob=0.6),
    dict(
        type='RandomChoice',
        transforms=[[{
            'type': 'Resize',
            'scale': 800,
            'keep_ratio': True
        }, {
            'type': 'Pad',
            'size': (800, 800)
        }], {
            'type': 'Resize',
            'scale': 800,
            'keep_ratio': False
        }],
        prob=[0.6, 0.4]),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='RandomFlip', prob=0.5, direction='vertical'),
    dict(
        type='RandomApply',
        transforms=[
            dict(type='TorchVisionWrapper', op='ElasticTransform', alpha=75.0)
        ],
        prob=0.3333333333333333),
    dict(
        type='RandomApply',
        transforms=[
            dict(
                type='RandomChoice',
                transforms=[
                    dict(
                        type='TorchVisionWrapper',
                        op='RandomAdjustSharpness',
                        sharpness_factor=0),
                    dict(
                        type='TorchVisionWrapper',
                        op='RandomAdjustSharpness',
                        sharpness_factor=60),
                    dict(
                        type='TorchVisionWrapper',
                        op='RandomAdjustSharpness',
                        sharpness_factor=90)
                ],
                prob=[
                    0.3333333333333333, 0.3333333333333333, 0.3333333333333333
                ])
        ],
        prob=0.75),
    dict(
        type='TorchVisionWrapper',
        op='ColorJitter',
        brightness=0.15,
        saturation=0.5,
        contrast=0.3),
    dict(
        type='RandomApply',
        transforms=[
            dict(
                type='RandomChoice',
                transforms=[
                    dict(type='TorchVisionWrapper', op='RandomEqualize'),
                    dict(type='TorchVisionWrapper', op='RandomAutocontrast')
                ],
                prob=[0.5, 0.5])
        ],
        prob=0.8),
    dict(type='FixInvalidPolygon', min_poly_points=4),
    dict(
        type='PackTextDetInputs',
        meta_keys=('img_path', 'ori_shape', 'img_shape', 'scale_factor'))
]
file_client_args = dict(backend='disk')
model = dict(
    type='FCENet',
    backbone=dict(
        type='mmdet.ResNet',
        depth=50,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'),
        norm_eval=True,
        style='pytorch',
),
    neck=dict(
        type='mmdet.FPN',
        in_channels=[512, 1024, 2048],
        out_channels=256,
        add_extra_convs='on_output',
        num_outs=3,
        relu_before_extra_convs=True,
        act_cfg=None),
    det_head=dict(
        type='FCEHead',
        in_channels=256,
        fourier_degree=5,
        module_loss=dict(
            type='FCEModuleLoss',
            num_sample=50,
            level_proportion_range=((0, 0.25), (0.2, 0.65), (0.55, 1.0))),
        postprocessor=dict(
            type='FCEPostprocessor',
            scales=(8, 16, 32),
            text_repr_type='poly',
            num_reconstr_points=50,
            alpha=1.0,
            beta=2.0,
            score_thr=0.3)),
    data_preprocessor=dict(
        type='TextDetDataPreprocessor',
        mean=[86.65888836888392, 67.92744567921709, 53.78325960605914],
        std=[68.98970994105028, 57.20489382979894, 48.230552014910586],
        bgr_to_rgb=True,
        pad_size_divisor=32))
default_scope = 'mmocr'
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))
randomness = dict(seed=None)
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='CustomLoggerHook', interval=1),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=100),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    sync_buffer=dict(type='SyncBuffersHook'),
    visualization=dict(
        type='VisualizationHook',
        interval=1,
        enable=True,
        show=False,
        draw_gt=True,
        draw_pred=True))
log_level = 'INFO'
log_processor = dict(type='LogProcessor', window_size=10, by_epoch=True)
load_from = '../../checkpoints/fcenet_resnet50-dcnv2.pth'
resume = False
val_evaluator = dict(type='HmeanIOUMetric')
test_evaluator = dict(type='HmeanIOUMetric')
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='TextDetLocalVisualizer',
    name='visualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend')
    ])
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.0003, momentum=0.9, weight_decay=0.0005))
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=1500, val_interval=50)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
param_scheduler = [dict(type='PolyLR', power=0.9, eta_min=1e-07, end=1500)]
train_pipeline = [
    dict(
        type='LoadImageFromFile',
        file_client_args=dict(backend='disk'),
        color_type='color_ignore_orientation'),
    dict(
        type='LoadOCRAnnotations',
        with_polygon=True,
        with_bbox=True,
        with_label=True),
    dict(type='FixInvalidPolygon', min_poly_points=4),
    dict(
        type='RandomResize',
        scale=(800, 800),
        ratio_range=(0.75, 2.5),
        keep_ratio=True),
    dict(
        type='TextDetRandomCropFlip',
        crop_ratio=0.5,
        iter_num=1,
        min_area_ratio=0.2),
    dict(
        type='RandomApply',
        transforms=[dict(type='RandomCrop', min_side_ratio=0.3)],
        prob=0.8),
    dict(
        type='RandomApply',
        transforms=[
            dict(
                type='RandomRotate',
                max_angle=35,
                pad_with_fixed_color=True,
                use_canvas=True)
        ],
        prob=0.6),
    dict(
        type='RandomChoice',
        transforms=[[{
            'type': 'Resize',
            'scale': 800,
            'keep_ratio': True
        }, {
            'type': 'Pad',
            'size': (800, 800)
        }], {
            'type': 'Resize',
            'scale': 800,
            'keep_ratio': False
        }],
        prob=[0.6, 0.4]),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='RandomFlip', prob=0.5, direction='vertical'),
    dict(
        type='RandomApply',
        transforms=[
            dict(type='TorchVisionWrapper', op='ElasticTransform', alpha=75.0)
        ],
        prob=0.3333333333333333),
    dict(
        type='RandomApply',
        transforms=[
            dict(
                type='RandomChoice',
                transforms=[
                    dict(
                        type='TorchVisionWrapper',
                        op='RandomAdjustSharpness',
                        sharpness_factor=0),
                    dict(
                        type='TorchVisionWrapper',
                        op='RandomAdjustSharpness',
                        sharpness_factor=60),
                    dict(
                        type='TorchVisionWrapper',
                        op='RandomAdjustSharpness',
                        sharpness_factor=90)
                ],
                prob=[
                    0.3333333333333333, 0.3333333333333333, 0.3333333333333333
                ])
        ],
        prob=0.75),
    dict(
        type='TorchVisionWrapper',
        op='ColorJitter',
        brightness=0.15,
        saturation=0.5,
        contrast=0.3),
    dict(
        type='RandomApply',
        transforms=[
            dict(
                type='RandomChoice',
                transforms=[
                    dict(type='TorchVisionWrapper', op='RandomEqualize'),
                    dict(type='TorchVisionWrapper', op='RandomAutocontrast')
                ],
                prob=[0.5, 0.5])
        ],
        prob=0.8),
    dict(type='FixInvalidPolygon', min_poly_points=4),
    dict(
        type='PackTextDetInputs',
        meta_keys=('img_path', 'ori_shape', 'img_shape', 'scale_factor'))
]
train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='OCRDataset',
        data_root='data/icdar2015',
        ann_file='textdet_train.json',
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=[
            dict(
                type='LoadImageFromFile',
                file_client_args=dict(backend='disk'),
                color_type='color_ignore_orientation'),
            dict(
                type='LoadOCRAnnotations',
                with_polygon=True,
                with_bbox=True,
                with_label=True),
            dict(type='FixInvalidPolygon', min_poly_points=4),
            dict(
                type='RandomResize',
                scale=(800, 800),
                ratio_range=(0.75, 2.5),
                keep_ratio=True),
            dict(
                type='TextDetRandomCropFlip',
                crop_ratio=0.5,
                iter_num=1,
                min_area_ratio=0.2),
            dict(
                type='RandomApply',
                transforms=[dict(type='RandomCrop', min_side_ratio=0.3)],
                prob=0.8),
            dict(
                type='RandomApply',
                transforms=[
                    dict(
                        type='RandomRotate',
                        max_angle=35,
                        pad_with_fixed_color=True,
                        use_canvas=True)
                ],
                prob=0.6),
            dict(
                type='RandomChoice',
                transforms=[[{
                    'type': 'Resize',
                    'scale': 800,
                    'keep_ratio': True
                }, {
                    'type': 'Pad',
                    'size': (800, 800)
                }], {
                    'type': 'Resize',
                    'scale': 800,
                    'keep_ratio': False
                }],
                prob=[0.6, 0.4]),
            dict(type='RandomFlip', prob=0.5, direction='horizontal'),
            dict(type='RandomFlip', prob=0.5, direction='vertical'),
            dict(
                type='RandomApply',
                transforms=[
                    dict(
                        type='TorchVisionWrapper',
                        op='ElasticTransform',
                        alpha=75.0)
                ],
                prob=0.3333333333333333),
            dict(
                type='RandomApply',
                transforms=[
                    dict(
                        type='RandomChoice',
                        transforms=[
                            dict(
                                type='TorchVisionWrapper',
                                op='RandomAdjustSharpness',
                                sharpness_factor=0),
                            dict(
                                type='TorchVisionWrapper',
                                op='RandomAdjustSharpness',
                                sharpness_factor=60),
                            dict(
                                type='TorchVisionWrapper',
                                op='RandomAdjustSharpness',
                                sharpness_factor=90)
                        ],
                        prob=[
                            0.3333333333333333, 0.3333333333333333,
                            0.3333333333333333
                        ])
                ],
                prob=0.75),
            dict(
                type='TorchVisionWrapper',
                op='ColorJitter',
                brightness=0.15,
                saturation=0.5,
                contrast=0.3),
            dict(
                type='RandomApply',
                transforms=[
                    dict(
                        type='RandomChoice',
                        transforms=[
                            dict(
                                type='TorchVisionWrapper',
                                op='RandomEqualize'),
                            dict(
                                type='TorchVisionWrapper',
                                op='RandomAutocontrast')
                        ],
                        prob=[0.5, 0.5])
                ],
                prob=0.8),
            dict(type='FixInvalidPolygon', min_poly_points=4),
            dict(
                type='PackTextDetInputs',
                meta_keys=('img_path', 'ori_shape', 'img_shape',
                           'scale_factor'))
        ]))
val_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='OCRDataset',
        data_root='data/icdar2015',
        ann_file='textdet_test.json',
        test_mode=True,
        pipeline=[
            dict(
                type='LoadImageFromFile',
                file_client_args=dict(backend='disk'),
                color_type='color_ignore_orientation'),
            dict(type='Resize', scale=(2260, 2260), keep_ratio=True),
            dict(
                type='LoadOCRAnnotations',
                with_polygon=True,
                with_bbox=True,
                with_label=True),
            dict(
                type='PackTextDetInputs',
                meta_keys=('img_path', 'ori_shape', 'img_shape',
                           'scale_factor'))
        ]))
test_dataloader = dict(
    batch_size=4,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='OCRDataset',
        data_root='data/icdar2015',
        ann_file='textdet_test.json',
        test_mode=True,
        pipeline=[
            dict(
                type='LoadImageFromFile',
                file_client_args=dict(backend='disk'),
                color_type='color_ignore_orientation'),
            dict(type='Resize', scale=(2260, 2260), keep_ratio=True),
            dict(
                type='LoadOCRAnnotations',
                with_polygon=True,
                with_bbox=True,
                with_label=True),
            dict(
                type='PackTextDetInputs',
                meta_keys=('img_path', 'ori_shape', 'img_shape',
                           'scale_factor'))
        ]))
auto_scale_lr = dict(base_batch_size=16)
optimizer_config = dict(
    type='GradientCumulativeOptimizerHook', cumulative_iters=4)
custom_hooks = [dict(type='CustomTensorboardLoggerHook', by_epoch=True)]
launcher = 'none'
work_dir = 'logs/yunus'