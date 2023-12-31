from super_gradients.training import Trainer, models
from super_gradients.training.losses import PPYoloELoss
from super_gradients.training.metrics import DetectionMetrics_050

from super_gradients.training.dataloaders.dataloaders import (
    coco_detection_yolo_format_train,
    coco_detection_yolo_format_val
)

from super_gradients.training.models.detection_models.pp_yolo_e import (
    PPYoloEPostPredictionCallback
)


class config:
    # trainer params
    CHECKPOINT_DIR = 'checkpoints'  # specify the path you want to save checkpoints to
    EXPERIMENT_NAME = 'finding-battleships'  # specify the experiment name
    # dataset params
    DATA_DIR = 'my_yolo_dataset/'  # parent directory to where data lives
    TRAIN_IMAGES_DIR = 'train/images'  # child dir of DATA_DIR where train images are
    TRAIN_LABELS_DIR = 'train/labels'  # child dir of DATA_DIR where train labels are
    VAL_IMAGES_DIR = 'val/images'  # child dir of DATA_DIR where validation images are
    VAL_LABELS_DIR = 'val/labels'  # child dir of DATA_DIR where validation labels are

    CLASSES = ['Prescription-Pharmacist', 'Reviewer', 'Dispenser']

    NUM_CLASSES = len(CLASSES)

    # dataloader params - you can add whatever PyTorch dataloader params you have
    # could be different across train, val, and test
    TRAIN_DATALOADER_PARAMS = {
        'batch_size': 16,
        'num_workers': 2,
        'shuffle': True
    }

    VALID_DATALOADER_PARAMS = {
        'batch_size': 1,
        'num_workers': 2,
        'shuffle': False
    }

    # model params
    MODEL_NAME = 'yolo_nas_l'  # choose from yolo_nas_s, yolo_nas_m, yolo_nas_l
    PRETRAINED_WEIGHTS = 'coco'  # only one option here: coco

trainer = Trainer(experiment_name=config.EXPERIMENT_NAME,
                  ckpt_root_dir=config.CHECKPOINT_DIR)


train_data = coco_detection_yolo_format_train(
    dataset_params={
        'data_dir': config.DATA_DIR,
        'images_dir': config.TRAIN_IMAGES_DIR,
        'labels_dir': config.TRAIN_LABELS_DIR,
        'classes': config.CLASSES
    },
    dataloader_params=config.TRAIN_DATALOADER_PARAMS
)

val_data = coco_detection_yolo_format_val(
    dataset_params={
        'data_dir': config.DATA_DIR,
        'images_dir': config.VAL_IMAGES_DIR,
        'labels_dir': config.VAL_LABELS_DIR,
        'classes': config.CLASSES
    },
    dataloader_params=config.VALID_DATALOADER_PARAMS
)

model = models.get(config.MODEL_NAME,
                   num_classes=config.NUM_CLASSES,
                   pretrained_weights=config.PRETRAINED_WEIGHTS
                   )

train_params = {
    # ENABLING SILENT MODE
    "average_best_models": True,
    "warmup_mode": "linear_epoch_step",
    "warmup_initial_lr": 1e-6,
    "lr_warmup_epochs": 3,
    "initial_lr": 5e-4,
    "lr_mode": "cosine",
    "cosine_final_lr_ratio": 0.1,
    "optimizer": "AdamW",
    "optimizer_params": {"weight_decay": 0.0001},
    "zero_weight_decay_on_bias_and_bn": True,
    "ema": True,
    "ema_params": {"decay": 0.9, "decay_type": "threshold"},
    # ONLY TRAINING FOR 3 EPOCHS FOR THIS EXAMPLE NOTEBOOK
    "max_epochs": 300,
    "mixed_precision": True,  # mixed precision is not available for CPU
    "loss": PPYoloELoss(
        use_static_assigner=False,
        # NOTE: num_classes needs to be defined here
        num_classes=config.NUM_CLASSES,
        reg_max=16
    ),

    "valid_metrics_list": [
        DetectionMetrics_050(
            score_thres=0.1,
            top_k_predictions=300,
            # NOTE: num_classes needs to be defined here
            num_cls=config.NUM_CLASSES,
            normalize_targets=True,
            post_prediction_callback=PPYoloEPostPredictionCallback(
                score_threshold=0.01,
                nms_top_k=1000,
                max_predictions=300,
                nms_threshold=0.7
            )
        )
    ],
    "metric_to_watch": 'mAP@0.50'
}


if __name__ == '__main__':
    trainer.train(model=model,
                  training_params=train_params,
                  train_loader=train_data,
                  valid_loader=val_data)