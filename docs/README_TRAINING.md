# Training Guide

## COCO Dataset Setup

### 1. Download COCO Dataset

COCO 2017 training dataset을 다운로드해야 합니다.

#### Option 1: Download Script 사용
```bash
bash scripts/download_coco.sh
```

#### Option 2: 수동 다운로드

1. **COCO 2017 Training Images** 다운로드:
   ```bash
   wget http://images.cocodataset.org/zips/train2017.zip
   unzip train2017.zip -d /data4/hongsik/data/COCO/
   ```

2. **COCO 2017 Captions Annotations** 다운로드:
   ```bash
   wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
   unzip annotations_trainval2017.zip
   cp annotations/captions_train2017.json /data4/hongsik/data/COCO/annotations/
   ```

#### 최종 디렉토리 구조
```
/data4/hongsik/data/COCO/
├── annotations/
│   └── captions_train2017.json
└── train2017/
    └── *.jpg (118,287 images)
```

### 2. Dataset 정보 확인
```bash
python -c "from data.coco_dataset import download_coco_info; download_coco_info()"
```

## Training 실행

### 기본 Training
```bash
python train.py \
    --data_root /data4/hongsik/data/COCO \
    --mode vision_tower \
    --batch_size 4 \
    --num_epochs 200 \
    --learning_rate 1e-3
```

### Mode 2로 Training
```bash
python train.py \
    --data_root /data4/hongsik/data/COCO \
    --mode image_proj_norm \
    --batch_size 4 \
    --num_epochs 200 \
    --learning_rate 1e-3
```

### Channel Noise 포함 Training
```bash
python train.py \
    --data_root /data4/hongsik/data/COCO \
    --mode vision_tower \
    --channel_type awgn \
    --snr_db 20.0 \
    --use_channel \
    --batch_size 4 \
    --num_epochs 200
```

### Gradient Accumulation 사용
```bash
python train.py \
    --data_root /data4/hongsik/data/COCO \
    --batch_size 2 \
    --accumulation_steps 4 \
    --num_epochs 200
```

## Training Arguments

### Model Arguments
- `--model_name`: Florence-2 model name (default: `microsoft/Florence-2-base`)
- `--mode`: Processing mode (`vision_tower` or `image_proj_norm`)

### Data Arguments
- `--data_root`: COCO dataset root directory (default: `/data4/hongsik/data/COCO`)
- `--task_prompt`: Task prompt (default: `<CAPTION>`)
- `--batch_size`: Batch size (default: 4)
- `--num_workers`: Data loading workers (default: 4)

### Training Arguments
- `--num_epochs`: Number of epochs (default: 200)
- `--learning_rate`: Learning rate (default: 1e-3)
- `--weight_decay`: Weight decay (default: 1e-4)
- `--accumulation_steps`: Gradient accumulation steps (default: 1)

### Channel Arguments
- `--channel_type`: Channel type (`noiseless`, `awgn`, `rayleigh`)
- `--snr_db`: SNR in dB (default: 20.0)
- `--use_channel`: Enable channel noise during training

### Output Arguments
- `--output_dir`: Checkpoint save directory (default: `./checkpoints`)
- `--save_interval`: Save checkpoint every N epochs (default: 1)

## Training Process

1. **Transmitter**: 이미지를 vision features로 변환
2. **Channel**: Noise 추가 (옵션)
3. **Receiver**: Received signal을 처리하고 text embeddings와 merge
4. **Language Model**: Caption generation을 위한 forward pass
5. **Loss**: Cross-entropy loss 계산 (caption tokens에 대해서만)

## Checkpoints

Checkpoints는 `--output_dir`에 저장되며, 다음 정보를 포함합니다:
- `epoch`: 현재 epoch
- `transmitter_state_dict`: Transmitter 가중치
- `receiver_state_dict`: Receiver 가중치
- `optimizer_state_dict`: Optimizer 상태
- `scheduler_state_dict`: Scheduler 상태
- `loss`: 현재 loss
- `mode`: 사용된 mode

## 주의사항

1. **메모리**: COCO dataset은 크므로 적절한 batch_size를 설정하세요
2. **Gradient Accumulation**: 메모리가 부족하면 `accumulation_steps`를 사용하세요
3. **Channel Noise**: Training 시 channel noise를 사용하려면 `--use_channel` 플래그를 추가하세요
4. **Learning Rate**: Florence-2는 pre-trained model이므로 작은 learning rate를 권장합니다 (1e-5 ~ 1e-6)
