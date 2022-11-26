

CUDA_VISIBLE_DEVICES=0 python scripts/train_mpii.py \
    --arch=hg2 \
    --image-path=data/images/ \
    --checkpoint=checkpoint/hg2-base-kl-latent-only \
    --epochs=30 \
    --train-batch=24 \
    --workers=24 \
    --test-batch=24 \
    --lr=1e-3 \
    --schedule 15 17