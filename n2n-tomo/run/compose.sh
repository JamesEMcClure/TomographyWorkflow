python ./n2n-tomo/compose.py \
  --data ./data/input \
  --load-ckpt ./ckpts/n2n-tomo.pt \
  --axis 0 \
  --crop-size 256 \
  --n-width 16 \
  --n-height 16 \
  --batch-size 256 \
  --seed 42 \
  --cuda \
  --show-output 2