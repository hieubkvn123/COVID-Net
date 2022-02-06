python3 inference.py \
    --weightspath models/COVIDNet-CXR-3 \
    --metaname model.meta \
    --ckptname model \
    --n_classes 2 \
    --imagepath assets/ex-covid.jpeg \
    --out_tensorname softmax/Softmax:0 \
    --is_medusa_backbone
