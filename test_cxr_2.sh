python3 inference.py \                                                                                                 minhhieu@minhhieu-Z490-GAMING-X 10:52:24
    --weightspath models/COVIDNet-CXR-2 \
    --metaname model.meta \
    --ckptname model \
    --n_classes 2 \
    --imagepath assets/ex-covid.jpeg \
    --in_tensorname input_1:0 \
    --out_tensorname norm_dense_2/Softmax:0

