python3 inference.py \
    --weightspath models/COVIDNet-CXR-2 \
    --metaname model.meta \
    --ckptname model \
    --n_classes 2 \
    --imagepath assets/ex-covid.jpeg \
    --out_tensorname norm_dense_2/Softmax:0 \
	--in_tensorname input_1:0 \

