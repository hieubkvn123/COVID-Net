python3 serve_model.py\
	--weightspath models/COVIDNet-CXR-2 \
	--metaname model.meta \
	--ckptname model \
	--n_classes 2 \
	--imagepath assets/ex-covid.jpeg \
	--in_tensorname input_1:0 \
	--out_tensorname norm_dense_2/Softmax:0
