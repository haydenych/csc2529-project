# Self-Supervised Image Denoising with Noise Correlation Priors

This repository includes code for our CSC2529 Project.

# Dataset
Download the SIDD SRGB Medium Training Dataset [here](https://www.eecs.yorku.ca/~kamel/sidd/dataset.php) and the Benchmark Dataset [here](https://www.eecs.yorku.ca/~kamel/sidd/benchmark.php).

To prepare the datasets for training AP-BSN or LG-BPN, follow the instructions [here](https://github.com/wooseoklee4/AP-BSN/blob/master/src/datahandler/prepare_dataset.md).


After downloading and preprocessing the data, place them in the following structure

```
├── prep
│   └── SIDD_s512_o128
│       ├── CL
│       └── RN
└── SIDD
    ├── SIDD_Benchmark
    │   └── BenchmarkNoisyBlocksSrgb.mat
    ├── SIDD_Medium_Srgb
    │   ├── Data
    │   ├── ReadMe_sRGB.txt
    │   └── Scene_Instances.txt
    └── SIDD_Validation
        ├── ValidationGtBlocksSrgb.mat
        └── ValidationNoisyBlocksSrgb.mat

```

# Setup
To setup the conda environment, run `conda env create -f env.yml` and then activate the environment using `conda activate CSC2529_Project`.

Since this code is run on a linux environment, there might be problems when setting up on MacBook, in particular with the libc files.


# Scripts
```
python3 main.py
```

By default, it is set to train the unet, if you wish to train other models, you may change it to the following, for example

To train BNN (Stage 1)
```
model_bnn = SSID_BNN(args.bnn_cfg_path)
model_bnn.train()
```

To train LAN (Stage 2)
```
model_bnn = SSID_BNN(args.bnn_cfg_path)
model_lan = SSID_LAN(args.lan_cfg_path, model_bnn)
model_lan.train()
```

To train UNet (Stage 3)
```
model_bnn = SSID_BNN(args.bnn_cfg_path)
model_lan = SSID_LAN(args.lan_cfg_path, model_bnn)
model_unet = SSID_UNet(args.unet_cfg_path, model_bnn, model_lan)
model_unet.train()
```

To validate, you may run
```
model_bnn = SSID_BNN(args.bnn_cfg_path)
model_lan = SSID_LAN(args.lan_cfg_path, model_bnn)
model_unet = SSID_UNet(args.unet_cfg_path, model_bnn, model_lan)
model_unet.validate()
```

Or inference,
```
model_bnn = SSID_BNN(args.bnn_cfg_path)
model_lan = SSID_LAN(args.lan_cfg_path, model_bnn)
model_unet = SSID_UNet(args.unet_cfg_path, model_bnn, model_lan)

x = Image.open("your path to image")
model_unet.inference(x)
```

# Configurations
Sample configs have been made available in the folder cfgs. We however do not include pretrained weights due to its large size.