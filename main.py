import argparse

from model.LGBPN_BNN import LGBPN_BNN
from model.SSID_BNN import SSID_BNN
from model.SSID_LAN import SSID_LAN
from model.SSID_UNet import SSID_UNet

def main(args):
    model_bnn = SSID_BNN(args.bnn_cfg_path)
    model_lan = SSID_LAN(args.lan_cfg_path, model_bnn)
    model_unet = SSID_UNet(args.unet_cfg_path, model_bnn, model_lan)

    model_unet.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--bnn-cfg-path', type=str, default='./cfgs/train_unet/SSID_BNN.json', help="Path to model config")
    parser.add_argument('--lan-cfg-path', type=str, default='./cfgs/train_unet/SSID_LAN.json', help="Path to model config")
    parser.add_argument('--unet-cfg-path', type=str, default='./cfgs/train_unet/SSID_UNet.json', help="Path to model config")

    args = parser.parse_args()

    main(args)