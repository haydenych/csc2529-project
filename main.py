import argparse

from model.LGBPN_BNN import LGBPN_BNN
from model.SSID_BNN import SSID_BNN
from model.SSID_LAN import SSID_LAN

def main(args):
    model_bnn = SSID_BNN(args.bnn_cfg_path)
    model = SSID_LAN(args.lan_cfg_path, model_bnn)
    model.train()

    # model_bnn = LGBPN_BNN(args.bnn_cfg_path)
    # model = SSID_LAN(args.lan_cfg_path, model_bnn)
    # model.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--bnn-cfg-path', type=str, default='./cfgs/train_lan_1/SSID_BNN.json', help="Path to model config")
    parser.add_argument('--lan-cfg-path', type=str, default='./cfgs/train_lan_1/SSID_LAN.json', help="Path to model config")

    args = parser.parse_args()

    main(args)