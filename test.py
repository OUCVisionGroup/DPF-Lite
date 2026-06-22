import torch
from torch.utils.data import DataLoader
from dataset import UIEB_Dataset
import os
import DPF_Lite
from DPEM import DPEM_S
from PIL import Image

device = "cuda:0" if torch.cuda.is_available() else "cpu"

model = DPF_Lite.UNet_lite().to(device).eval()
model.load_state_dict(torch.load('./checkpoint/EncNet.pth'))
DPEM = DPEM_S.MainNet().to(device).eval()
DPEM.load_state_dict(torch.load('./DPEM/checkpoint/DPEM.pth'))

dataset = UIEB_Dataset(device=device, raw_images_path='./UIEB/test/raw', isTrain=False)
dataloader = DataLoader(dataset, batch_size=1)

save_path = './output'

for batch_idx, (data_raw, data_ref, data_depth, BL, name) in enumerate(dataloader):
    x_B, x_betaD, x_betaB, x_d = DPEM(data_raw, BL/255.0)
    xB_map = x_B * (1 - torch.exp(-x_betaB * x_d))
    xT_map = torch.exp(-x_d * x_betaD)
    outputs = model(data_raw, xB_map/255.0, xT_map)
    enc_img = (outputs[0] * 255).to('cpu', dtype=torch.uint8).permute(1, 2, 0)
    save_name = str(name)[2:-3]
    img_save = Image.fromarray(enc_img.numpy())
    img_save.save(os.path.join(save_path, save_name + '.jpg'))


