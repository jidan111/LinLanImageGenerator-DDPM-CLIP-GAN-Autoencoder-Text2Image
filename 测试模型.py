from LinLanImageGeneratorFrame import *
from LinLanImageGeneratorFrame.globals import *
from LinLanImageGeneratorFrame.utils import *

with open("./upload/model/DDPM/cartoon_ddpm_config.json", 'r') as f:
    config = json.load(f)
unet = set_model_from_config(config["model"], globals_=globals())
ddpm = set_model_from_config(config, globals_=globals())
ddpm.model = unet
ddpm.load_state_dict(torch.load("./upload/model/DDPM/cartoon_ddpm_state_dict.pth"))
out = ddpm.ddim_sample(batch_size=4, device="cpu", step=2).clamp(0, 1)
save_image(out, "./out.png", nrow=2, padding=1)
