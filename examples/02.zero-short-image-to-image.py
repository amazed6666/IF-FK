import sys
sys.path.append('/root/autodl-tmp/IF-easy-webui')
from deepfloyd_if.modules import IFStageI, IFStageII, StableStageIII
from deepfloyd_if.modules.t5 import T5Embedder
from PIL import Image

# 手动设置 config.yml 路径
config_path = '/root/autodl-tmp/transformers-cache/'

device = 'cuda:0'

# 加载配置文件
#config = OmegaConf.load(config_path)

# 初始化模型
device = 'cuda:0'
if_I = IFStageI('IF-I-XL-v1.0', device=device, cache_dir=config_path)
if_II = IFStageII('IF-II-L-v1.0', device=device, cache_dir=config_path)
if_III = StableStageIII('stable-diffusion-x4-upscaler', device=device, cache_dir=config_path)
t5 = T5Embedder(device="cpu", cache_dir=config_path)

from deepfloyd_if.pipelines import style_transfer

# 加载原始图像
raw_pil_image = Image.open('/root/autodl-tmp/IF-easy-webui/raw-image/raw-image-01.png') 
result = style_transfer(
    t5=t5, if_I=if_I, if_II=if_II,
    support_pil_img=raw_pil_image,
    style_prompt=[
        'in style of professional origami',
        'in style of oil art, Tate modern',
        'in style of plastic building bricks',
        'in style of classic anime from 1990',
    ],
    seed=42,
    if_I_kwargs={
        "guidance_scale": 10.0,
        "sample_timestep_respacing": "10,10,10,10,10,10,10,10,0,0",
        'support_noise_less_qsample_steps': 5,
    },
    if_II_kwargs={
        "guidance_scale": 4.0,
        "sample_timestep_respacing": 'smart50',
        "support_noise_less_qsample_steps": 5,
    },
)

print(result)
# 保存生成的图片

for stage, images in result.items():
    for i, img in enumerate(images):
        img.save(f'/root/autodl-tmp/IF-easy-webui/generate-imgs/02.zero-short-image2image_tran_{stage}_{i}_2.png')


    