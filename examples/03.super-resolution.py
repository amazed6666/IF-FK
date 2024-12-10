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

from deepfloyd_if.pipelines import super_resolution

# 加载原始图像
raw_pil_image = Image.open('/root/autodl-tmp/IF-easy-webui/raw-image/raw-image-02.png')


middle_res = super_resolution(
    t5,
    if_III=if_II,
    prompt=['woman with a blue headscarf and a blue sweaterp, detailed picture, 4k dslr, best quality'],
    support_pil_img=raw_pil_image,
    img_scale=4.,
    img_size=64,
    if_III_kwargs={
        'sample_timestep_respacing': 'smart100',
        'aug_level': 0.5,
        'guidance_scale': 6.0,
    },
)
high_res = super_resolution(
    t5,
    if_III=if_III,
    prompt=[''],
    support_pil_img=middle_res['III'][0],
    img_scale=4.,
    img_size=256,
    if_III_kwargs={
        "guidance_scale": 9.0,
        "noise_level": 20,
        "sample_timestep_respacing": "75",
    },
)
#show_superres(raw_pil_image, high_res['III'][0])

#print(result)
# 保存生成的图片

for stage, images in middle_res.items():
    for i, img in enumerate(images):
        img.save(f'img/06.super_middle_resolution_{stage}_{i}-3.png')

for stage, images in high_res.items():
    for i, img in enumerate(images):
        img.save(f'/root/autodl-tmp/IF-easy-webui/generate-imgs/03.super_high_resolution_{stage}_{i}-3.png')


    