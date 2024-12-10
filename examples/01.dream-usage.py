import sys
sys.path.append('/root/autodl-tmp/IF-easy-webui')
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
from deepfloyd_if.modules import IFStageI, IFStageII, StableStageIII
from deepfloyd_if.modules.t5 import T5Embedder

device = 'cuda:0'
# setting model cache lcation
config_path = '/root/autodl-tmp/transformers-cache/'

if_I = IFStageI('IF-I-XL-v1.0', device=device, cache_dir=config_path)
if_II = IFStageII('IF-II-L-v1.0', device=device, cache_dir=config_path)
if_III = StableStageIII('stable-diffusion-x4-upscaler', device=device, cache_dir=config_path)
t5 = T5Embedder(device="cpu", cache_dir=config_path)

from deepfloyd_if.pipelines import dream

prompt = 'ultra close-up color photo portrait of rainbow owl with deer horns in the woods'
count = 4

result = dream(
    t5=t5, if_I=if_I, if_II=if_II, #if_III=if_III,
    prompt=[prompt]*count,
    seed=42,
    if_I_kwargs={
        "guidance_scale": 7.0,
        "sample_timestep_respacing": "smart100",
    },
    if_II_kwargs={
        "guidance_scale": 4.0,
        "sample_timestep_respacing": "smart50",
    },
#    if_III_kwargs={
#        "guidance_scale": 9.0,
#        "noise_level": 20,
#        "sample_timestep_respacing": "75",
#    },
)

# if_III.show(result['III'], size=14)
# save the generated images as png files
for stage, images in result.items():
    for i, image in enumerate(images):
        # 保存到generate_imgs文件夹中
        image.save(f'/root/autodl-tmp/IF-easy-webui/generate-imgs/01.dream-usage_{stage}_{i}.png')