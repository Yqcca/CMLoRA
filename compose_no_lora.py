import os
import torch
import argparse
import json
from tqdm import tqdm
from os.path import join, exists
from diffusers import AutoencoderKL
from diffusers import DPMSolverMultistepScheduler

from LoRA_Cache.pipelines.stable_diffusion.pipeline_no_lora import StableDiffusionPipeline
from utils import get_prompt
# from utils import calculate_clip_score
from utils import load_lora_info, generate_combinations
import numpy as np

def main(args):
    # set path based on the image style
    args.save_path = args.save_path + "_" + args.image_style

    # set base model based on the image style
    if args.image_style == 'anime':
        model_name = 'gsdf/Counterfeit-V2.5'
    else:
        model_name = 'SG161222/Realistic_Vision_V5.1_noVAE'

    pipeline = StableDiffusionPipeline.from_pretrained(
        model_name,
        # torch_dtype=torch.float16,
        use_safetensors=True
    ).to("cuda")

    # set vae
    if args.image_style == "reality":
        vae = AutoencoderKL.from_pretrained(
            "stabilityai/sd-vae-ft-mse",
            # torch_dtype=torch.float16
        ).to("cuda")
        pipeline.vae = vae

    # set scheduler
    schedule_config = dict(pipeline.scheduler.config)
    schedule_config["algorithm_type"] = "dpmsolver++"
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(schedule_config)

    # prompt initialization
    init_prompt, negative_prompt = get_prompt(args.image_style)
    prompt = init_prompt

    args.lora_path = join(args.lora_path, args.image_style)
    lora_info = load_lora_info(args.image_style, args.lora_info_path)
    # for element in list(lora_info.keys()):
    #     for lora in lora_info[element]:
    #         pipeline.load_lora_weights(
    #             args.lora_path,
    #             weight_name=lora['id'] + '.safetensors',
    #             adapter_name=lora['id']
    #         )
    combinations = generate_combinations(lora_info, args.compos_num)
    init_prompt, negative_prompt = get_prompt(args.image_style)
    for combo in tqdm(combinations):
        # tmp = '_'.join([lora['id'] for lora in combo])
        # scores[tmp] = {}
        # cur_loras = [lora['id'] for lora in combo]
        
        # set prompt
        triggers = [trigger for lora in combo for trigger in lora['trigger']]
        prompt = init_prompt + ', ' + ', '.join(triggers)
        
        for inter in range(len(args.interval)):
            # generate images
            image = pipeline(
                prompt=prompt, 
                negative_prompt=negative_prompt,
                height=args.height,
                width=args.width,
                num_inference_steps=args.denoise_steps,
                guidance_scale=args.cfg_scale,
                generator=args.generator,
                cache_interval=args.interval[inter],
                cache_layer_id=0,
                cache_block_id=1
            ).images[0]
        
            # save image
            save_path = join(args.save_path, f'{args.compos_num}_elements')
            if not exists(save_path):
                os.makedirs(save_path)

            # file name    
            file_name = 'naive' + '_'.join([lora['id'] for lora in combo]) + '_' + str(args.interval[inter]) + '.png'
            image.save(join(save_path, file_name))

            # # clip score
            # image = np.array(image, dtype=np.float32)[np.newaxis, ...] / 255.0
            # score = calculate_clip_score(image, ', '.join(triggers))
            # scores[tmp][str(args.interval[inter])] = [score, rm_score]

            # print('_'.join([lora['id'] for lora in combo]) + '_' + str(args.interval[inter]))
            # print(score)
            # print('--------------------')

    # w_name = args.method+'_'+'results.json'
    # with open(join(save_path, w_name), 'w') as file:
    #     json.dump(scores, file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Given LoRAs in the ComposLoRA benchmark, generate images with arbitrary combinations based on LoRAs'
    )

    # Arguments for composing LoRAs
    # parser.add_argument('--num', default=24, type=int)
    parser.add_argument('--save_path', default='output_pure',
                        help='path to save the generated image', type=str)
    parser.add_argument('--interval', default=[1,2,3],
                        help='number of steps to cache LoRA during denoising', type=json.loads)
    parser.add_argument('--compos_num', default=2,
                        help='number of elements to be combined in a single image', type=int)
    parser.add_argument('--lora_path', default='LoRA_Cache/models/lora',
                        help='path to store all LoRA models', type=str)
    parser.add_argument('--lora_info_path', default='lora_info.json',
                        help='path to stroe all LoRA information', type=str)
    # Arguments for generating images
    parser.add_argument('--height', default=512,
                        help='height of the generated images', type=int)
    parser.add_argument('--width', default=512,
                        help='width of the generated images', type=int)
    parser.add_argument('--denoise_steps', default=200,
                        help='number of the denoising steps', type=int)
    parser.add_argument('--cfg_scale', default=10,
                        help='scale for classifier-free guidance', type=float)
    parser.add_argument('--seed', default=111,
                        help='seed for generating images', type=int)
    parser.add_argument('--image_style', default='anime',
                        choices=['anime', 'reality'],
                        help='sytles of the generated images', type=str)

    args = parser.parse_args()
    args.generator = torch.manual_seed(args.seed)

    main(args)