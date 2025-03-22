import os
import torch
import argparse
from diffusers import DiffusionPipeline, AutoencoderKL
from diffusers import DPMSolverMultistepScheduler
from callbacks_sa import make_callback
from utils import load_lora_info, generate_combinations, get_prompt
import json
from os.path import join
from tqdm import tqdm

from LoRA_Cache.pipelines.stable_diffusion.pipeline_offline import StableDiffusionPipeline

def main(args):
    if args.image_style == 'anime':
        model_name = 'gsdf/Counterfeit-V2.5'
    else:
        model_name = 'SG161222/Realistic_Vision_V5.1_noVAE'

    pipeline = StableDiffusionPipeline.from_pretrained(
        model_name,
        # torch_dtype=torch.float16,
        use_safetensors=True,
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

    args.lora_path = join(args.lora_path, args.image_style)
    print(args.lora_path)
    lora_info = load_lora_info(args.image_style, args.lora_info_path)

    for element in list(lora_info.keys()):
        for lora in lora_info[element]:
            pipeline.load_lora_weights(
                args.lora_path,
                weight_name=lora['id'] + '.safetensors',
                adapter_name=lora['id']
            )
    # generate all combinations that can be composed
    combinations = generate_combinations(lora_info, 2)

    # prompt initialization
    init_prompt, negative_prompt = get_prompt(args.image_style)

    # generate images for each combination based on LoRAs
    for combo in tqdm(combinations):
        cur_loras = [lora['id'] for lora in combo]

        # set prompt
        triggers = [trigger for lora in combo for trigger in lora['trigger']]
        prompt = init_prompt + ', ' + ', '.join(triggers)
        
        # set LoRAs
        if args.method == "switch":
            pipeline.set_adapters([cur_loras[0]])
            switch_callback = make_callback(args.switch_step,
                                        cur_loras)
        elif args.method == "merge":
            pipeline.set_adapters(cur_loras)
            switch_callback = None
        else:
            pipeline.set_adapters(cur_loras)
            switch_callback = None

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
                cross_attention_kwargs={"scale": args.lora_scale},
                callback_on_step_end=switch_callback,
                cache_interval=args.interval[inter],
                cache_layer_id=0,
                cache_block_id=1,
                lora_composite=True if args.method == "composite" else False,
                combo = combo,
            ).images[0]
            
            image.save(os.path.join('images', f'{args.method}_{args.interval[inter]}.png'), 'PNG')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Example code for multi-LoRA composition'
    )

    # Arguments for composing LoRAs
    parser.add_argument('--method', default='composite',
                        choices=['merge', 'switch', 'composite'],
                        help='methods for combining LoRAs', type=str)
    parser.add_argument('--lora_path', default='LoRA_Cache/models/lora',
                        help='path to store all LoRAs', type=str)
    parser.add_argument('--lora_scale', default=1,
                        help='scale of each LoRA when generating images', type=float)
    parser.add_argument('--switch_step', default=5,
                        help='number of steps to switch LoRA during denoising, applicable only in the switch method', type=int)
    parser.add_argument('--lora_info_path', default='example_info.json',
                        help='path to stroe all LoRA information', type=str)
    parser.add_argument('--interval', default='[1,2,3,5]',
                        help='number of steps to cache LoRA during denoising', type=json.loads)

    # Arguments for generating images
    parser.add_argument('--height', default=512,
                        help='height of the generated images', type=int)
    parser.add_argument('--width', default=512,
                        help='width of the generated images', type=int)
    parser.add_argument('--denoise_steps', default=220,
                        help='number of the denoising steps', type=int)
    parser.add_argument('--cfg_scale', default=8.5,
                        help='scale for classifier-free guidance', type=float)
    parser.add_argument('--image_style', default='anime',
                        choices=['anime', 'reality'],
                        help='sytles of the generated images', type=str)
    parser.add_argument('--seed', default=14,
                        help='seed for generating images', type=int)

    args = parser.parse_args()
    args.generator = torch.manual_seed(args.seed)

    main(args)