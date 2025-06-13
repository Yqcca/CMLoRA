import ImageReward as RM
import json
import itertools
import os
from os.path import join
import argparse

from utils import load_lora_info, generate_combinations
from utils import get_prompt


def main(args):
    rm_model = RM.load("ImageReward-v1.0")
    args.lora_path = join(args.lora_path, args.image_style)
    lora_info = load_lora_info(args.image_style, args.lora_info_path)

    combinations = generate_combinations(lora_info, args.compos_num)
    init_prompt, negative_prompt = get_prompt(args.image_style)
    int_list = [0,1,2,3,5]

    for i in int_list:
        for j in int_list:
            win=0
            tie=0
            lose=0

            for combo in combinations:

                # set prompt
                triggers = [trigger for lora in combo for trigger in lora['trigger']]
                prompt = init_prompt + ', ' + ', '.join(triggers)
                
                # file name
                file_name1 = args.method1 + '_' + '_'.join([lora['id'] for lora in combo]) + '_' + str(i) + '.png'
                file_name2 = args.method2 + '_' + '_'.join([lora['id'] for lora in combo]) + '_' + str(j) + '.png'

                image1 = os.path.join(args.path1, file_name1)
                image2 = os.path.join(args.path2, file_name2)
                
                ranking, rewards = rm_model.inference_rank(prompt, [image1, image2])
                rewards = [round(r, 2) for r in rewards]

                if rewards[0] > rewards[1]:
                    win += 1
                elif rewards[0] == rewards[1]:
                    tie += 1
                else:
                    lose += 1
            print(f'Win:{win}, Tie:{tie}, Lose:{lose}\n')
            print(f'Method1: {args.method1} Interval:{i} VS Method2: {args.method2} Interval:{j}: {(win+tie)/(win+tie+lose)}\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Given LoRAs in the ComposLoRA benchmark, generate images with arbitrary combinations based on LoRAs'
    )

    # Arguments for composing LoRAs
    parser.add_argument('--compos_num', default=2,
                        help='number of elements to be combined in a single image', type=int)
    parser.add_argument('--lora_path', default='LoRA_Cache/models/lora',
                        help='path to store all LoRA models', type=str)
    parser.add_argument('--lora_info_path', default='lora_info.json',
                        help='path to stroe all LoRA information', type=str)
    parser.add_argument('--image_style', default='anime',
                        choices=['anime', 'reality'],
                        help='sytles of the generated images', type=str)
    parser.add_argument('--method1', default='composite', type=str)
    parser.add_argument('--method2', default='merge', type=str)
    parser.add_argument('--path1', default='ex/output_67_anime/2_elements', type=str)
    parser.add_argument('--path2', default='ex/output_m_anime/2_elements', type=str)

    args = parser.parse_args()

    main(args)
