import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer
import os
from os.path import join
import argparse
from utils import load_lora_info, generate_combinations
from utils import get_prompt

torch.manual_seed(0)

def main(args):
    model = AutoModel.from_pretrained('openbmb/MiniCPM-V-2_6', trust_remote_code=True,
        attn_implementation='sdpa', torch_dtype=torch.bfloat16) # sdpa or flash_attention_2, no eager
    model = model.eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained('openbmb/MiniCPM-V-2_6', trust_remote_code=True)

    args.lora_path = join(args.lora_path, args.image_style)
    lora_info = load_lora_info(args.image_style, args.lora_info_path)

    combinations = generate_combinations(lora_info, args.compos_num)
    init_prompt, negative_prompt = get_prompt(args.image_style)
    int_list = [0,1,2,3,5]

    # Number of in-context learning samples
    num = 5

    # Provide the ground-truth scores based human-based evaluation or LLM-based evaluation
    image_1_integration = []
    image_1_consistency = []
    image_1_accuracy = []
    image_1_appeal = []
    image_2_integration = []
    image_2_consistency = []
    image_2_accuracy = []
    image_2_appeal = []

    # Provide the specific evluation senario and criteria
    task_description = f"1) Element Integration:\nHow seamlessly different elements are combined within the image.\n\nCriteria:\n- Visual Cohesion: Evaluate whether the elements appear as part of a unified scene, rather than as disjointed parts.\n- Object Overlap and Interaction: Check for natural overlaps and interactions between objects, ensuring no awkward placements or intersections.\n\n2) Spatial Consistency:\nUniformity in style, lighting, and perspective across all elements.\n\nCriteria:\n- Stylistic Uniformity: Ensure that all elements share a consistent artistic style (e.g., realism, cartoonish).\n- Lighting and Shadows: Verify that light sources and shadow directions are consistent, contributing to a realistic portrayal.\n- Perspective Alignment: Confirm that elements adhere to a shared perspective, with no mismatched viewpoints.\n\n3) Semantic Accuracy:\nCorrect interpretation and representation of each element as described in the prompt.\n\nCriteria:\n- Object Accuracy: Objects should align with their descriptions in terms of type, attributes, and context.\n- Action and Interaction: Actions or interactions between objects should be depicted accurately and appropriately.\n\n4) Aesthetic Quality:\nOverall visual appeal and artistic quality of the generated image.\n\nCriteria:\n- Color Harmony: The use of color palettes should be visually pleasing and fitting for the scene.\n- Composition Balance: Elements should be arranged in a balanced way to create an engaging and harmonious composition.\n- Clarity and Sharpness: The image should be clear, with well-defined elements, free from unwanted blurriness or distortion."
    # provide more specific task description
    advance_task_description = ''

    for i in int_list:
        for j in int_list:
            for combo in range(len(combinations)):
                if combo < num:
                    triggers = [trigger for lora in combinations[combo] for trigger in lora['trigger']]
                    prompt = init_prompt + ', ' + ', '.join(triggers)
                    file_name1 = args.method1 + '_' + '_'.join([lora['id'] for lora in combinations[combo]]) + '_' + str(i) + '.png'
                    file_name2 = args.method2 + '_' + '_'.join([lora['id'] for lora in combinations[combo]]) + '_' + str(j) + '.png'

                    image1 = os.path.join(args.path1, file_name1)
                    image2 = os.path.join(args.path2, file_name2)

                    image1 = Image.open(image1).convert('RGB')
                    image2 = Image.open(image2).convert('RGB')

                    # First round chat
                    question = f'Please assist in comparatively evaluating two text-to-image models based on their ability to compose different elements into a single image. The expected conpects in the image include: {prompt}. Key attributes for comparison include: {task_description}. {advance_task_description} \
                    The evaluation should be based on these factors: 1) Integration of Elements 2) Consistency in Composition 3) Accuracy in Depiction 4) Visual Appeal. \
                    Kindly provide the ratings in the following format: Image 1: Integration: [score]/10, Consistency: [score]/10, Accuracy: [score]/10, Visual Appeal: [score]/10; \
                    Image 2: Integration: [score]/10, Consistency: [score]/10, Accuracy: [score]/10, Visual Appeal: [score]/10. \
                    Your evaluation should be detailed, with scores that avoid ties between the models on both aspects.'
                    answer = f'Image 1: Integration: {image_1_integration[combo]}/10, Consistency: {image_1_consistency[combo]}/10, Accuracy: {image_1_accuracy[combo]}/10, Visual Appeal: {image_1_appeal[combo]}/10.\n'\
                     f'Image 2: Integration: {image_2_integration[combo]}/10, Consistency: {image_2_consistency[combo]}/10, Accuracy: {image_2_accuracy[combo]}/10, Visual Appeal: {image_2_appeal[combo]}/10.'
                    msgs = [
                        {'role': 'user', 'content': [image1, image2, question]}, {'role': 'assistant', 'content': [answer]},
                        {'role': 'user', 'content': [image1, image2, question]}
                        ]

                    answer = model.chat(
                        image=None,
                        msgs=msgs,
                        tokenizer=tokenizer
                    )
                    print(f'Method1: {args.method1} Interval:{i} VS Method2: {args.method2} Interval:{j}.\n{answer}')
                else:
                    triggers = [trigger for lora in combinations[combo-1] for trigger in lora['trigger']]
                    prompt = init_prompt + ', ' + ', '.join(triggers)
                    file_name1 = args.method1 + '_' + '_'.join([lora['id'] for lora in combinations[combo-1]]) + '_' + str(i) + '.png'
                    file_name2 = args.method2 + '_' + '_'.join([lora['id'] for lora in combinations[combo-1]]) + '_' + str(j) + '.png'

                    image1 = os.path.join(args.path1, file_name1)
                    image2 = os.path.join(args.path2, file_name2)

                    image1 = Image.open(image1).convert('RGB')
                    image2 = Image.open(image2).convert('RGB')

                    # First round chat
                    question = f'Please assist in comparatively evaluating two text-to-image models based on their ability to compose different elements into a single image. The expected conpects in the image include: {prompt}. Key attributes for comparison include: {task_description}. {advance_task_description} \
                    The evaluation should be based on these factors: 1) Integration of Elements 2) Consistency in Composition 3) Accuracy in Depiction 4) Visual Appeal. \
                    Kindly provide the ratings in the following format: Model A: Integration: [score]/10, Consistency: [score]/10, Accuracy: [score]/10, Visual Appeal: [score]/10; \
                    Model B: Integration: [score]/10, Consistency: [score]/10, Accuracy: [score]/10, Visual Appeal: [score]/10. \
                    Your evaluation should be detailed, with scores that avoid ties between the models on both aspects.'
                    answer = f'Image 1: Integration: {image_1_integration[combo]}/10, Consistency: {image_1_consistency[combo]}/10, Accuracy: {image_1_accuracy[combo]}/10, Visual Appeal: {image_1_appeal[combo]}/10.\n'\
                     f'Image 2: Integration: {image_2_integration[combo]}/10, Consistency: {image_2_consistency[combo]}/10, Accuracy: {image_2_accuracy[combo]}/10, Visual Appeal: {image_2_appeal[combo]}/10.'

                    triggers1 = [trigger for lora in combinations[combo] for trigger in lora['trigger']]
                    prompt1 = init_prompt + ', ' + ', '.join(triggers1)
                    file_name3 = args.method1 + '_' + '_'.join([lora['id'] for lora in combinations[combo]]) + '_' + str(i) + '.png'
                    file_name4 = args.method2 + '_' + '_'.join([lora['id'] for lora in combinations[combo]]) + '_' + str(j) + '.png'

                    image3 = os.path.join(args.path1, file_name3)
                    image4 = os.path.join(args.path2, file_name4)

                    image3 = Image.open(image3).convert('RGB')
                    image4 = Image.open(image4).convert('RGB')

                    # First round chat
                    question = f'Please assist in comparatively evaluating two text-to-image models based on their ability to compose different elements into a single image. The expected conpects in the image include: {prompt}. Key attributes for comparison include: {task_description}. {advance_task_description} \
                    The evaluation should be based on these factors: 1) Integration of Elements 2) Consistency in Composition 3) Accuracy in Depiction 4) Visual Appeal. \
                    Kindly provide the ratings in the following format: Model A: Integration: [score]/10, Consistency: [score]/10, Accuracy: [score]/10, Visual Appeal: [score]/10; \
                    Model B: Integration: [score]/10, Consistency: [score]/10, Accuracy: [score]/10, Visual Appeal: [score]/10. \
                    Your evaluation should be detailed, with scores that avoid ties between the models on both aspects.'

                    msgs = [
                        {'role': 'user', 'content': [image1, image2, question]}, {'role': 'assistant', 'content': [answer]},
                        {'role': 'user', 'content': [image3, image4, question]}
                        ]

                    answer = model.chat(
                        image=None,
                        msgs=msgs,
                        tokenizer=tokenizer
                    )
                    print(f'Method1: {args.method1} Interval:{i} VS Method2: {args.method2} Interval:{j}.\n{answer}')
                combo += 1         

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
    parser.add_argument('--path1', default='ex/output_anime/2_elements', type=str)
    parser.add_argument('--path2', default='ex/output_m_anime/2_elements', type=str)

    args = parser.parse_args()

    main(args)