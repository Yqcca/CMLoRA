def check_anime_lora(adapters):
    cha = ["character_1", "character_2", "character_3"]
    cloth = ["clothing_1", "clothing_2"]
    obj = ["object_1", "object_2"]
    sty = ["style_1", "style_2"]
    back = ["background_1", "background_2"]
    real_list = []
    abs_list = []
    for i in adapters:
        if i in sty:
            real_list.append(i)
        elif i in cha:
            real_list.append(i)
        elif i in cloth:
            real_list.append(i)
        elif i in obj:
            real_list.append(i)
        elif i in back:
            abs_list.append(i)
    return real_list + abs_list

threshold = 120

def make_callback(switch_step, loras):
    def switch_callback(pipeline, step_index, timestep, callback_kwargs):
        callback_outputs = {}
        alist = check_anime_lora(loras)
        if len(loras)>2:
            if step_index > 0 and step_index % switch_step == 0:
                if step_index <=120:
                    for cur_lora_index, lora in enumerate(alist[:-1]):
                        if lora in pipeline.get_active_adapters():
                            next_lora_index = (cur_lora_index + 1) % len(alist[:-1])
                            pipeline.set_adapters(alist[:-1][next_lora_index])
                            break
                else:
                    for cur_lora_index, lora in enumerate(alist[-1]):
                        if lora in pipeline.get_active_adapters():
                            next_lora_index = (cur_lora_index + 1) % len(alist[-1])
                            pipeline.set_adapters(alist[-1][next_lora_index])
                            break
        else:
            for cur_lora_index, lora in enumerate(loras):
                if lora in pipeline.get_active_adapters():
                    next_lora_index = (cur_lora_index + 1) % len(loras)
                    pipeline.set_adapters(loras[next_lora_index])
                    break
        return callback_outputs
    return switch_callback