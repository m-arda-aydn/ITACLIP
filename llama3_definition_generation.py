import torch
import os
import time 

cache_dir = YOUR_CACHE_DIR
os.environ['HF_HOME'] = cache_dir
print(os.getenv('HF_HOME'))
import transformers

dataset_name = 'coco_object'
assert dataset_name in ['coco_stuff','coco_object','voc21','context60']
bg = False
if dataset_name in ['coco_object','voc21','context60']:
    bg = True

access_token = YOUR_HF_TOKEN
path = f'./configs/cls_{dataset_name}.txt'
model_id = "meta-llama/Meta-Llama-3-8B-Instruct" # different LLama models can be used here
txt_path = f'/mnt/disk2/arda_efe/graduation/llama_generated_texts/{dataset_name}_definitions.txt'

with open(path, 'r') as f:
    if bg:
        next(f)
    name_sets = f.readlines()

for i, name in enumerate(name_sets):
    name_sets[i] = name.replace('\n','')
    if len(name_sets[i].split(',')) > 1:
        name_sets[i] = name_sets[i].split(',')[0]

print(name_sets)
print(len(name_sets))

pipeline = transformers.pipeline(
    "text-generation", model=model_id, model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto", token=access_token)

start_time = time.time()

for class_name in name_sets:

    messages = [
        {"role": "system", "content": "Give a brief definition of prompted word like given example definitions: house >= a building that people, usually one family, live in; car >= a road vehicle with an engine, four wheels, and seats for a small number of people; (no more than 50 words, do not use extra words other than the definition of given word)"},
    ]
    messages.append({"role": "user", "content": f"{class_name} >="})

    print('class name: ', class_name)
    prompt = pipeline.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
    )

    terminators = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = pipeline(
        prompt,
        max_new_tokens=256,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )

    print(outputs[0]["generated_text"][len(prompt):])

    with open(txt_path, 'a') as file:
        file.write(f'{class_name} >=')
        file.write(outputs[0]["generated_text"][len(prompt):])
        file.write('\n')

end_time = time.time()

print('total time: ', end_time - start_time)

