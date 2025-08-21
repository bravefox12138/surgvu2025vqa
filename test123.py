from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch

# # default: Load the model on the available device(s)
# model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen2.5-VL-7B-Instruct", torch_dtype="auto", device_map="auto"
# )

# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "/data/zhaoxy/project/surgvu2025-category2-submission/model/checkpoint",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="cuda:6",
)

# default processor
processor = AutoProcessor.from_pretrained("/data/zhaoxy/project/surgvu2025-category2-submission/model/checkpoint")

# The default range for the number of visual tokens per image in the model is 4-16384.
# You can set min_pixels and max_pixels according to your needs, such as a token range of 256-1280, to balance performance and cost.
# min_pixels = 256*28*28
# max_pixels = 1280*28*28
# processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)



def analyze_video(video_messages):
    system_prompt = system_prompt = """
    You are an expert surgical video analysis assistant.

## Primary Objective
Analyze the surgical video and output findings in EXACTLY the following 4 steps, in order.
## Step 1 — Surgical Task
- Describe in detail the main surgical task being performed in the video.
- Keep the description precise and specific to the procedure context.

## Step 2 — Tools Used
- Recognize the numeber of the visiable tools
- List **all** tools visible/used in the video. DO NOT list the tools not seen in the video.

## Step 3 — Surgical Action
- Describe the precise action being performed (e.g., suturing, dissecting, clipping, cauterizing, stapling, retracting).
- Avoid vague verbs like "working" or "handling".

## Step 4 — Anatomy / Organ(s) Involved
- Identify the anatomical structure(s) being operated on.
- Use standard anatomical terms.
- If multiple structures are involved, list them all.
---
## Output Format
Your output must follow this **exact format**:
Step 1 — Surgical Task: <detailed task description>  
Step 2 — Tools Used: <Count the number of tools used, comma-separated list of used tools name in the video>  
Step 3 — Surgical Action: <precise action description>
Step 4 — Anatomy: <list of anatomical structures>  

---
"""
    # 只用视频消息 + system prompt
    messages = [{"role": "system", "content": system_prompt}] + video_messages 

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text

def answer_question(video_description, user_question, video_messages):
    prompt = f"""
You are a precise surgical-video question answering assistant.
Use the video and following video description to answer the user's question.
There should be logic before and after answering.
Do NOT add any information not contained in the description.
tools has both commercial name and ground truth name, and they share the same meaning. the tool name must standard referring to Mapping Table,  
### Mapping Table (Do not change or extend) groundtruth name : [commercial names]
    "curved scissors": ["curved scissors"],
    "cadiere forceps": ["cadiere forceps"],
    "permanent cautery hook/spatula": [
        "permanent cautery hook",
        "permanent cautery spatula",
        "cautery hook"
    ],
    "stapler": [
        "stapler 30 curved-tip",
        "sureform stapler 45",
        "stapler 45 curved-tip",
        "stapler 45",
        "sureform stapler 60"
    ],
    "needle driver": [
        "wristed needle driver",
        "large suturecut needle driver",
        "mega needle driver",
        "mega suturecut needle driver",
        "large needle driver"
    ],
    "bipolar forceps": [
        "maryland bipolar",
        "micro bipolar forceps",
        "fenestrated bipolar",
        "maryland bipolar forceps",
        "fenestrated bipolar forceps"
    ],
    "suction irrigator": ["endowrist suction irrigator"],
    "synchroseal": ["synchroseal"],
    "bipolar dissector": [
        "maryland dissector",
        "s curved bipolar dissector"
    ],
    "clip applier": [
        "large clip applier",
        "small clip applier",
        "medium-large clip"
    ],
    "nan(camera in)": [
        "0° endoscope",
        "30° endoscope"
    ],
    "monopolar curved scissors": ["monopolar curved scissors"],
    "crocodile grasper": ["crocodile grasper"],
    "vessel sealer": [
        "vessel sealer",
        "vessel sealer extend"
    ],
    "tip-up fenestrated grasper": ["tip-up fenestrated grasper"],
    "force bipolar": ["force bipolar"],
    "prograsp forceps": ["prograsp forceps"],
    "grasping retractor": ["small grasping retractor"],
    "tenaculum forceps": ["tenaculum forceps"],
    "potts scissors": ["potts scissors"]


Video description:
{video_description}

User question:
{user_question}
Answer concisely.
Please output in clear, briefly,structured natural language in one sentence less than 10 words.
"""
    # 组成对话，带上视频消息
    # messages = [
    #     {"role": "system", "content": prompt},
    #     {"role": "user", "content": user_question}
    # ] + video_messages
    # video_msgs, video_kwargs = prepare_message_for_vllm(messages)
    # response = vlm.invoke(video_msgs, extra_body={"mm_processor_kwargs": video_kwargs})
    
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": user_question}
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text

if __name__ == "__main__":
    input_video_msgs = [
        {"role": "user", "content": [
            {"type": "video",
             "video": "/data/datasets/sur25/SURGVU25_cat_2_sample_set_public/case122/case122_blur.mp4",
             "total_pixels": 20480 * 28 * 28,
             "min_pixels": 16 * 28 * 2,
             "fps": 2.0
            }
        ]}
    ]
    # 第一步：只用视频让模型描述
    video_description = analyze_video(input_video_msgs)
    print("Video description:\n", video_description)

    # 第二步：用视频描述+用户问题+视频一起问
    question = "Are there forceps being used here?"
    answer = answer_question(video_description, question, input_video_msgs)
    print("Answer:\n", answer)


