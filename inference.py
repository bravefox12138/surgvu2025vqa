"""
The following is a simple example algorithm.

It is meant to run within a container.

To run the container locally, you can call the following bash script:

  ./do_test_run.sh

This will start the inference and reads from ./test/input and writes to ./test/output

To save the container and prep it for upload to Grand-Challenge.org you can call:

  ./do_save.sh

Any container that shows the same behaviour will do, this is purely an example of how one COULD do it.

Reference the documentation to get details on the runtime environment on the platform:
https://grand-challenge.org/documentation/runtime-environment/

Happy programming!
"""

from pathlib import Path
import json
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch

INPUT_PATH = Path("/input")
OUTPUT_PATH = Path("/output")
# INPUT_PATH = Path("/data/zhaoxy/project/surgvu2025-category2-submission/test/input/interf0")
# OUTPUT_PATH = Path("/data/zhaoxy/project/surgvu2025-category2-submission/test/output/interf0")
RESOURCE_PATH = Path("resources")

# # Global model and processor variables
# model = None
# processor = None

def load_model():
    """Load the model and processor globally"""
    global model, processor
    
    # Load model from the checkpoint directory
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "/opt/app/model",  # Updated path for container
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",  # Force all model parts to use cuda:0
    )
    
    # Load processor
    # processor = AutoProcessor.from_pretrained("/opt/app/model")
    processor = AutoProcessor.from_pretrained("/opt/app/model")

def analyze_video(video_path, video_messages):
    """Analyze video and return description"""
    system_prompt = """
    You are an expert surgical video analysis assistant.

## Primary Objective
Analyze the surgical video and output findings in EXACTLY the following 4 steps, in order.
## Step 1 — Surgical Task
- Describe in detail the main surgical task being performed in the video.
- Keep the description precise and specific to the procedure context.

## Step 2 — Tools count Used
- Recognize the numeber of the visiable tools

## Step 3 — Tools Used name
- List **all** tools visible/used in the video. DO NOT list the tools not seen in the video.

## Step 4 — Surgical Action
- Describe the precise action being performed (e.g., suturing, dissecting, clipping, cauterizing, stapling, retracting).
- Avoid vague verbs like "working" or "handling".

## Step 5 — Anatomy / Organ(s) Involved
- Identify the anatomical structure(s) being operated on.
- Use standard anatomical terms.
- If multiple structures are involved, list them all.
---
## Output Format
Your output must follow this **exact format**:
Step 1 — Surgical Task: <detailed task description>  
Step 2 — Count the number of tools used: 
Step 3 - <comma-separated list of used tools name in the video>  
Step 4 — Surgical Action: <precise action description>
Step 5 — Anatomy: <list of anatomical structures>  

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
    return output_text[0] if output_text else ""

def answer_question(video_description, user_question, video_messages):
    """Answer question based on video description and question"""
    prompt = f"""
You are a precise question answering assistant.
You are only allowed to answer based on the Video description to answer the user's question.Do NOT add any information not contained in the description.If a tool is not explicitly listed in Step 3, your answer must be "No".
There should be logic before and after answering.


    Video description:
    {video_description}

    User question:
    {user_question}
    Answer concisely.
    Please output in clear, briefly,structured natural language in one sentence less than 10 words.
    """
    
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": user_question}
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
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
    return output_text[0] if output_text else ""

def run():
    # Load model first
    load_model()
    
    # The key is a tuple of the slugs of the input sockets
    interface_key = get_interface_key()
    print("Inputs: ", interface_key)
    # Lookup the handler for this particular set of sockets (i.e. the interface)
    handler = {
        (
            "endoscopic-robotic-surgery-video",
            "visual-context-question",
        ): interf0_handler,
    }[interface_key]

    # Call the handler
    return handler()


def interf0_handler():
    # Read the input
    input_endoscopic_robotic_surgery_video = INPUT_PATH / "endoscopic-robotic-surgery-video.mp4"
    input_visual_context_question = load_json_file(
        location=INPUT_PATH / "visual-context-question.json",
    )
    print('Question: ', json.dumps(input_visual_context_question, indent=4))

    # Prepare video messages for the model
    video_path = str(input_endoscopic_robotic_surgery_video)
    video_messages = [
        {"role": "user", "content": [
            {"type": "video",
             "video": video_path,
             "total_pixels": 10240 * 28 * 28,
             "min_pixels": 16 * 28 * 2,
             "fps": 0.15
            }
        ]}
    ]
    
    # Step 1: Analyze video to get description
    print("Analyzing video...")
    video_description = analyze_video(video_path, video_messages)
    print("Video description:\n", video_description)
    
    # Step 2: Answer the question using video description and question
    user_question = input_visual_context_question  # The question is directly a string
    print("Answering question:", user_question)
    answer = answer_question(video_description, user_question, video_messages)
    print("Answer:\n", answer)
    
    # # Save your output
    # output_visual_context_response = {
    #      answer
    # }
    
    write_json_file(
        location=OUTPUT_PATH / "visual-context-response.json",
        content=answer,
    )
    print('output saved to  ', OUTPUT_PATH)

    return 0


def get_interface_key():
    # The inputs.json is a system generated file that contains information about
    # the inputs that interface with the algorithm
    inputs = load_json_file(
        location=INPUT_PATH / "inputs.json",
    )
    print('These are the inputs:' , inputs)
    socket_slugs = [sv["interface"]["slug"] for sv in inputs]
    return tuple(sorted(socket_slugs))


def load_json_file(*, location):
    # Reads a json file
    with open(location, "r") as f:
        return json.loads(f.read())


def write_json_file(*, location, content):
    # Writes a json file
    with open(location, "w") as f:
        f.write(json.dumps(content, indent=4))


# Note to the developer:
#   the following function is very generic and should likely
#   be adopted to something more specific for your algorithm/challenge
def load_file(*, location):
    # Reads the content of a file
    with open(location) as f:
        return f.read()


def _show_torch_cuda_info():
    import torch

    print("=+=" * 10)
    print("Collecting Torch CUDA information")
    print(f"Torch CUDA is available: {(available := torch.cuda.is_available())}")
    if available:
        print(f"\tnumber of devices: {torch.cuda.device_count()}")
        print(f"\tcurrent device: { (current_device := torch.cuda.current_device())}")
        print(f"\tproperties: {torch.cuda.get_device_properties(current_device)}")
    print("=+=" * 10)


if __name__ == "__main__":
    raise SystemExit(run())
