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
from qwen_vl_utils import process_vision_info, vision_process
import torch

# INPUT_PATH = Path("/input")
# OUTPUT_PATH = Path("/output")
INPUT_PATH = Path(
    "/data/lxy/code/surgvu2025-category2-submission/test/input/interf0")
OUTPUT_PATH = Path(
    "/data/lxy/code/surgvu2025-category2-submission/test/output/interf0")
RESOURCE_PATH = Path("resources")

# # Global model and processor variables
# model = None
# processor = None

def log_cuda_memory(message):
    print(f"========== {message} ==========")
    print("Allocated:", torch.cuda.memory_allocated(device="cuda:0")/1024**2, "MB")
    print("Reserved :", torch.cuda.memory_reserved(device="cuda:0")/1024**2, "MB")
    print("Max Allocated:", torch.cuda.max_memory_allocated(device="cuda:0")/1024**2, "MB")


def load_model():
    """Load the model and processor globally"""
    global model, processor
    
    # Load model from the checkpoint directory
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        # "/opt/app/model",  # Updated path for container
        "/data/lxy/code/surgvu2025-category2-submission/model",
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",  # Force all model parts to use cuda:0
    )

    model.eval()
    # Load processor
    # processor = AutoProcessor.from_pretrained("/opt/app/model")
    processor = AutoProcessor.from_pretrained(
        "/data/lxy/code/surgvu2025-category2-submission/model")


def build_qwen_input_by_file(file_path, input_text, vision_type="video", frames=8):

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": vision_type,
                    vision_type: f"{file_path}",
                    "resized_height": 480,
                    "resized_width": 854,
                },
                {"type": "text", "text": input_text},
            ],
        }
    ]
    try:
        # 处理视频
        vision_process.FPS_MAX_FRAMES = frames
        vision_process.FPS_MIN_FRAMES = frames
        image_inputs, video_inputs = process_vision_info(
            messages)  # 获取数据（预处理过）

        print(f'input tensor shape {video_inputs[0].shape}')
    except Exception as e:
        print(f"处理视频文件时出错: {e}")
        return None

    # 获取文本
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # 获取输入
    inputs = processor(
        text=[text],
        # images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    return inputs


def infer(file_path, input_text):
    inputs = build_qwen_input_by_file(
         file_path, input_text, vision_type="video", frames=8).to("cuda:0")

    # 打印inputs的属性
    for k, v in inputs.items():
        if hasattr(v, "shape"):
            print(f"{k:20s} shape={tuple(v.shape)}, dtype={v.dtype}, device={v.device}")
        else:
            print(f"{k:20s} value={v}")

    log_cuda_memory("before generate")
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    log_cuda_memory("after generate")
    return output_text[0] if output_text else ""


def run():
    log_cuda_memory("before load model")
    # Load model first
    load_model()
    log_cuda_memory("after load model")
    
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
    
    user_question = input_visual_context_question
    print("Answering question:", user_question)
    answer = infer(video_path, input_visual_context_question)
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

    peak_memory = torch.cuda.max_memory_allocated(device="cuda:0") / 1024**2
    print("Peak memory usage:", peak_memory, "MiB")
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
