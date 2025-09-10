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
import os
import csv
import configparser
import logging
from timm.models import create_model
import torchvision.transforms as transforms

INPUT_PATH = Path("/input")
OUTPUT_PATH = Path("/output")
MODEL_PATH = Path("/opt/app/model")

# INPUT_PATH = Path(
#     "/data/lxy/code/surgvu2025-category2-submission/test/input/interf0")
# OUTPUT_PATH = Path(
#     "/data/lxy/code/surgvu2025-category2-submission/test/output/interf0")
# MODEL_PATH = Path("/data/lxy/code/surgvu2025-category2-submission/model")

RESOURCE_PATH = Path("resources")

# # Global model and processor variables
# model = None
# processor = None

def log_cuda_memory(message):
    print(f"========== {message} ==========")
    print("Allocated:", torch.cuda.memory_allocated(device="cuda:0")/1024**2, "MB")
    print("Reserved :", torch.cuda.memory_reserved(device="cuda:0")/1024**2, "MB")
    print("Max Allocated:", torch.cuda.max_memory_allocated(device="cuda:0")/1024**2, "MB")

    # 打印 GPU 显存使用情况
    os.system("nvidia-smi")


def load_model():
    """Load the model and processor globally"""
    
    global config
    config = configparser.ConfigParser()
    config.read(os.path.join(MODEL_PATH, "alg.cfg"), encoding='utf-8')

    global det_model, cls_model, cls_transform, device
    det_model_path = os.path.join(MODEL_PATH, "last.pt")
    cls_model_path = os.path.join(MODEL_PATH, "checkpoint-best-ema_224.pth")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 初始化检测模型
    from ultralytics.utils import LOGGER
    LOGGER.setLevel(logging.ERROR)  # 只显示错误
    from ultralytics import YOLO
    det_model = YOLO(det_model_path)

    # 分类模型可选，如果不需要可以设置为 None
    cls_model = create_model("resnet50", pretrained=False, num_classes=12)
    import numpy as np
    # with torch.serialization.safe_globals([np.core.multiarray.scalar]):
    checkpoint = torch.load(
        cls_model_path, map_location="cpu", weights_only=False)['model']
    # 这里假设 load_state_dict 函数已定义
    load_state_dict(cls_model, checkpoint)
    cls_model.to(device)
    cls_model.eval()
    cls_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    global model, processor
    
    # Load model from the checkpoint directory
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,  # Updated path for container
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",  # Force all model parts to use cuda:0
    )

    model.eval()
    # Load processor
    processor = AutoProcessor.from_pretrained(MODEL_PATH, use_fast=True)


def build_qwen_input_by_file(messages, frames=4):
    try:
        # 处理视频
        vision_process.FPS_MAX_FRAMES = frames
        vision_process.FPS_MIN_FRAMES = frames
        image_inputs, video_inputs = process_vision_info(
            messages)  # 获取数据（预处理过）

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

def infer_by_message(messages, model):
    inputs = build_qwen_input_by_file(messages, frames=4).to("cuda")
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs, max_new_tokens=1024, do_sample=False, num_beams=1)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text[0]


def load_state_dict(model, state_dict, prefix='', ignore_missing="relative_position_index"):
    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(
            prefix[:-1], {})
        module._load_from_state_dict(
            state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(model, prefix=prefix)

    warn_missing_keys = []
    ignore_missing_keys = []
    for key in missing_keys:
        keep_flag = True
        for ignore_key in ignore_missing.split('|'):
            if ignore_key in key:
                keep_flag = False
                break
        if keep_flag:
            warn_missing_keys.append(key)
        else:
            ignore_missing_keys.append(key)

    missing_keys = warn_missing_keys

    if len(missing_keys) > 0:
        print("Weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, missing_keys))
    if len(unexpected_keys) > 0:
        print("Weights from pretrained model not used in {}: {}".format(
            model.__class__.__name__, unexpected_keys))
    if len(ignore_missing_keys) > 0:
        print("Ignored weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, ignore_missing_keys))
    if len(error_msgs) > 0:
        print('\n'.join(error_msgs))

def extract_tools_list(video_path, det_model, id_label_dict, cls_model=None, cls_transform=None, device="cuda"):
    """
    输入视频路径，返回视频中出现过的手术工具列表（去重）。
    """
    import cv2
    import torch
    import numpy as np
    from PIL import Image
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))  # 帧率
    frame_interval = max(fps * 1, 1)      # 每 2 秒取一帧
    frame_idx = 0

    tools_set = set()
    prev_tools = set()  # 保存上一帧的检测结果

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        if frame_idx % frame_interval == 0:
            current_tools = set()

            # 检测
            det_res = det_model.predict(frame, conf=0.65, iou=0.3)[0]
            boxes = det_res.boxes.xyxy.cpu().numpy().astype(np.int32)
            clses = det_res.boxes.cls.cpu().numpy().astype(np.int32)

            for i in range(len(boxes)):
                det_cls = clses[i]
                det_label = id_label_dict[det_cls]

                # 可选分类模型
                if cls_model is not None and cls_transform is not None:
                    x1, y1, x2, y2 = boxes[i]
                    h, w = frame.shape[:2]
                    bw, bh = x2 - x1, y2 - y1
                    xmin = max(x1 - bw // 2, 0)
                    ymin = max(y1 - bh // 2, 0)
                    xmax = min(x2 + bw // 2, w)
                    ymax = min(y2 + bh // 2, h)
                    if xmin < xmax and ymin < ymax:
                        crop = frame[ymin:ymax, xmin:xmax, :]
                        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                        crop_rgb = Image.fromarray(crop_rgb)
                        image_trans = cls_transform(crop_rgb).unsqueeze(0).to(device)
                        with torch.no_grad():
                            res = cls_model(image_trans)
                            output = torch.softmax(res, dim=1).detach().cpu().numpy()
                            cls_id = np.argmax(output, axis=1)[0]
                            det_label = id_label_dict[cls_id]

                current_tools.add(det_label)

            # 取和上一帧的交集 → 连续两帧都出现的工具才加入最终结果
            stable_tools = prev_tools & current_tools
            tools_set.update(stable_tools)

            # 更新上一帧
            prev_tools = current_tools

        frame_idx += 1

    cap.release()
    return sorted(list(tools_set))

def detect_tool_list(video_path):
    id_label_dict = {0: 'needle driver',
                     1: 'monopolar curved scissors',
                     2: 'force bipolar',
                     3: 'clip applier',
                     4: 'cadiere forceps',
                     5: 'bipolar forceps',
                     6: 'vessel sealer',
                     7: 'permanent cautery hook spatula',
                     8: 'prograsp forceps',
                     9: 'stapler',
                     10: 'grasping retractor',
                     11: 'tip-up fenestrated grasper'}

    tools_list = extract_tools_list(
        video_path, det_model, id_label_dict, cls_model, cls_transform, device)
    print("Detected tools:", tools_list)

    return tools_list

def load_commercial2gt(csv_path):
    """
    读取 CSV, 返回 commercial2gt 映射 (lowercase key)
    """
    mapping = {}
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            c_name = row["commercial_toolname"].strip().lower()
            g_name = row["groundtruth_toolname"].strip()
            mapping[c_name] = g_name
    return mapping

def merge_tools(video_description, tools_list, commercial2gt):
    """
    合并检测模型 tools_list 和 Qwen 输出的 JSON
    - tools_list 决定最终有哪些工具
    - 如果 Qwen 有对应描述，就优先保留它的 commercial 名
    - 否则用 groundtruth 名
    """
    def normalize_name(name):
        return name.strip().lower()

    # Step 1. 把检测模型输出 tools_list 映射到 groundtruth
    gt_tools = set()
    for tool in tools_list:
        tool_norm = normalize_name(tool)
        gt_name = commercial2gt.get(tool_norm, tool)
        gt_tools.add(gt_name)

    # Step 2. 遍历 Qwen 输出
    desc_dict = {}
    commercial_name_dict = {}
    if "used_tools_and_function" in video_description:
        for name, func in video_description["used_tools_and_function"].items():
            name_norm = normalize_name(name)
            gt_name = commercial2gt.get(name_norm, name)
            if gt_name in gt_tools:  # 必须检测模型确认过
                desc_dict[gt_name] = func
                commercial_name_dict[gt_name] = name  # 保留 Qwen 原始 commercial 名

    # Step 3. 合并
    merged = {}
    for gt_name in gt_tools:
        key_name = commercial_name_dict.get(
            gt_name, gt_name)  # 优先保留 Qwen commercial 名
        merged[key_name] = desc_dict.get(gt_name, " ")

    # Step 4. 为新的tools list添加function
    gt_tools_function_map = {
        "monopolar curved scissors": "Used for tissue cutting and dissection, providing monopolar electrosurgical energy for cutting and coagulation.",
        "force bipolar": "Used for tissue cutting and dissection, providing bipolar electrosurgical energy for cutting and coagulation.",
        "permanent cautery hook spatula": "Used for tissue dissection, blunt dissection, and cauterization.",
        "stapler": "Used for tissue or vessel transection, sealing, and stapling.",
        "needle driver": "Used for holding needles, suturing, and knot tying.",
        "bipolar forceps": "Used for grasping tissue and providing bipolar coagulation for hemostasis.",
        "suction irrigator": "Used for irrigating, aspirating fluids, and clearing the surgical field.",
        "synchroseal": "Used for sealing, cutting, and achieving hemostasis in tissue.",
        "bipolar dissector": "Used for tissue dissection with bipolar coagulation capability.",
        "clip applier": "Used for vessel or duct occlusion through clip application.",
        "cadiere forceps": "Used for grasping, retracting, and dissecting tissue.",
        "crocodile grasper": "Used for tissue grasping and traction.",
        "vessel sealer": "Used for vessel sealing, hemostasis, and tissue division.",
        "tip-up fenestrated grasper": "Used for tissue retraction, exposure, and dissection.",
        "prograsp forceps": "Used for strong grasping and holding of thick or firm tissue.",
        "grasping retractor": "Used for tissue retraction and exposure of the surgical field.",
        "tenaculum forceps": "Used for firmly grasping and stabilizing tissue or organs.",
        "potts scissors": "Used for fine dissection and cutting of vessels or ducts."
    }
    
    for key, value in merged.items():
        if value is None or value == " " or value == "":
            if key in gt_tools_function_map:
                merged[key] = gt_tools_function_map[key]
            else:
                merged[key] = " "

    # 现在是 dict，可以安全更新
    video_description["used_tools_and_function"] = merged
    return video_description

def filter_tools(file_path, description_json):
    use_tools_filter = config.getboolean('ALG', 'use_tools_filter', fallback=True)
    if use_tools_filter:
        tools_list = detect_tool_list(file_path)
        commercial2gt = load_commercial2gt(
            os.path.join(MODEL_PATH, "toolname_mapping_filter.csv"))
        description_json = merge_tools(description_json, tools_list, commercial2gt)
    return description_json


def deepthink_infer_by_file(file_path, input_text):
    description_question = [
        "What is the used_tools_and_function? answer in json format",
        "What is the operated_organ_and_tissue? answer in json format",
        "What is the task_name, task_description and matched_description? answer in json format",
        "Analyze the surgical procedure in the video and provide a structured JSON output with keys: used_tools_and_function, operated_organ_and_tissue, task_name, task_description, matched_description"
    ]

    # 1. 分析question的语法结构
    prompt = f"""You are a linguistic analyzer.  
            Your task is to analyze the grammatical structure of the given English sentence.

            Step 1: Determine the sentence structure: 
            - If the sentence uses a linking verb (like is, are, was, were, seem, become) connecting the subject to a predicative, classify it as Subject–Linking Verb–Predicative (S-LV-P).

            Step 2: Extract components according to the structure:

            - subject (主语)  
            - linking_verb (系动词)  
            - predicative / complement (表语)  
            - adverbial (状语)  
            - complement (补语)

            Rules:  
            - If a component does not exist, set its value to null.  
            - Output only valid JSON, nothing else.

            Example 1:
            Input: "Is a scissor among the tools?"  
            Structure: S-LV-P  
            Output:  
            {{
            "structure": "S-LV-P",
            "subject": "a scissor",
            "linking_verb": "is",
            "predicative": "among the tools",
            "adverbial": null,
            "complement": null
            }}

            Example 2:
            Input: "What object is being manipulated/used?"  
            Structure: S-LV-P  
            Output:  
            {{
            "structure": "S-LV-P",
            "subject": "What object",
            "linking_verb": "is",
            "predicative": "being manipulated/used",
            "adverbial": null,
            "complement": null
            }}

            Example 3:
            Input: "What is the purpose of using scissor in this procedure?"  
            Structure: S-LV-P  
            Output:  
            {{
            "structure": "S-LV-P",
            "subject": "the purpose of using scissor ",
            "linking_verb": "is",
            "predicative": "what",
            "adverbial": "in this procedure",
            "complement": null
            }}

            Now analyze the following sentence:  
            "{input_text}"

        """
    messages = []
    messages.append({
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
        ],
    })

    structure = infer_by_message(messages, model)

    try:
        structure_json = json.loads(structure)
        # 1.1 去除特殊疑问词
        special_words = ["what", "which", "who", "whom", "whose", "when", "where", "why", "how", "how many", "how much",
                         "how long", "how often", "how far", "how big", "how small", "how heavy", "how light",
                         "how tall", "how short", "how wide", "how deep", "how thick", "how thin", "how long",
                         "how short", "how wide", "how deep", "how thick", "how thin"]
        if input_text.lower().split(" ")[0] in special_words:
            structure_json["type"] = "special"
        else:
            structure_json["type"] = "normal"

        if structure_json["structure"] == "S-LV-P":
            for key, value in structure_json.items():
                if value is None:
                    continue
                for word in special_words:
                    if word in value.lower().split(" "):
                        structure_json[key] = value.lower().replace(word, "")
        # 1.2 去除表语中的状语
        if structure_json["structure"] == "S-LV-P":
            if structure_json['adverbial'] is not None and structure_json["predicative"] is not None:
                if structure_json['adverbial'] in structure_json["predicative"]:
                    structure_json["predicative"] = structure_json["predicative"].replace(
                        structure_json['adverbial'], "")
        # 1.3 特殊疑问句将主语上添加限定词
        definite_words = ["the", "this", "an",
                          "a", "these", "that", "those", "such"]
        indefinite_words = ["some", "any", "all", "both",
                            "each", "every", "many", "much", "few"]

        if structure_json["subject"] is not None:
            for word in indefinite_words:
                if word in structure_json["subject"].lower().split(" "):
                    structure_json["subject"] = structure_json["subject"].lower().replace(
                        word, "")

        if structure_json["type"] == "special":
            if structure_json["subject"] is not None:
                has_definite_word = False
                for word in definite_words:
                    if word in structure_json["subject"].lower().split(" "):
                        has_definite_word = True
                        break
                if not has_definite_word:
                    structure_json["subject"] = "The " + \
                        structure_json["subject"].lower()

        # 1.4 如果系动词是being, 则改为is
        if structure_json["linking_verb"] == "being":
            structure_json["linking_verb"] = "is"

        # 1.5 将所有字符串中的"  "替换为" "
        for key, value in structure_json.items():
            if value is not None:
                structure_json[key] = value.replace("  ", " ")

        print(structure_json)
    except:
        print(f"Error: {structure}")
        structure_json = None

    # 1. get the description of the surgery
    messages = []
    messages.append({
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": f"{file_path}",
                "resized_height": 480,
                "resized_width": 854,
            },
            {"type": "text", "text": description_question[3]},
        ],
    })

    description = infer_by_message(messages, model)
    description_json = json.loads(description)
    description_json["summary_describing"] = "endoscopic or laparoscopic surgery"

    if ("monopolar curved scissors" in description_json["used_tools_and_function"].keys()):
        description_json["used_tools_and_function"]["monopolar curved scissors"] = "tissue cut and dissection, while also providing monopolar electrosurgical energy for cutting and coagulation during minimally invasive procedures."

    # 1.1 filter tools list by detect model
    description_json = filter_tools(file_path, description_json)

    # 2. get the actions of the surgery
    action_list = ["dissection", "cut", "cut tissue", "hemostasis", "suture", "knotting",
                   "resection", "specimen retrieval", "irrigation", "suction", "anastomosis"]
    prompt = f"""
        You are given a description of a laparoscopic (endoscopic) surgery scene.  
    From the following action list: {action_list}, identify all the actions that are explicitly present in the description.  

    Surgery description: "{description}"  

    Output only a Python list of the selected actions.  
    Do not include any actions outside of {action_list}. 
    
    """
    messages = []
    messages.append({
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
        ],
    })

    actions = infer_by_message(messages, model)

    actions_list = [x.strip().strip("'")
                    for x in actions.strip("[]").split(",")]
    if 'suture' in actions_list and 'cut tissue' in actions_list:
        actions_list.remove('cut tissue')
    if 'suture' in actions_list and 'cut' in actions_list:
        actions_list.remove('cut')

    description_json["actions"] = actions_list

    print(description_json)

    # 3. answer the user's question
    prompt = f"""
        You are a precise question answering assistant. You are given a laparoscopic surgery(endoscopic surgery) video description.
    You are only allowed to answer based on the Video description to answer the user's question.Do NOT add any information not contained in the description.
    There should be logic before and after answering.
    Your task is to answer questions strictly following the predefined response style.
    
    Answering Rules:
        1. instrument presence questions(used_tools_and_function contains the instrument in use):
        - If a tool is explicitly listed in used_tools_and_function, answer with: "Yes, [instrument] are being used." / "Yes, a [instrument] was used."
        - If a tool is not listed in used_tools_and_function, answer with: "No [instrument] are being used." / "No [instrument] is being used."

        2. what type of [instrument class] is mentioned?
        - Answer with one short sentence starting with "The type of [instrument class] mentioned is [instrument]"

        3. used_tools_and_function list check questions ("Is a [instrument] among the listed tools?"):
        - If listed: "Yes, a [instrument] is listed."
        - If not listed: "No, a [instrument] is not listed."

        4. actions list check questions ("Is a [action] required in this surgical step?"):
        - Answer with yes/no + requirement: 
            e.g., "Yes, the procedure involves [action]." / "No, [action] are not required."
        
        5. Requirement questions ("Is a [task] required in this surgical step?"):
        - Answer with yes/no + requirement: 
            e.g., "Yes, the procedure involves [action]." / "No, [task] are not required."

        6. Organ/tissue manipulation questions ("What organ is being manipulated?"):
        - Answer with one short sentence starting with: The organ being manipulated is the [organ]."

        7. Procedure identification questions ("What procedure is this summary describing?"):
        - Answer with one short sentence starting with "The summary is describing [refer to summary_describing, task_name and matched_description]."

        8. Purpose/function questions ("What is the purpose of using [tool] in this procedure?"):
        - Answer with one short sentence starting with: "The [tool] are used for [function]."
   

    Video description:
    {description_json}

    User question:
    {input_text}
    Answer concisely, logically, and less than 10 words
    """

    if structure_json is not None and structure_json["linking_verb"] is not None:
        if structure_json["type"] == "normal":
            structure_prompt = f"""
            if Yes: Yes, {structure_json["subject"]} {structure_json["linking_verb"]} {structure_json["predicative"]}
            if No: No, {structure_json["subject"]} {structure_json["linking_verb"]} not {structure_json["predicative"]}
            """
        elif structure_json["type"] == "special":
            if "why" in input_text.lower().split(" "):
                structure_prompt = f"""
                because {structure_json["subject"]} [answer from description]
                """
            else:
                structure_prompt = f"""
                {structure_json["subject"]} {structure_json["predicative"]} {structure_json["linking_verb"]} [answer from description]
                """
            prompt = f"""
                You are a precise question answering assistant. You are given a laparoscopic surgery(endoscopic surgery) video description.
            You are only allowed to answer based on the Video description to answer the user's question.Do NOT add any information not contained in the description.
            There should be logic before and after answering.
            Your task is to answer questions strictly following the predefined response style.
            
            Answering Rules:
                1. instrument presence questions(used_tools_and_function contains the instrument in use):
                - If a tool is explicitly listed in used_tools_and_function, answer with: "Yes, [instrument] are being used." / "Yes, a [instrument] was used."
                - If a tool is not listed in used_tools_and_function, answer with: "No [instrument] are being used." / "No [instrument] is being used."

                2. what type of [instrument class] is mentioned?
                - Answer with one short sentence starting with "The type of [instrument class] mentioned is [instrument]"

                3. used_tools_and_function list check questions ("Is a [instrument] among the listed tools?"):
                - If listed: "Yes, a [instrument] is listed."
                - If not listed: "No, a [instrument] is not listed."

                4. actions list check questions ("Is a [action] required in this surgical step?"):
                - Answer with yes/no + requirement: 
                    e.g., "Yes, the procedure involves [action]." / "No, [action] are not required."
                
                5. Requirement questions ("Is a [task] required in this surgical step?"):
                - Answer with yes/no + requirement: 
                    e.g., "Yes, the procedure involves [action]." / "No, [task] are not required."

                6. Organ/tissue manipulation questions ("What organ is being manipulated?"):
                - Answer with one short sentence starting with: The organ being manipulated is the [organ]."

                7. Procedure identification questions ("What procedure is this summary describing?"):
                - Answer with one short sentence starting with "The summary is describing [procedure]."

                8. Purpose/function questions ("What is the purpose of using [tool] in this procedure?"):
                - Answer with one short sentence starting with: "The [tool] are used for [function]."
        

            Video description:
            {description_json}

            User question:
            {input_text}

            Please organize your answer in the following format:{structure_prompt}
            Answer concisely, logically, and less than 10 words
            """
    messages = []
    messages.append({
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
        ],
    })

    response = infer_by_message(messages, model)

    # 4. simplify the answer
    answer_word_num = len(response.split(" "))
    if structure_json is not None and answer_word_num > 10:

        subject = structure_json["subject"] if structure_json["type"] == "normal" else structure_json["subject"] + \
            " " + structure_json["predicative"]

        prompt_rebuild = f"""You are an assistant that simplifies answers.
            Task:  
            Please keep the subject "{subject}" and linking verb "{structure_json["linking_verb"]}" unchanged. 
            Simplify the part of the sentence after the "{subject}", make sure the entire sentence is no more than 11 words.
            
            Examples:  

            Input: "The purpose of using forceps in this procedure is to grasp tissue safely and securely."  
            Output: "The purpose of using forceps is to grasp tissue safely."

            Input: "Yes, a large needle driver is among the listed tools used here in surgery."  
            Output: "Yes, a large needle driver is among the listed tools."

            Now simplify the following answer, keeping the subject and linking verb unchanged:  
            "{response}"
        """

        messages = []
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_rebuild},
            ],
        })
        response = infer_by_message(messages, model)

    return response


def infer(file_path, input_text):
    description_question = [
        "What is the used_tools_and_function? answer in json format",
        "What is the operated_organ_and_tissue? answer in json format",
        "What is the task_name, task_description and matched_description? answer in json format",
        "Analyze the surgical procedure in the video and provide a structured JSON output with keys: used_tools_and_function, operated_organ_and_tissue, task_name, task_description, matched_description"
    ]

    # 1. get the description of the surgery
    messages = []
    messages.append({
        "role": "user",
        "content": [
            {
                "type": "video",
                    "video": f"{file_path}",
                    "resized_height": 480,
                    "resized_width": 854,
                },
            {"type": "text", "text": description_question[3]},
        ],
    })

    description = infer_by_message(messages, model)
    description_json = json.loads(description)
    if ("monopolar curved scissors" in description_json["used_tools_and_function"].keys()):
            description_json["used_tools_and_function"]["monopolar curved scissors"] = "used for precise tissue cutting and dissection, while also providing monopolar electrosurgical energy for cutting and coagulation during minimally invasive procedures."

    # 2. get the actions of the surgery
    action_list = ["dissection", "cut tissue", "hemostasis", "suture", "knotting",
                "resection", "specimen retrieval", "irrigation", "suction", "anastomosis"]
    prompt = f"""
        You are given a description of a laparoscopic (endoscopic) surgery scene.  
    From the following action list: {action_list}, identify all the actions that are explicitly present in the description.  

    Surgery description: "{description}"  

    Output only a Python list of the selected actions.  
    Do not include any actions outside of {action_list}. 
    
    """
    messages = []
    messages.append({
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
        ],
    })

    actions = infer_by_message(messages, model)
    actions_list = [x.strip().strip("'")
                    for x in actions.strip("[]").split(",")]
    if 'suture' in actions_list and 'cut tissue' in actions_list:
        actions_list.remove('cut tissue')
    description_json["actions"] = actions_list

    # 3. answer the user's question
    prompt = f"""
        You are a precise question answering assistant. You are given a laparoscopic surgery(endoscopic surgery) video description.
    You are only allowed to answer based on the Video description to answer the user's question.Do NOT add any information not contained in the description.
    There should be logic before and after answering.
    Your task is to answer questions strictly following the predefined response style.
    
    Answering Rules:
        1. instrument presence questions(used_tools_and_function contains the instrument in use):
        - If a tool is explicitly listed in used_tools_and_function, answer with: "Yes, [instrument] are being used." / "Yes, a [instrument] was used."
        - If a tool is not listed in used_tools_and_function, answer with: "No [instrument] are being used." / "No [instrument] is being used."

        2. what type of [instrument class] is mentioned?
        - Answer with one short sentence starting with "The type of [instrument class] mentioned is [instrument]"

        3. used_tools_and_function list check questions ("Is a [instrument] among the listed tools?"):
        - If listed: "Yes, a [instrument] is listed."
        - If not listed: "No, a [instrument] is not listed."

        4. actions list check questions ("Is a [action] required in this surgical step?"):
        - Answer with yes/no + requirement: 
            e.g., "Yes, the procedure involves [action]." / "No, [action] are not required."
        
        5. Requirement questions ("Is a [task] required in this surgical step?"):
        - Answer with yes/no + requirement: 
            e.g., "Yes, the procedure involves [action]." / "No, [task] are not required."

        6. Organ/tissue manipulation questions ("What organ is being manipulated?"):
        - Answer with one short sentence starting with: The organ being manipulated is the [organ]."

        7. Procedure identification questions ("What procedure is this summary describing?"):
        - Answer with one short sentence starting with "The summary is describing [procedure]."

        8. Purpose/function questions ("What is the purpose of using [tool] in this procedure?"):
        - Answer with one short sentence starting with: "The [tool] are used for [function]."


    Video description:
    {description_json}

    User question:
    {input_text}
    Answer concisely, logically, and less than 10 words
    """

    messages = []
    messages.append({
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
        ],
    })

    response = infer_by_message(messages, model)

    return response


def run():
    # log_cuda_memory("before load model")
    # Load model first
    load_model()
    # log_cuda_memory("after load model")
    
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
    answer = deepthink_infer_by_file(video_path, input_visual_context_question)
    # answer = infer("/data/lxy/data/case_154_9_0_part5.mp4", "is needle driver used in this surgery?")
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
