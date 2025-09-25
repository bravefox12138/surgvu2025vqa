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
import string

# INPUT_PATH = Path("/input")
# OUTPUT_PATH = Path("/output")
# MODEL_PATH = Path("/opt/app/model")

INPUT_PATH = Path("./test/input")
OUTPUT_PATH = Path("./test/output")
MODEL_PATH = Path("./model")

RESOURCE_PATH = Path("resources")


def log_cuda_memory(message):
    print(f"========== {message} ==========")
    print("Allocated:", torch.cuda.memory_allocated(
        device="cuda:0")/1024**2, "MB")
    print("Reserved :", torch.cuda.memory_reserved(
        device="cuda:0")/1024**2, "MB")
    print("Max Allocated:", torch.cuda.max_memory_allocated(
        device="cuda:0")/1024**2, "MB")
    os.system("nvidia-smi")


def load_model():
    """Load the model and processor globally"""

    global config
    config = configparser.ConfigParser()
    config.read(os.path.join(MODEL_PATH, "alg.cfg"), encoding='utf-8')

    global det_model, cls_model, cls_transform, device
    det_model_path = os.path.join(MODEL_PATH, "last.pt")
    cls_model_path = os.path.join(MODEL_PATH, "checkpoint-399.pth")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load detect model
    from ultralytics.utils import LOGGER
    LOGGER.setLevel(logging.ERROR)
    from ultralytics import YOLO
    det_model = YOLO(det_model_path)
    # load classify model
    cls_model = create_model("resnet50", pretrained=False, num_classes=12)
    import numpy as np
    # with torch.serialization.safe_globals([np.core.multiarray.scalar]):
    checkpoint = torch.load(
        cls_model_path, map_location="cpu", weights_only=False)['model']

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
        vision_process.FPS_MAX_FRAMES = frames
        vision_process.FPS_MIN_FRAMES = frames
        image_inputs, video_inputs = process_vision_info(
            messages)

    except Exception as e:
        print(f"处理视频文件时出错: {e}")
        return None

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

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

    import cv2
    import torch
    import numpy as np
    from PIL import Image
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_interval = max(fps * 1, 1)
    frame_idx = 0

    tools_set = set()
    prev_tools = set()

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        if frame_idx % frame_interval == 0:
            current_tools = set()

            # detect
            det_res = det_model.predict(frame, conf=0.65, iou=0.3)[0]
            boxes = det_res.boxes.xyxy.cpu().numpy().astype(np.int32)
            clses = det_res.boxes.cls.cpu().numpy().astype(np.int32)

            for i in range(len(boxes)):
                det_cls = clses[i]
                det_label = id_label_dict[det_cls]

                # use classify model
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
                        image_trans = cls_transform(
                            crop_rgb).unsqueeze(0).to(device)
                        with torch.no_grad():
                            res = cls_model(image_trans)
                            output = torch.softmax(
                                res, dim=1).detach().cpu().numpy()
                            cls_id = np.argmax(output, axis=1)[0]
                            det_label = id_label_dict[cls_id]

                current_tools.add(det_label)

            stable_tools = prev_tools & current_tools
            tools_set.update(stable_tools)

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

    return tools_list


def load_commercial2gt(csv_path):

    mapping = {}
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            c_name = row["commercial_toolname"].strip().lower()
            g_name = row["groundtruth_toolname"].strip()
            mapping[c_name] = g_name
    return mapping


def merge_tools(video_description, tools_list, commercial2gt):
    
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

    def normalize_name(name):
        return name.strip().lower()

    detect_tools = set()
    for tool in tools_list:
        tool_norm = normalize_name(tool)
        gt_name = commercial2gt.get(tool_norm, tool)
        detect_tools.add(gt_name)

    description_tools = {}
    description_tools_gt = {}
    description_tools_func = {}
    if "used_tools_and_function" in video_description:
        for idx, (name, func) in enumerate(video_description["used_tools_and_function"].items()):
            description_tools[idx] = (name)
            name_norm = normalize_name(name)
            gt_name = commercial2gt.get(name_norm, name)
            description_tools_gt[idx] = (gt_name)
            description_tools_func[idx] = (func)

    print("Description tools:", list(description_tools.values()))
    print("Detected tools:", tools_list)

    gt_tools = detect_tools.copy()
    description_tools_gt_copy = description_tools_gt.copy()
    for idx, gt_name in description_tools_gt_copy.items():
        if gt_name not in detect_tools:
            description_tools_gt.pop(idx)
            description_tools.pop(idx)
            description_tools_func.pop(idx)
        elif gt_name in gt_tools:
            gt_tools.remove(gt_name)

    merged = {}
    for idx, name in description_tools.items():
        merged[name] = description_tools_func[idx]
    for gt_name in gt_tools:
        if gt_name in gt_tools_function_map:
            merged[gt_name.capitalize()] = gt_tools_function_map[gt_name]
        else:
            merged[gt_name.capitalize()] = " "

    video_description["used_tools_and_function"] = merged
    return video_description


def filter_tools(file_path, description_json):
    use_tools_filter = config.getboolean(
        'ALG', 'use_tools_filter', fallback=True)
    if use_tools_filter:
        tools_list = detect_tool_list(file_path)
        commercial2gt = load_commercial2gt(
            os.path.join(MODEL_PATH, "toolname_mapping_filter.csv"))
        description_json = merge_tools(
            description_json, tools_list, commercial2gt)
    return description_json


def longest_common_word_substring(s1, s2):
    words1 = s1.split()
    words2 = s2.split()
    m = [[0] * (len(words2) + 1) for _ in range(len(words1) + 1)]
    longest, end_index = 0, 0

    for i in range(1, len(words1) + 1):
        for j in range(1, len(words2) + 1):
            if words1[i - 1] == words2[j - 1]:
                m[i][j] = m[i - 1][j - 1] + 1
                if m[i][j] > longest:
                    longest = m[i][j]
                    end_index = i
            else:
                m[i][j] = 0

    return " ".join(words1[end_index - longest:end_index])


def is_forceps_type_question(text: str) -> bool:
    text = text.lower()
    type_synonyms = ["type", "kind", "sort", "class", "category", "form"]
    question_words = ["what", "which"]
    return (
        "forceps" in text
        and any(word in text for word in type_synonyms)
        and any(q in text.split() for q in question_words)
    )


def deepthink_infer_by_file(file_path, input_text):
    print(f"question: {input_text}")
    description_question = [
        "What is the used_tools_and_function? answer in json format",
        "What is the operated_organ_and_tissue? answer in json format",
        "What is the task_name, task_description and matched_description? answer in json format",
        "Analyze the surgical procedure in the video and provide a structured JSON output with keys: used_tools_and_function, operated_organ_and_tissue, task_name, task_description, matched_description"
    ]

    question = input_text.strip(string.punctuation).lower().split(" ")
    definite_words = ["the", "this", "an",
                      "a", "these", "that", "those", "such"]
    definite_words_map = {}
    for word in definite_words:
        if word in question:
            p = [i for i, x in enumerate(question) if x == word]
            for i in p:
                if i + 1 < len(question):
                    definite_words_map[question[i + 1]
                                       ] = word + " " + question[i + 1]
    for word in definite_words + ['there', 'being']:
        if word in question:
            question.remove(word)
    question = " ".join(question)

    special_words = ["what", "which", "who", "whom", "whose", "when", "where", "why", "how", "how many", "how much",
                     "how long", "how often", "how far", "how big", "how small", "how heavy", "how light",
                     "how tall", "how short", "how wide", "how deep", "how thick", "how thin", "how long",
                     "how short", "how wide", "how deep", "how thick", "how thin"]
    if input_text.lower().split(" ")[0] in special_words:
        question_type = "special"
    else:
        question_type = "normal"

    # 3. analyze the structure of the question
    structure_prompt = f"""You are a linguistic analyzer.  
    Your task is to analyze the grammatical structure of the given English sentence.

    Step 1: Determine the sentence structure: 
    - If the sentence uses a linking verb (like is, are, was, were, seem, become) connecting the subject to a predicative, classify it as Subject–Linking Verb–Predicative (S-LV-P).

    Step 2: Extract components according to the structure:

    - subject
    - linking_verb
    - predicative
    - adverbial
    - complement

    Special rule for **WH-questions** (what, which, who, where, etc.):
    - If a WH-word starts the sentence, the noun or noun phrase immediately following it is usually the **subject**.
    - Do not confuse auxiliary verbs (do, does, did) or main verbs with linking verbs in WH-questions.

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
    Input: "Is a needle driver being used in this procedure?"  
    Structure: S-LV-P  
    Output:  
    {{
    "structure": "S-LV-P",
    "subject": "a needle driver",
    "linking_verb": "is",
    "predicative": "being used",
    "adverbial": "in this procedure",
    "complement": null
    }}

    Example 3:
    Input: "Which structure is being cauterized during this surgery?"  
    Structure: S-LV-P  
    Output:  
    {{
    "structure": "S-LV-P",
    "subject": "Which structure",
    "linking_verb": "is",
    "predicative": "being cauterized",
    "adverbial": during this surgery,
    "complement": null
    }}

    Example 4:
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

    Example 5:
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
    "{question}"
    
    """
    messages = []
    messages.append({
        "role": "user",
        "content": [
            {"type": "text", "text": structure_prompt},
        ],
    })
    structure = infer_by_message(messages, model)

    try:
        structure = structure.strip("{}")
        structure = "{" + structure + "}"
        structure_json = json.loads(structure)
        structure_json["type"] = question_type
        print(f"response structure: {structure}")

        for key, value in definite_words_map.items():
            for structure in ["subject", "predicative", "adverbial", "complement"]:
                if structure_json[structure] is None or structure_json[structure] == "":
                    continue
                if key in structure_json[structure]:
                    structure_json[structure] = structure_json[structure].replace(
                        key, value)

        print(f"relocate structure: {structure_json}")

        for key, value in structure_json.items():
            if value is None:
                continue
            for word in special_words:
                if word in value.lower().split(" "):
                    structure_json[key] = value.lower().replace(word, "the")

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

        if structure_json["linking_verb"] == "being":
            structure_json["linking_verb"] = "is being"

        if structure_json['adverbial'] is not None and structure_json["predicative"] is not None:
            common = longest_common_word_substring(
                structure_json["adverbial"], structure_json["predicative"])
            structure_json["predicative"] = structure_json["predicative"].replace(
                common, "").strip()
        if structure_json['subject'] is not None and structure_json["predicative"] is not None:
            common = longest_common_word_substring(
                structure_json["subject"], structure_json["predicative"])
            structure_json["predicative"] = structure_json["predicative"].replace(
                common, "").strip()

        for key, value in structure_json.items():
            if value is not None:
                structure_json[key] = value.replace("  ", " ")
        print(f"final structure: {structure_json}")

    except Exception as e:
        print(f"Error: {e}")
        structure_json = None

    # get the description of the surgery video
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
    description_json["summary_describing"] = "This is a description of endoscopic or laparoscopic surgery."

    description_json = filter_tools(file_path, description_json)
    description_json["used_tools_and_function"] = {k.lower(): v for k,
                                                   v in description_json["used_tools_and_function"].items()}

    for key, value in description_json["used_tools_and_function"].items():
        if "monopolar curved scissors" in key:
            description_json["used_tools_and_function"][key] = "used to cut and coagulate tissues."
        if "cadiere forceps" in key:
            description_json["used_tools_and_function"][key] = "this is one kind of forceps, used to grasp and hold tissues or objects."
        if "bipolar forceps" in key:
            description_json["used_tools_and_function"][key] = "this is one kind of forceps, used to grasp and coagulate tissues."
        if "bipolar grasper" in key or "force bipolar" in key:
            description_json["used_tools_and_function"][key] = "used to grasp and coagulate tissues."
        if "suturecut needle driver" in key:
            description_json["used_tools_and_function"][key] = "used to grasp needles and cut sutures."
        elif "needle driver" in key:
            description_json["used_tools_and_function"][key] = "used to grasp needles."
        if "clip applicator" in key:
            description_json["used_tools_and_function"][key] = "used to apply clips to vessels or ducts for hemostasis or closure."
        if "cautery" in key:
            description_json["used_tools_and_function"][key] = "used to cut and coagulate tissues."
        if "suction irrigator" in key:
            description_json["used_tools_and_function"][key] = "used to aspirate fluids and irrigate tissues"
        if "synchroseal" in key:
            description_json["used_tools_and_function"][key] = "used to seal vessels and tissues while cutting."
        if "stapler" in key:
            description_json["used_tools_and_function"][key] = "used to transect tissues and apply staples for closure."
        if "retractor" in key:
            description_json["used_tools_and_function"][key] = "used to grasp and retract tissues."
        if "prograsp forceps" in key:
            description_json["used_tools_and_function"][key] = "this is one kind of forceps, used to strongly grasp and retract tissues."

    description_json["operated_organ_and_tissue"] = [
        v.lower() for v in description_json["operated_organ_and_tissue"]]
    for i in range(len(description_json["operated_organ_and_tissue"])):
        if "uterine horn" in description_json["operated_organ_and_tissue"][i] and "connective tissue" in description_json["operated_organ_and_tissue"][i]:
            description_json["operated_organ_and_tissue"][i] = "uterine horn"
    description_json["operated_organ_and_tissue"] = list(
        set(description_json["operated_organ_and_tissue"]))

    TOOL_ACTIONS_MAP = {
        "scissors": [
            "cut",
            "cut tissue",
            "dissect tissue",
            "separate tissue planes",
            "expose structures"
        ],
        "needle": [
            "grasp needle",
            "position needle",
            "drive needle",
            "retrieve needle",
            "suture tissue",
            "suture",
            "tie knots",
            "knots",
            "maintain tension"
        ],
        "cautery": [
            "dissect tissue",
            "coagulate tissue (monopolar energy)",
            "separate tissue planes",
            "expose anatomical structures",
            "elevate tissue"
        ],
        "stapler": [
            "staple tissue",
            "transect tissue",
            "seal vessels",
            "create anastomosis",
            "close tissue"
        ],
        "clip": [
            "apply clips",
            "occlude vessels",
            "seal ducts",
            "control bleeding",
            "secure tissue structures"
        ],
        "bipolar": [
            "grasp tissue",
            "dissect tissue",
            "coagulate tissue",
            "control bleeding",
            "maintain hemostasis"
        ],
        "vessel": [
            "seal vessels",
            "transect tissue",
            "coagulate tissue",
            "divide vessels",
            "maintain hemostasis"
        ],
        "forceps": [
            "grasp tissue",
            "hold tissue",
            "retract tissue",
            "stabilize structures",
            "assist suturing"
        ]
    }

    actions_list = []
    consumables_list = ["trocar"]
    for tool in description_json.get("used_tools_and_function", {}).keys():
        tool_lower = tool.lower()
        for key, actions in TOOL_ACTIONS_MAP.items():
            if key in tool_lower:
                actions_list.extend(actions)
        if "clip" in tool_lower:
            consumables_list.append("clip")
        if "needle" in tool_lower:
            consumables_list.extend(["needle", "suture needle", "sutures"])

    description_json["actions"] = list(set(actions_list))
    description_json["consumables"] = list(set(consumables_list))

    if "task_description" in description_json:
        description_json.pop("task_description")
    if "matched_description" in description_json:
        description_json.pop("matched_description")
    print(description_json)

    ###########################################
    import pandas as pd
    df_tool = pd.read_csv(os.path.join(
        MODEL_PATH, "toolname_mapping_filter.csv"))
    mapping = {row["commercial_toolname"].lower(): row["groundtruth_toolname"]
               for _, row in df_tool.iterrows()}

    new_used_tools_and_function = {}
    for tool, func in description_json["used_tools_and_function"].items():
        tool_lower = tool.lower()
        if tool_lower in mapping:
            new_tool = mapping[tool_lower]
        else:
            new_tool = tool
        new_used_tools_and_function[new_tool] = func

    description_json_gtool = description_json.copy()
    description_json_gtool["used_tools_and_function"] = new_used_tools_and_function

    tool_list_gt = list(
        description_json_gtool['used_tools_and_function'].keys())
    tool_list = list(description_json['used_tools_and_function'].keys())

    # 3. answer the user's question
    prompt_normal = f"""You are a precise question answering assistant. You are given a laparoscopic surgery(endoscopic surgery) video description.
    You are only allowed to answer based on the Video description and Tool List to answer the user's question.Do NOT add any information not contained in the description.
    There should be logic before and after answering.
    Your task is to answer questions strictly following the predefined response style.
    
    Answering Rules:
        1. instrument presence questions:
        - If instrument exists, answer with: "Yes, [instrument] are being used." / "Yes, a [instrument] was used."
        - If instrument does not exist, answer with: "No [instrument] are being used." / "No [instrument] is being used."

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

    Video description:
    {description_json}
    
    Tool List:
    {tool_list}

    Now please answer the following question:
    {input_text}
    """
    prompt_forceps = f"""You are a precise question answering assistant. You are given a laparoscopic surgery(endoscopic surgery) video description.
    You are only allowed to answer based on the Video description and Tool List to answer the user's question.Do NOT add any information not contained in the description.
    There should be logic before and after answering.
    Your task is to answer questions strictly following the predefined response style.
    
    Answering Rules:
        1. instrument presence questions:
        - If instrument exists, answer with: "Yes, [instrument] are being used." / "Yes, a [instrument] was used."
        - If instrument does not exist, answer with: "No [instrument] are being used." / "No [instrument] is being used."

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

    Video description:
    {description_json_gtool}
    
    Tool List:
    {tool_list_gt}

    Now please answer the following question:
    {input_text}
    """

    if (is_forceps_type_question(input_text)):
        prompt = prompt_forceps
    else:
        prompt = prompt_normal

    ##########################################

    linking_verbs = ['is', 'are', 'was', 'were', 'has', 'have', 'had',  'will', 'would', 'shall', 'should',
                     'can', 'could', 'may', 'might', 'must']
    answer_p = "[answer from Video description]"
    if structure_json is not None and structure_json["linking_verb"] is not None and structure_json["subject"] is not None and structure_json["predicative"] is not None:
        is_question = False
        for word in special_words + linking_verbs + ["do", "does", "did"]:
            if input_text.lower().startswith(word):
                is_question = True
                break

        if structure_json["type"] == "normal" and is_question:
            if any(w in structure_json["linking_verb"].lower() for w in linking_verbs):
                structure_prompt = f"""
                If the answer is Yes, reply: Yes, {structure_json["subject"]} {structure_json["linking_verb"]} {structure_json["predicative"]}
                If the answer is No, reply: No, {structure_json["subject"]} {structure_json["linking_verb"]} not {structure_json["predicative"]}
                """
            else:
                structure_prompt = f"""
                If the answer is Yes, reply: Yes, {structure_json["subject"]} {answer_p}
                If the answer is No, reply: No, {structure_json["subject"]} {answer_p}
                """
        elif structure_json["type"] == "special" and is_question:
            if structure_json["subject"] == "the" or structure_json["subject"] == "":
                structure_json["subject"] = structure_json["subject"] + \
                    " " + structure_json["predicative"]
                structure_json["predicative"] = ""
            if "why" in input_text.lower().split(" "):
                structure_prompt = f"""
                because {structure_json["subject"]} {answer_p}
                """
            elif any(a in structure_json["linking_verb"].lower() for a in linking_verbs):
                w = structure_json["predicative"].split(" ")[0]
                if w.endswith("ed"):
                    structure_prompt = f"""
                    {structure_json["subject"]} {w} {structure_json["linking_verb"]} {answer_p}
                    """
                elif structure_json["predicative"].startswith("of") or structure_json["predicative"].startswith("being"):
                    structure_prompt = f"""
                    {structure_json["subject"]} {structure_json["predicative"]} {structure_json["linking_verb"]} {answer_p}
                    """
                else:
                    structure_prompt = f"""
                    {structure_json["subject"]} {structure_json["linking_verb"]} {answer_p}
                    """
            else:
                structure_prompt = f"""
                {structure_json["subject"]} {answer_p}
                """
        else:
            structure_prompt = f"""
            {structure_json["subject"]} {answer_p}
            """
        prompt += f"""
        Answer in the following format, and less than 11 words:{structure_prompt}
        Ensure the answer is grammatically correct. 
        """
    else:
        prompt += f"""
        Answer concisely, logically, and less than 11 words
        Ensure the answer is grammatically correct.  
        """

    messages = []
    messages.append({
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
        ],
    })

    response = infer_by_message(messages, model)

    print(f"response answer: {response}")

    # simplify the answer
    answer_word_num = len(response.split(" "))
    if structure_json is not None and answer_word_num > 8:

        subject = structure_json["subject"] if structure_json["type"] == "normal" else structure_json["subject"] + \
            " " + structure_json["predicative"]

        prompt_rebuild = f"""you are an assistant that simplifies sentence.
        Task:

        Keep the subject "{subject}" and the linking verb "{structure_json["linking_verb"]}" unchanged.

        Ensure the entire sentence is no more than 11 words.

        Do not change the sentence structure (remain subject + linking verb + complement).

        Examples:
        Input: "The purpose of using forceps in this procedure is to grasp tissue safely and securely."
        Output: "The purpose of forceps is to grasp tissue."

        Input: "The cadiere forceps are used for grasping and manipulating tissue during endoscopic and laparoscopic surgery."
        Output: "The cadiere forceps are used for grasping and manipulating tissue."

        Input: "Yes, a large needle driver is among the listed tools used here in surgery."
        Output: "Yes, a large needle driver is listed."

        Input: "The type of forceps mentioned is maryland bipolar forceps in this surgery."
        Output: "The type of forceps mentioned is maryland bipolar forceps."

        Input: "Because the cadiere forceps are used during dissection."
        Output: "Because the cadiere forceps are used."

        Input: "The maryland bipolar forceps are used to grasp and coagulate tissues during vessel isolation."
        Output: "The maryland bipolar forceps are used to grasp and coagulate tissues."

        Now simplify the following sentence, keeping the subject and linking verb unchanged:
        "{response}"

        Do not simplify the instrument name: {[key for key in description_json["used_tools_and_function"].keys()]}.

        """

        messages = []
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_rebuild},
            ],
        })
        response = infer_by_message(messages, model)
        print(f"answer simplify: {response}")

    response = response.strip('""')
    response = response.lower()
    response = response.strip('.')
    if '.' in response:
        response = response.split('.')[0]
    if ';' in response:
        response = response.split(';')[0]
    split_response = response.split(", ")
    response_list = split_response[:2] if len(
        split_response) > 2 else split_response
    if len(response_list[-1].split(" ")) < 3:
        response = " and ".join(response_list)
    else:
        response = ", ".join(response_list)
    if "of using" in response:
        response = response.replace("of using", "of")
    if "is rectal artery/vein" in response:
        response = response.replace(
            "is rectal artery/vein", "are rectal artery and rectal vein")
    if len(response.split(" during")[-1].split(" ")) < 4:
        response = " ".join(response.split("during")[:-1])
    if len(response.split(" during")[0].split(" ")) >= 8:
        response = response.split(" during")[0]
    question = input_text.strip(string.punctuation).lower().split(" ")
    definite_words = ["being"]
    definite_words_map = {}
    for word in definite_words:
        if word in question:
            p = [i for i, x in enumerate(question) if x == word]
            for i in p:
                if i + 1 < len(question):
                    definite_words_map[question[i +
                                                1].strip(string.punctuation)] = word
    response_word_list = response.split(" ")
    for key, value in definite_words_map.items():
        if key.strip(string.punctuation) in response_word_list:
            index = response_word_list.index(key)
            if index - 1 >= 0:
                if response_word_list[index - 1] != value:
                    response_word_list.insert(index-1, value)
                    response = response.replace(key, value + " " + key)
    response = response.strip()
    response += '.'
    response = response.capitalize()
    print(f"answer: {response}")

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
    input_endoscopic_robotic_surgery_video = INPUT_PATH / \
        "endoscopic-robotic-surgery-video.mp4"
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
    print('These are the inputs:', inputs)
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
    print(
        f"Torch CUDA is available: {(available := torch.cuda.is_available())}")
    if available:
        print(f"\tnumber of devices: {torch.cuda.device_count()}")
        print(
            f"\tcurrent device: { (current_device := torch.cuda.current_device())}")
        print(
            f"\tproperties: {torch.cuda.get_device_properties(current_device)}")
    print("=+=" * 10)


if __name__ == "__main__":
    raise SystemExit(run())
