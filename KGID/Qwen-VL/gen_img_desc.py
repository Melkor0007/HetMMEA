from argparse import ArgumentParser
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import os
import pickle
import pandas as pd
from tqdm import tqdm
from PIL import Image
import os.path as osp
import tempfile

# 默认模型检查点路径
DEFAULT_CKPT_PATH = '/root/autodl-tmp/huggingface/hub/models--Qwen--Qwen-VL-Chat/snapshots/548275c8b99de56dec203c0e793be18e030f2f4c'
# 默认图片根目录（如有需要）
DEFAULT_IMAGE_ROOT = '/path/to/default/image/root'
# 默认文件保存路径
DEFAULT_SAVE_PATH = '/path/to/save/directory'
# 默认训练属性文件路径
DEFAULT_TRAINING_ATTRS = 'training_attrs_1'
# 默认CSV文件名称，可以根据需要修改
DEFAULT_CSV_FILENAME = 'entity_descriptions_FR.csv'
# 限制属性数量，防止提问模板过长
MAX_ATTR_COUNT = 3

def _get_args():
    parser = ArgumentParser()
    parser.add_argument("-c", "--checkpoint-path", type=str, default=DEFAULT_CKPT_PATH, help="Checkpoint path")
    parser.add_argument("--cpu-only", action="store_true", help="Use CPU only")
    parser.add_argument("--pkl-file", type=str, required=True, help="Input PKL file containing entity images")
    parser.add_argument("--save-path", type=str, default=DEFAULT_SAVE_PATH, help="Path to save processed descriptions and (optionally) images")
    # 新增参数：训练属性文件（每行记录实体及其属性）
    parser.add_argument("--training-attrs", type=str, default=DEFAULT_TRAINING_ATTRS, help="Path to training attributes file")
    # 新增参数：是否保存图片（默认不保存图片，使用临时文件供模型使用，处理后删除）
    parser.add_argument("--save-images", action="store_true", help="If set, images will be saved to disk. Default is not to save images.")
    return parser.parse_args()

def _load_model_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint_path, trust_remote_code=True, resume_download=True
    )

    device_map = "cpu" if args.cpu_only else "cuda"
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint_path,
        device_map=device_map,
        trust_remote_code=True,
        resume_download=True,
    ).eval()

    model.generation_config = GenerationConfig.from_pretrained(
        args.checkpoint_path, trust_remote_code=True, resume_download=True
    )
    return model, tokenizer

def process_image_and_question(model, tokenizer, image_path, question):
    """
    输入模型、分词器、图片路径和问题，返回回答。
    """
    query = tokenizer.from_list_format([
        {'image': image_path},
        {'text': question}
    ])
    response, history = model.chat(tokenizer, query=query, history=None)
    return response

def load_training_attrs(training_attrs_file):
    """
    从训练属性文件中读取数据，返回一个字典：
       key   : 实体名称（取 URL 最后部分）
       value : 属性列表（每个属性为 URL 最后部分）
    文件每行的字段以制表符分隔，第一项为实体，后续为该实体的属性。
    """
    attrs_dict = {}
    with open(training_attrs_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 2:
                continue
            # 提取实体名称：取 URL 最后一个 '/' 后面的部分
            entity_name = parts[0].split('/')[-1]
            # 对后续每个属性字段同样取最后一部分
            attributes = [p.split('/')[-1] for p in parts[1:] if p]
            attrs_dict[entity_name] = attributes
    return attrs_dict

def process_entities_from_pkl(model, tokenizer, pkl_file, save_path, training_attrs_file, save_images, batch_size=50):
    """
    读取 PKL 文件中的实例图片，生成图片描述并保存描述结果到 CSV 文件。
    根据 training_attrs_file 中提供的属性列表构造提问模板，
    模板示例：
      "请简略描述一下这张图片的整体，并尝试从以下几个角度进行描述：图片的属性1,属性2,属性3等。"
    为防止模板过长，会限制使用的属性数量（最多 MAX_ATTR_COUNT 个）。
    同时将回答中的所有换行符去掉，并支持断点续跑：若 CSV 文件中已有记录，则跳过已处理的实体。
    """
    os.makedirs(save_path, exist_ok=True)
    entities = pickle.load(open(pkl_file, 'rb'))
    
    # 预先加载训练属性文件，生成实体 -> 属性列表的映射
    training_attrs = load_training_attrs(training_attrs_file)
    
    csv_path = osp.join(save_path, DEFAULT_CSV_FILENAME)
    
    # 读取已有的CSV文件以支持断点续跑
    processed_entities = set()
    if osp.exists(csv_path):
        try:
            df_existing = pd.read_csv(csv_path, encoding='utf-8')
            processed_entities = set(df_existing['Entity'].astype(str).tolist())
        except Exception as e:
            print("读取已有CSV文件失败，断点续跑可能失效。", e)
    else:
        # 如果CSV文件不存在，写入表头
        pd.DataFrame(columns=["Entity", "Description"]).to_csv(csv_path, index=False, encoding="utf-8")
    
    # 如果需要保存图片，则创建图片保存目录
    if save_images:
        image_save_path = osp.join(save_path, "images")
        os.makedirs(image_save_path, exist_ok=True)

    results = []  # 用于暂存结果

    for entity, image in tqdm(entities.items(), desc="Processing entities"):
        try:
            # 获取实体名称（仅保留最后一个 '/' 后面的部分）
            entity_name = entity.split("/")[-1]
            if entity_name in processed_entities:
                # 如果该实体已处理，则跳过
                continue

            # 根据是否保存图片决定如何获得图片路径
            if save_images:
                image_filename = f"{entity_name}.png"
                image_path = osp.join(image_save_path, image_filename)
                image.save(image_path)
            else:
                # 使用临时文件保存图片供模型调用，处理后删除
                tmp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                image_path = tmp_file.name
                tmp_file.close()
                image.save(image_path)
            
            # 根据训练属性文件确定提问模板
            if entity_name in training_attrs:
                attrs = training_attrs[entity_name]
                # 限制属性数量，取最多 MAX_ATTR_COUNT 个属性
                limited_attrs = attrs[:MAX_ATTR_COUNT]
                # 拼接属性字符串
                attrs_str = "，".join(limited_attrs)
                # 如果属性数量超过限制，则在末尾追加“等”
                if len(attrs) > MAX_ATTR_COUNT:
                    attrs_str += "等"
                question = f"请简略描述一下这张图片的整体,识别它可能属于什么实体，并尝试从以下几个角度进行描述：图片的{attrs_str}。"
            else:
                question = "请描述这张图片，提供关键词列表"

            # 调用模型生成描述
            description = process_image_and_question(model, tokenizer, image_path, question)
            # 去掉回答中所有的换行符
            description = description.replace('\n', ' ').replace('\r', ' ')

            results.append({"Entity": entity_name, "Description": description})

            # 如果使用临时文件，则处理完后删除文件
            if not save_images:
                try:
                    os.remove(image_path)
                except Exception as e:
                    print(f"删除临时文件 {image_path} 失败：", e)

            # 批量写入 CSV 文件
            if len(results) >= batch_size:
                pd.DataFrame(results).to_csv(csv_path, mode="a", header=False, index=False, encoding="utf-8")
                results.clear()

        except Exception as e:
            print(f"Error processing entity {entity}: {e}")

    # 写入剩余结果
    if results:
        pd.DataFrame(results).to_csv(csv_path, mode="a", header=False, index=False, encoding="utf-8")

def main():
    args = _get_args()
    model, tokenizer = _load_model_tokenizer(args)

    # 处理 PKL 文件中的实体图片，同时传入训练属性文件路径和是否保存图片的选项
    process_entities_from_pkl(model, tokenizer, args.pkl_file, args.save_path, args.training_attrs, args.save_images)

if __name__ == '__main__':
    main()
