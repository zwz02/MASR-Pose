# --- 新建文件: preprocess_instance_index.py ---
import os
import _pickle as cPickle
import tqdm
import collections

# 配置你的数据路径
DATA_DIR = './data'  # 指向你的NOCS数据根目录
# 分别处理 CAMERA 和 REAL 数据集
DATASETS = {
    'camera': {'list_path': 'camera/train_list.txt', 'prefix': 'camera'},
    'real': {'list_path': 'real/train_list.txt', 'prefix': 'real'}
}
OUTPUT_PATH = os.path.join(DATA_DIR, 'instance_occurrence_index.pkl')


def build_index():
    # 结构: { 'model_name_xxx': [ ('camera/train/00000/0000', 0), ('real/train/scene_1/0000', 2), ... ] }
    # 列表中的元组表示 (图像路径前缀, 该图像中该物体的索引)
    model_to_occurrences = collections.defaultdict(list)
    total_instances = 0

    for source, config in DATASETS.items():
        list_path = os.path.join(DATA_DIR, config['list_path'])
        if not os.path.exists(list_path):
            print(f"Warning: {list_path} not found, skipping.")
            continue

        with open(list_path, 'r') as f:
            img_list = [line.strip() for line in f.readlines()]

        print(f"Processing {source} dataset ({len(img_list)} images)...")

        for img_path_prefix in tqdm.tqdm(img_list):
            # 构造完整的 label 文件路径
            full_img_path_prefix = os.path.join(DATA_DIR, config['prefix'], img_path_prefix)
            label_path = full_img_path_prefix + '_label.pkl'

            if not os.path.exists(label_path):
                continue

            with open(label_path, 'rb') as f:
                gts = cPickle.load(f)

            # 遍历该图像中的所有物体
            # model_list 存储了每个物体的唯一 CAD 模型名称
            # 遍历该图像中的所有物体
            for obj_idx, model_name in enumerate(gts['model_list']):
                relative_path = os.path.join(config['prefix'], img_path_prefix)
                # --- vvvvv 新增：记录来源 'syn' 或 'real' vvvvv ---
                source_tag = 'syn' if source == 'camera' else 'real'
                model_to_occurrences[model_name].append((relative_path, obj_idx, source_tag))
                # --- ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ ---
                total_instances += 1

    print(f"Index built. Found {len(model_to_occurrences)} unique models and {total_instances} total occurrences.")

    # 过滤掉只出现过一次的模型（无法构建正样本对）
    valid_models = 0
    final_index = {}
    for model_name, occs in model_to_occurrences.items():
        if len(occs) > 1:
            final_index[model_name] = occs
            valid_models += 1

    print(f"Filtered models with only 1 view. Remaining {valid_models} models valid for contrastive learning.")

    with open(OUTPUT_PATH, 'wb') as f:
        cPickle.dump(final_index, f)
    print(f"Index saved to {OUTPUT_PATH}")


if __name__ == '__main__':
    build_index()