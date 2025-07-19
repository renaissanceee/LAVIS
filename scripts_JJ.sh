python evaluate.py --cfg-path lavis/projects/blip2/eval/gqa_zeroshot_flant5xl_eval.yaml

# ours!
python evaluate.py --cfg-path lavis/projects/blip2/eval/syn_coco_zeroshot_flant5xl_eval.yaml
# makedirs
./lavis/configs/datasets/syn_coco/
- defaults.yaml

# need to edit path, prompt
./lavis/projects/blip2/eval/: syn_coco_zeroshot_flant5xl_eval.yaml

# output
./lavis/output/BLIP2/SYNVQA/:
20250704130/ # _15
20250708065/ # _1000

# new dataset, register here
# ./lavis/datasets/builders/: vqa_builder.py
@registry.register_builder("syn_coco_vqa")
class OKVQABuilder(COCOVQABuilder):
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/syn_coco/defaults.yaml",
    }
or
@registry.register_builder("syn_coco_vqa")
class SYNCOCOVQABuilder(BaseDatasetBuilder):
    train_dataset_cls = SYNCOCODataset
    eval_dataset_cls = SYNCOCOEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/syn_coco/defaults.yaml",
    }

# need to import
# ./lavis/datasets/builders/: __init__.py
import SYNCOCOVQABuilder

# write corresponding loader
# ./lavis/datasets/datasets/syn_coco_datasets.py