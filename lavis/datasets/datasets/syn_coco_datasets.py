    # {
    #     "id": 37,
    #     "image_id": 203564,
    #     "original_caption": "A bicycle replica with a clock as the front wheel.",
    #     "conflicting_caption": "A bicycle replica with a bell as the front wheel.",
    #     "question": "What circular object serves as the front wheel of the bicycle replica?",
    #     "change_type": "object"
    # },

import os
import json
import random

from PIL import Image

from lavis.datasets.datasets.vqa_datasets import VQADataset, VQAEvalDataset

from collections import OrderedDict


class __DisplMixin:
    def displ_item(self, index):
        sample, ann = self.__getitem__(index), self.annotation[index]

        return OrderedDict(
            {
                "file": ann["image"],
                "question": ann["question"],
                "question_id": ann["question_id"],
                "answers": "; ".join(ann["answer"]),
                "image": sample["image"],
            }
        )


class SYNCOCODataset(VQADataset, __DisplMixin): # non used?
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

    def __getitem__(self, index):
        ann = self.annotation[index]

        image_path = os.path.join(self.vis_root, ann["image"])
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)
        question = self.text_processor(ann["question"])

        answers = [ann["answer"]]
        weights = [1]

        return {
            "image": image,
            "text_input": question,
            "answers": answers,
            "weights": weights,
        }


class SYNCOCOEvalDataset(VQAEvalDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. gqa/images/)
        ann_root (string): directory to store the annotation file
        """

        self.vis_root = vis_root

        self.annotation = json.load(open(ann_paths[0]))

        answer_list_path = ann_paths[1] if len(ann_paths) > 1 else ''
        if os.path.exists(answer_list_path):
            self.answer_list = json.load(open(answer_list_path))
        else:
            self.answer_list = None

        self.vis_processor = vis_processor
        self.text_processor = text_processor

        self._add_instance_ids()

    def __getitem__(self, index):
        ann = self.annotation[index]

        image_path = os.path.join(self.vis_root, f'{str(ann["image_id"]).zfill(12)}.jpg')
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)
        text_input = ann["conflicting_caption"]+ ' '+ ann["question"]
        question = self.text_processor(text_input)

        if "answer" in ann:
            answer = ann["answer"]
        else:
            answer = None

        return {
            "image": image,
            "text_input": question,
            "answer": answer,
            "question_id": ann["id"],
            "change_type": ann["change_type"],
        }
