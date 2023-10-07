# face3w/
#drwxrwxr-x 2 dell dell 1376256 Sep 10 23:58 conditioning_images
#drwxrwxr-x 2 dell dell      36 Sep 10 22:33 conv_images
#-rw-rw-r-- 1 dell dell    3964 Oct  7 10:53 face1set.py
#-rw-rw-r-- 1 dell dell    3888 Sep 26 14:28 face3w.py
#drwxrwxr-x 2 dell dell 7438336 Sep  6 04:13 images
#drwxrwxr-x 2 dell dell       6 Sep 10 21:01 test
#-rw-rw-r-- 1 dell dell     895 Sep 11 20:12 test.jsonl
#-rw-rw-r-- 1 dell dell 6427866 Sep 11 23:22 train.jsonl
#-rw-rw-r-- 1 dell dell     316 Sep 26 14:34 train1set.jsonl
#-rw-rw-r-- 1 dell dell   10399 Sep 11 11:05 trainlit.jsonl
#  train1set.jsonl         ---------------
# {"text":"a face of a woman in a pink jacket with a yellow handle","image":"images\/123343_001.jpg","conditioning_image":"conditioning_images\/123343_001.jpg"}
# {"text":"a face of a man with a white jacket and a white jacket","image":"images\/123344_001.jpg","conditioning_image":"conditioning_images\/123344_001.jpg"}


import pandas as pd
from huggingface_hub import hf_hub_url
import datasets
import os

# 构造只有一张图的数据集来训练 lora  controlnet Reference
_VERSION = datasets.Version("0.0.2")

_DESCRIPTION = "facess-s-s-s--s-s- vslidate if adapter traning process is valid"
_HOMEPAGE = "TODO"
_LICENSE = "TODO"
_CITATION = "TODO"

_FEATURES = datasets.Features(
    {
        "image": datasets.Image(),
        "conditioning_image": datasets.Image(),
        "text": datasets.Value("string"),
    },
)

METADATA_URL = hf_hub_url(
    "zdx/face3w",
    filename="train.jsonl",
    repo_type="dataset",
)

IMAGES_URL = hf_hub_url(
    "zdx/face3w",
    filename="images.zip",
    repo_type="dataset",
)

CONDITIONING_IMAGES_URL = hf_hub_url(
    "zdx/face3w",
    filename="conditioning_images.zip",
    repo_type="dataset",
)

_DEFAULT_CONFIG = datasets.BuilderConfig(name="default", version=_VERSION)


class face3w(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [_DEFAULT_CONFIG]
    DEFAULT_CONFIG_NAME = "default"

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=_FEATURES,
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        # metadata_path = dl_manager.download(METADATA_URL)
        # metadata_path = "/home/dell/workspace/T2IAdapter-SDXL-Diffusers/face3w/trainlit.jsonl"
        metadata_path = "/home/dell/workspace/T2IAdapter-SDXL/face3w/train1set.jsonl"
        test_metadata_path = "/home/dell/workspace/T2IAdapter-SDXL/face3w/test.jsonl"
        # {"text": "pale golden rod circle with old lace background",
        #  "image": "images/0.png", "conditioning_image": "conditioning_images/0.png"}
        # images_dir = dl_manager.download_and_extract(IMAGES_URL)
        images_dir = "/home/dell/workspace/T2IAdapter-SDXL/face3w/"
        # conditioning_images_dir = dl_manager.download_and_extract(
        #     CONDITIONING_IMAGES_URL
        # )
        conditioning_images_dir = "/home/dell/workspace/T2IAdapter-SDXL/face3w/"

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "metadata_path": test_metadata_path,
                    "images_dir": images_dir,
                    "conditioning_images_dir": conditioning_images_dir,
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "metadata_path": metadata_path,
                    "images_dir": images_dir,
                    "conditioning_images_dir": conditioning_images_dir,
                },
            ),
        ]

    def _generate_examples(self, metadata_path, images_dir, conditioning_images_dir):
        metadata = pd.read_json(metadata_path, lines=True)

        for _, row in metadata.iterrows():
            text = row["text"]

            image_path = row["image"]
            image_path = os.path.join(images_dir, image_path)
            image = open(image_path, "rb").read()

            conditioning_image_path = row["conditioning_image"]
            conditioning_image_path = os.path.join(
                conditioning_images_dir, row["conditioning_image"]
            )
            conditioning_image = open(conditioning_image_path, "rb").read()

            yield row["image"], {
                "text": text,
                "image": {
                    "path": image_path,
                    "bytes": image,
                },
                "conditioning_image": {
                    "path": conditioning_image_path,
                    "bytes": conditioning_image,
                },
            }
