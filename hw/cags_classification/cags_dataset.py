import os
import sys
import urllib.request

import tensorflow as tf

class CAGS:
    H, W, C = 224, 224, 3
    LABELS = [
        # Cats
        "Abyssinian", "Bengal", "Bombay", "British_Shorthair", "Egyptian_Mau",
        "Maine_Coon", "Russian_Blue", "Siamese", "Sphynx",
        # Dogs
        "american_bulldog", "american_pit_bull_terrier", "basset_hound",
        "beagle", "boxer", "chihuahua", "english_cocker_spaniel",
        "english_setter", "german_shorthaired", "great_pyrenees", "havanese",
        "japanese_chin", "keeshond", "leonberger", "miniature_pinscher",
        "newfoundland", "pomeranian", "pug", "saint_bernard", "samoyed",
        "scottish_terrier", "shiba_inu", "staffordshire_bull_terrier",
        "wheaten_terrier", "yorkshire_terrier",
    ]

    _URL = "https://ufal.mff.cuni.cz/~straka/courses/npfl114/1920/datasets/"

    TRAIN_EXAMPLES = 2142
    DEV_EXAMPLES = 306
    TEST_EXAMPLES = 612

    @staticmethod
    def parse(example):
        example = tf.io.parse_single_example(example, {
            "input_1": tf.io.FixedLenFeature([], tf.string),
            "mask": tf.io.FixedLenFeature([], tf.string),
            "label": tf.io.FixedLenFeature([], tf.int64)})
        example["input_1"] = tf.image.convert_image_dtype(tf.image.decode_jpeg(example["input_1"], channels=3), tf.float32)
        example["mask"] = tf.image.convert_image_dtype(tf.image.decode_png(example["mask"], channels=1), tf.float32)
        return example

    def __init__(self):
        if (not os.path.isdir("data")):
            os.mkdir("data")
        
        for dataset, size in [("train", 57463494), ("dev", 8138328), ("test", None)]:
            path = "cags.{}.tfrecord".format(dataset)
            local = os.path.join("data", path)
            if not os.path.exists(local) or (size is not None and os.path.getsize(local) != size):
                print("Downloading file {}...".format(path), file=sys.stderr)
                urllib.request.urlretrieve("{}/{}".format(self._URL, path), filename=local)

            setattr(self, dataset, tf.data.TFRecordDataset(local))
