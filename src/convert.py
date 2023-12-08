# https://www.kaggle.com/datasets/victorolufemi/mosquitoes-detection

import csv
import os
import shutil
from collections import defaultdict
from urllib.parse import unquote, urlparse

import supervisely as sly
from dotenv import load_dotenv
from supervisely.io.fs import get_file_name, get_file_name_with_ext, get_file_size
from tqdm import tqdm

import src.settings as s
from dataset_tools.convert import unpack_if_archive


def count_files(path, extension):
    count = 0
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(extension):
                count += 1
    return count


def convert_and_upload_supervisely_project(
    api: sly.Api, workspace_id: int, project_name: str
) -> sly.ProjectInfo:
    # project_name = "Mosquitoes detection"
    train_data_path = "/home/grokhi/rawdata/mosquito-alert-2023/final"
    # test_data_path = "/home/alex/DATASETS/TODO/Mosquitoes detection/archive/test_images_phase1"
    train_anns_path = "/home/grokhi/rawdata/mosquito-alert-2023/phase2_train_v0.csv"
    batch_size = 30

    ds_name_to_data = {
        "train": (train_data_path, train_anns_path),
        # "test": (test_data_path, None),  # test bboxes are not correct
    }

    def create_ann(image_path):
        labels = []

        image_np = sly.imaging.image.read(image_path)[:, :, 0]  # shapes in csv are not correct
        img_height = image_np.shape[0]
        img_wight = image_np.shape[1]

        image_name = get_file_name_with_ext(image_path)

        ann_data = name_to_data[image_name]
        for curr_ann_data in ann_data:
            obj_class = meta.get_obj_class(curr_ann_data[0])
            coords = curr_ann_data[1]
            left = int(coords[0])
            top = int(coords[1])
            right = int(coords[2])
            bottom = int(coords[3])
            rect = sly.Rectangle(left=left, top=top, right=right, bottom=bottom)
            label = sly.Label(rect, obj_class)
            labels.append(label)

        return sly.Annotation(img_size=(img_height, img_wight), labels=labels)

    japonicus = sly.ObjClass("japonicus/koreicus", sly.Rectangle)
    anopheles = sly.ObjClass("anopheles", sly.Rectangle)
    culex = sly.ObjClass("culex", sly.Rectangle)
    albopictus = sly.ObjClass("albopictus", sly.Rectangle)
    culiseta = sly.ObjClass("culiseta", sly.Rectangle)
    aegypti = sly.ObjClass("aegypti", sly.Rectangle)

    project = api.project.create(workspace_id, project_name, change_name_if_conflict=True)
    meta = sly.ProjectMeta(obj_classes=[japonicus, anopheles, culex, albopictus, culiseta, aegypti])
    api.project.update_meta(project.id, meta.to_json())

    for ds_name, ds_data in ds_name_to_data.items():
        dataset = api.dataset.create(project.id, ds_name, change_name_if_conflict=True)

        images_path, ann_path = ds_data

        name_to_data = defaultdict(list)

        if ann_path is not None:
            with open(ann_path, "r") as file:
                csvreader = csv.reader(file)
                for idx, row in enumerate(csvreader):
                    if idx == 0:
                        continue
                    name_to_data[row[0]].append([row[-1], list(map(float, row[3:-1]))])

        images_names = os.listdir(images_path)

        progress = sly.Progress("Create dataset {}".format(ds_name), len(images_names))

        for images_names_batch in sly.batched(images_names, batch_size=batch_size):
            img_pathes_batch = [
                os.path.join(images_path, image_name) for image_name in images_names_batch
            ]

            img_infos = api.image.upload_paths(dataset.id, images_names_batch, img_pathes_batch)
            img_ids = [im_info.id for im_info in img_infos]

            if ann_path is not None:
                anns = [create_ann(image_path) for image_path in img_pathes_batch]
                api.annotation.upload_anns(img_ids, anns)

            progress.iters_done_report(len(images_names_batch))

    return project
