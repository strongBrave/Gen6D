import os
import subprocess
import os.path as osp
from turtle import distance
from loguru import logger
import glob
import pickle

img_dir = "data/OCCLUDED_LINEMOD/RGB-D/rgb_noseg"
model_dir = "data/OCCLUDED_LINEMOD/models"
split_dir = "data/OCCLUDED_LINEMOD/anns"
lm_o_dir = "data/OCCLUDED_LINEMOD"
lm_dir = "data/LINEMOD"
model_names = ["ape", "can", "cat", "driller", "duck", "eggbox", "glue", "holepuncher"]

def modify_img_name(img_dir):
    for img_name in os.listdir(img_dir):
        if img_name.startswith("color_") and img_name.endswith(".png"):

            num_part = img_name.split("_")[1].split(".")[0]

            new_num_part = f"{int(num_part):06d}"

            new_img_name = f"{new_num_part}.jpg"

            old_img_path = osp.join(img_dir, img_name)
            new_img_path = osp.join(img_dir, new_img_name)

            os.rename(old_img_path, new_img_path)
        
    logger.info(f"Comple rename")


def modify_model_name(model_dir):
    for model_name in os.listdir(model_dir):
        sub_model_dir = osp.join(model_dir, model_name)

        model_paths = glob.glob(osp.join(sub_model_dir, "*.xyz"))
        if model_paths:
            model_path = model_paths[0]

            new_model_path = osp.join(sub_model_dir, f"{model_name}.xyz")

            os.rename(model_path, new_model_path)
    logger.info("Complete modify model name")

def split_pkl_to_txt(split_dir):
    for model_name in os.listdir(split_dir):
        sub_split_dir = osp.join(split_dir, model_name)

        split_paths = glob.glob(osp.join(sub_split_dir, "*.pkl"))
        if split_paths:
            for split_path in split_paths:
                splits = []
                split_type = osp.basename(split_path).split(".")[0]
                with open(split_path, "rb")  as f:
                    annos = pickle.load(f)
                for anno in annos:
                    img_path = anno[0]
                    img_path = img_path.replace("occ_linemod", "OCCLUDED_LINEMOD")

                    old_img_name = os.path.basename(img_path)
                    num_part = old_img_name.split("_")[1].split(".")[0]

                    new_num_part = f"{int(num_part):06d}"

                    new_img_name = f"{new_num_part}.jpg"

                    img_path = img_path.replace(old_img_name, new_img_name)

                    splits.append((int(num_part), img_path))
                
                splits.sort(key=lambda x: x[0])

                txt_path = osp.join(sub_split_dir, f"{split_type}.txt")
                with open(txt_path, "w") as txt_file:
                    for _, line in splits:
                        txt_file.write(line + "\n")
                logger.info(f"Save {split_path} to {txt_path}")

def create_txt_file(lm_o_dir, lm_dir, txt_name="distance"):
    """
    Params:
        txt_name (str): "distance" or "corners"
    """
    assert txt_name in ["distance", "corners"]
    txt_dir = osp.join(lm_o_dir, txt_name)
    os.makedirs(txt_dir)

    for model_name in model_names:
        sub_txt_dir = osp.join(txt_dir, model_name)
        os.makedirs(sub_txt_dir)
        
        lm_distance_path = osp.join(lm_dir, model_name, f"{txt_name}.txt")

        command = f"cp {lm_distance_path} {sub_txt_dir}"
        os.system(command)
        logger.info(f"Successfully complete command: {command}")

if __name__ == "__main__":
    # modify_img_name(img_dir)
    # modify_model_name(model_dir)
    # split_pkl_to_txt(split_dir)
    create_txt_file(lm_o_dir=lm_o_dir, lm_dir=lm_dir, txt_name="corners")