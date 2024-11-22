import os
from loguru import logger

# 添加日志文件，记录所有日志
logger.add("run.log", format="{time} {level} {message}", level="INFO")

lm_models = ["ape", 
            "benchvise",
            "cam",
            "can", 
            "cat", 
            "driller", 
            "duck", 
            "eggbox", 
            "glue",
            "holepuncher",
            "iron",
            "lamp",
            "phone"]

for id, model in enumerate(lm_models):
    lm_command = f"python eval.py --cfg configs/gen6d_pretrain.yaml --object_name linemod/{model}"
    # if id < 10:
    #     lm_o_command = f"python eval.py --cfg configs/gen6d_pretrain.yaml --object_name occluded_linemod/{model}"
    #     os.system(lm_o_command)
    os.system(lm_command)

    logger.info(f"Complete {model} eval")

