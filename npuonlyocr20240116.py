import argparse
import time
import cv2
import numpy as np
import torch
import math
import os
import sys
sys.path.append("./acllite")
import acl
import LaneFinder
import constants as const
from acllite_model import AclLiteModel
from acllite_resource import AclLiteResource


def allFilePath(rootPath, allFIleList):
    """ 遍历文件 """
    fileList = os.listdir(rootPath)
    for temp in fileList:
        if os.path.isfile(os.path.join(rootPath, temp)):
            allFIleList.append(os.path.join(rootPath, temp))
        else:
            allFilePath(os.path.join(rootPath, temp), allFIleList)


def resize_norm_img(img, image_shape=[3, 48, 168]):
    """车牌图像resize、归一化、通道顺序变换"""
    imgC, imgH, imgW = image_shape
    h = img.shape[0]
    w = img.shape[1]
    ratio = w / float(h)
    if math.ceil(imgH * ratio) > imgW:
        resized_w = imgW
    else:
        resized_w = int(math.ceil(imgH * ratio))
    resized_image = cv2.resize(img, (resized_w, imgH))
    resized_image = resized_image.astype('float32')
    if image_shape[0] == 1:
        resized_image = resized_image / 255
        resized_image = resized_image[np.newaxis, :]
    else:
        resized_image = resized_image.transpose((2, 0, 1)) / 255
    resized_image -= 0.5
    resized_image /= 0.5
    padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
    padding_im[:, :, 0:resized_w] = resized_image
    return padding_im


def get_ignored_tokens():
    return [0]


def decode(character, text_index, text_prob=None, is_remove_duplicate=False):
    """
    convert text-index into text-label.
    Args:
        character: 处理后的字典
        text_index: 模型预测的最大索引
        text_prob: 模型预测的每个位置的置信度
        is_remove_duplicate: 去除重复的字符

    Returns: 车牌号+平均置信度（每个字符置信度相加/字符总数）
    """
    result_list = []
    ignored_tokens = get_ignored_tokens()
    batch_size = len(text_index)
    for batch_idx in range(batch_size):
        selection = np.ones(len(text_index[batch_idx]), dtype=bool)
        if is_remove_duplicate:
            selection[1:] = text_index[batch_idx][1:] != text_index[batch_idx][:-1]
        for ignored_token in ignored_tokens:
            selection &= text_index[batch_idx] != ignored_token

        char_list = [
            character[int(text_id)].replace('\n', '')
            for text_id in text_index[batch_idx][selection]
        ]
        if text_prob is not None:
            conf_list = text_prob[batch_idx][selection]
        else:
            conf_list = [1] * len(selection)
        if len(conf_list) == 0:
            conf_list = [0]

        text = ''.join(char_list)
        result_list.append((text, np.mean(conf_list).tolist()))

    return result_list


def npu_ocr_infer(character, lpimg, image_shape=[3, 48, 168]):
    """ 车牌字符识别推理模块 """
    lpimg = resize_norm_img(lpimg, image_shape=image_shape)
    img_data = lpimg[np.newaxis, :, :, :]
    img_data = torch.from_numpy(img_data)
    img_data = np.array(img_data, dtype=np.float32)
    # ocr by .om
    t3 = time.time()
    result_om = modelocr.execute([img_data, ])
    t4 = time.time()
    npu_ocr_time = t4 - t3
    # print(f"NPUOCRInferTimePerLicensePlate:{npu_ocr_time}")

    result_om_arr = np.asarray(result_om[0])
    preds_om_idx = result_om_arr.argmax(axis=2)
    preds_om_prob = result_om_arr.max(axis=2)
    om_result = decode(character, preds_om_idx, preds_om_prob, is_remove_duplicate=True)
    if isinstance(om_result, dict):
        rec_info = dict()
        for key in om_result:
            if len(om_result[key][0]) >= 2:
                rec_info[key] = {
                    "layer": om_result[key][0][0],
                    "score": float(om_result[key][0][1]),
                }
        # print("om_res:", rec_info)
    else:
        if len(om_result[0]) >= 2:
            # info = om_result[0][0] + "\t" + str(om_result[0][1])
            info = om_result[0][0]
            info_conf = om_result[0][1]
            # print("om_infer_res:", info, info_conf)
            return info, info_conf, npu_ocr_time


def om_character_dict_mapping(character_dict_path, use_space_char=True):
    """ 将字典映射成可以使用的格式 : blank+ 字符 + '' ;"""
    character = []
    character.append("blank")
    with open(character_dict_path, "rb") as fin:
        lines = fin.readlines()
        for line in lines:
            line = line.decode('utf-8').strip("\n").strip("\r\n")
            character.append(line)
    if use_space_char:
        character.append(" ")
    return character


if __name__ == "__main__":

    """ 20240116说明：使用.om 进行车牌字符识别 华为昇腾服务器 """

    # https://blog.csdn.net/qq_22764813/article/details/133787584?spm=1001.2101.3001.6650.2&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-2-133787584-blog-115270800.235%5Ev38%5Epc_relevant_sort_base3&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-2-133787584-blog-115270800.235%5Ev38%5Epc_relevant_sort_base3&utm_relevant_index=5
    # 华为:ONNX模型转换成om
    # atc --model=yolov7-lite-s.onnx --framework=5 --input_format=NCHW --input_shape="images:1,3,768,1280"  --soc_version=Ascend310P3 --output=yolov7lpdet --log=error
    # batchsize 动态
    # atc --model=ocr_rec111302.onnx --framework=5 --input_format=NCHW --input_shape="input:-1,3,48,168" --dynamic_batch_size="1,2,4,8"   --soc_version=Ascend310P3 --output=ocr_rec111302 --log=error
    # atc --model=ocr_rec111302.onnx --framework=5 --input_format=NCHW --input_shape="input:1,3,48,168"  --soc_version=Ascend310P3 --output=ocr_rec111302 --log=error

    # 华为 om格式模型推理代码
    parser = argparse.ArgumentParser()
    parser.add_argument('--character_dict_path', type=str, default='./weight/chinese_plate_dict.txt', help='ocr dict path(s)')  # 检测模型
    parser.add_argument('--rec_model', type=str, default='./weight/ocr_rec111302.om', help='model path(s)')  # 识别模型
    parser.add_argument('--image_path', type=str, default='./images/lp5749', help='source')
    parser.add_argument('--img_size', type=int, default=[3,48, 168], help='inference size (pixels)')
    opt = parser.parse_args()

    # 遍历文件（图片）绝对路径
    file_list = []
    allFilePath(opt.image_path, file_list)
    # print(f"file_list:{file_list}")
    # 加载字典
    character = om_character_dict_mapping(opt.character_dict_path, use_space_char=True)
    
    # 识别模型初始化
    acl_resource = AclLiteResource()
    acl_resource.init()
    modelocr = AclLiteModel(opt.rec_model)
    
    # 计算ocr推理时间    
    npu_lp_ocr_total_time = 0  # OCR compute infer time
    npuOcrPrePostTotalTime = 0
    count = 0  # 统计计算的图片数量
    for file in file_list:
        count += 1
        img_onnx = cv2.imread(file)
        t1 = time.time()
        info, character_conf, npu_ocr_time = npu_ocr_infer(character, img_onnx, opt.img_size)
        t2 = time.time()
        npuOcrPrePostTime = t2-t1
        npu_lp_ocr_total_time += npu_ocr_time  # NPU

        npuOcrPrePostTotalTime += npuOcrPrePostTime
        print(f"{file}, {info, character_conf}")

print("--------------------华为昇腾 NPU 车牌字符识别推理评估---------------------")
print(f"NPUOCRAvgTime:{npu_lp_ocr_total_time/count}; license plate number: {count}, TotalTime:{npu_lp_ocr_total_time}")
print(
    f"NPUOCRPrePostAvgTotalTime:{npuOcrPrePostTotalTime/count}; license plate number: {count}, TotalTime:{npuOcrPrePostTotalTime}\n")




