import cv2
import numpy
import numpy as np
import torch
import onnxruntime
import math
import os
import time
import argparse


def creat_new_dir(dir_path):
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    else:
        pass


def onnx_infer_type(onnx_use_gpt=True):
    # print(onnxruntime.get_device())
    # print(onnxruntime.get_available_providers())
    if onnx_use_gpt:
        providers = ['CUDAExecutionProvider']
    else:
        providers = ['CPUExecutionProvider']
    print(providers)

    return providers


def license_plate_mapping(OCR_onnx_file, character_dict_path, use_space_char=True, onnx_use_gpt=False):
    """
    功能： 获取加载的OCR模型、输入/输出名称： input/output、加载的字典
    Args:
        onnx_file: 检测模型路径
        character_dict_path: 字符识别字典
        use_space_char: 字符空格是否使用
    Returns:获取加载的OCR模型、输入/输出名称： input/output、加载的字典

    """

    # 基于onnx结果进行车牌字符对应
    providers = onnx_infer_type(onnx_use_gpt=onnx_use_gpt)  # 是否调用GPU, 默认调用
    sess = onnxruntime.InferenceSession(OCR_onnx_file, providers=providers)  # 加载模型
    input_names = [input.name for input in sess.get_inputs()]  # 获取输入节点名称
    output_names = [output.name for output in sess.get_outputs()]  # 获取输出节点名称
    character = []
    character.append("blank")
    with open(character_dict_path, "rb") as fin:
        lines = fin.readlines()
        for line in lines:
            line = line.decode('utf-8').strip("\n").strip("\r\n")
            character.append(line)
    if use_space_char:
        character.append(" ")

    return sess, input_names, output_names, character


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


def process(input_names, image):
    """ 准备模型运行的feed_dict"""
    feed_dict = dict()
    for input_name in input_names:
        feed_dict[input_name] = image

    return feed_dict


def get_ignored_tokens():

    return [0]


def decode(character, text_index, text_prob=None, is_remove_duplicate=False):
    """ convert text-index into text-label. """
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


def ocr_infer(sess, character, output_names, input_names, lp_bbox_data):
    """
        Args:
            sess: onnx加载的字符识别模型
            character: 处理后的字典
            output_names: 模型输入名称
            input_names: 模型输出名称
            lp_bbox_data:cv读取的图片上截取车牌区域图
            boxlp:检测模型预测得到的结果:N*14, N=1

    """
    t3 = time.time()
    img_onnx = resize_norm_img(lp_bbox_data)  # （3 48 168）height, width, channel
    onnx_indata = img_onnx[np.newaxis, :, :, :]
    onnx_indata = torch.from_numpy(onnx_indata)
    onnx_indata = np.array(onnx_indata, dtype=np.float32)
    feed_dict = process(input_names, onnx_indata)

    # ----------- ocr onnx infer------
    t1 = time.time()
    # onnx 模型推理：输入{input:[1,3,48,168]};输出：{}
    output_onnx = sess.run(output_names, feed_dict)  # 最内部的长度为85=blank+字典（83）+' '长度一致
    t2 = time.time()
    ocr_onnx_infer_time = t2 - t1
    # print(f"onnx_ocr_infer_time:{t2-t1}")
    output_onnx = numpy.asarray(output_onnx[0])  # （1，21，85）
    preds_idx = output_onnx.argmax(axis=2)
    preds_prob = output_onnx.max(axis=2)
    # 字符已去重复：车牌号+平均置信度（每个字符置信度相加/字符总数）
    post_result = decode(character, preds_idx, preds_prob, is_remove_duplicate=True)
    # print(f"post_result:{post_result}")
    t4 = time.time()
    preposOCRTime = t4-t3

    if isinstance(post_result, dict):
        rec_info = dict()
        for key in post_result:
            if len(post_result[key][0]) >= 2:
                rec_info[key] = {
                    "label": post_result[key][0][0],
                    "score": float(post_result[key][0][1]),
                }
        # print(image_path, rec_info)
    else:
        if len(post_result[0]) >= 2:
            # info = post_result[0][0] + "\t" + str(post_result[0][1])
            info = post_result[0][0]
            info_conf = post_result[0][1]

            return info, info_conf, ocr_onnx_infer_time, preposOCRTime


if __name__ == '__main__':
    # https://blog.csdn.net/qq_22764813/article/details/133787584?spm=1001.2101.3001.6650.2&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-2-133787584-blog-115270800.235%5Ev38%5Epc_relevant_sort_base3&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-2-133787584-blog-115270800.235%5Ev38%5Epc_relevant_sort_base3&utm_relevant_index=5

    parser = argparse.ArgumentParser()

    # RTX3090台式机
    #parser.add_argument('--character_dict_path', type=str, default='chinese_plate_dict.txt',help='ocr dict path(s)')  # 检测模型
    #parser.add_argument('--rec_model', type=str, default='ocr_rec111302.onnx', help='model.pt path(s)')  # 识别模型
    #parser.add_argument('--image_dir', type=str, default='/home/dell/桌面/huaweiDetOCR/images/lp5479', help='source')
    # parser.add_argument('--image_dir', type=str, default='/home/dell/桌面/huaweiDetOCR/images/lp2', help='source')

    # huawei
    parser.add_argument('--character_dict_path', type=str, default='./weight/chinese_plate_dict.txt',help='ocr dict path(s)')  # 检测模型
    parser.add_argument('--rec_model', type=str, default='./weight/ocr_rec111302.onnx', help='model.pt path(s)')  # 识别模型
    parser.add_argument('--image_dir', type=str, default='./images/lp5749', help='source')
    # parser.add_argument('--image_dir', type=str, default='./images/lp2', help='source')

    # A100:/home/jinbo/items/huaweionnxtest/testhuaweiOnnx
    # parser.add_argument('--character_dict_path', type=str, default='/home/jinbo/items/huaweionnxtest/testhuaweiOnnx/weight/paddleocr1113/chinese_plate_dict.txt',help='ocr dict path(s)')  # 检测模型
    # parser.add_argument('--rec_model', type=str, default='/home/jinbo/items/huaweionnxtest/testhuaweiOnnx/weight/paddleocr1113/ocr_rec111302.onnx', help='model.pt path(s)')  # 识别模型
    # # parser.add_argument('--image_dir', type=str, default='/media/dell/sata4t/jwang/datasets/items_datasets/plate_det/license_plate_eval/color_eval/test/blue', help='source')
    # parser.add_argument('--image_dir', type=str, default='/home/jinbo/items/huaweionnxtest/testhuaweiOnnx/images/lp5749', help='source')

    opt = parser.parse_args()

    # 参数加载
    image_dir = opt.image_dir
    rec_model = opt.rec_model
    character_dict_path = opt.character_dict_path

    # 获取加载的OCR模型、输入 / 输出名称： input / output、加载的字典
    sess, input_names, output_names, character = license_plate_mapping(rec_model, character_dict_path, use_space_char=True, onnx_use_gpt=False)

    use_gpu_infer = False
    # use_gpu_infer = True
    if use_gpu_infer:
        providers = onnx_infer_type(onnx_use_gpt=True)  # 调用GPU, 默认调用
    else:
        providers = onnx_infer_type(onnx_use_gpt=False) # 不调用GPU
    session_rec = onnxruntime.InferenceSession(rec_model, providers=providers)  # 加载ONNX格式的车牌OCR模型
    sess = session_rec

    # 统计加载时间
    onnx_OCRTotalTime = 0
    onnx_prepostTotalTime = 0


    files = os.listdir(image_dir)
    count = len(files)
    for file in files:
        image_path = os.path.join(image_dir, file)
        img_onnx = cv2.imread(image_path)

        ocr_result, ocr_conf, onnx_infer_time, preposOCRTime = ocr_infer(sess, character, output_names, input_names, img_onnx)

        onnx_OCRTotalTime += onnx_infer_time
        onnx_prepostTotalTime += preposOCRTime
    # print("-------------------- RTX3090 GPU 车牌检测/字符识别 ONNX 推理评估---------------------")
    print("-------------------- 华为昇腾 ONNX CPU 车牌字符识别推理评估---------------------")
    print(f"onnx-ocrAveTime:{onnx_OCRTotalTime/count}, count:{count}， totalTime:{onnx_OCRTotalTime}")
    print(f"onnx-prepostocrAveTime:{onnx_prepostTotalTime/count}, count:{count}， totalTime:{onnx_prepostTotalTime}")



