import copy
import argparse
from PIL import Image, ImageDraw, ImageFont
import time
import cv2
import numpy as np
import torch
import math
import os
import sys
sys.path.append("./acllite")
#sys.path.append("../")
import acl
import LaneFinder
import constants as const
from acllite_model import AclLiteModel
from acllite_resource import AclLiteResource

mean_value, std_value = ((0.588, 0.193))  # 识别模型均值标准差


def allFilePath(rootPath, allFIleList):
    """ 遍历文件 """
    fileList = os.listdir(rootPath)
    for temp in fileList:
        if os.path.isfile(os.path.join(rootPath, temp)):
            allFIleList.append(os.path.join(rootPath, temp))
        else:
            allFilePath(os.path.join(rootPath, temp), allFIleList)


def detect_pre_precessing(img, img_size):  
    """ 检测前处理: 1.图片缩放2.通道变换 3.归一化  """

    img, r, left, top = my_letter_box(img, img_size)  # 图片缩放与填充
    img = img[:, :, ::-1].transpose(2, 0, 1).copy().astype(np.float32)
    img = img / 255
    img = img.reshape(1, *img.shape)  # 增加通道维度（C（rgb）,H,W）-->（batch=1, C（rgb）,H,W）
    return img, r, left, top


def my_letter_box(img, size=(640, 640)):
    """ 图片缩放与填充 """
    h, w, c = img.shape
    r = min(size[0] / h, size[1] / w)
    new_h, new_w = int(h * r), int(w * r)
    top = int((size[0] - new_h) / 2)
    left = int((size[1] - new_w) / 2)

    bottom = size[0] - new_h - top
    right = size[1] - new_w - left
    img_resize = cv2.resize(img, (new_w, new_h))
    img = cv2.copyMakeBorder(img_resize, top, bottom, left, right, borderType=cv2.BORDER_CONSTANT,
                             value=(114, 114, 114))
    return img, r, left, top


def rec_pre_precessing(img, size=(48, 168)): 
    """识别前处理"""
    img = cv2.resize(img, (168, 48))
    img = img.astype(np.float32)
    img = (img / 255 - mean_value) / std_value  # 归一化 减均值 除标准差
    img = img.transpose(2, 0, 1)  # h,w,c 转为 c,h,w
    img = img.reshape(1, *img.shape)  # channel,height,width转为batch,channel,height,channel
    return img


def four_point_transform(image, pts):
    """ 透视变换得到矫正后的图像，方便识别"""
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped


def draw_result(orgimg, dict_list):
    """ 检测识别结果绘制到图片上进行保存 """
    result_str = ""
    for result in dict_list:
        rect_area = result['bbox_x1y1x2y2']

        x, y, w, h = rect_area[0], rect_area[1], rect_area[2] - rect_area[0], rect_area[3] - rect_area[1]
        padding_w = 0.05 * w
        padding_h = 0.11 * h
        rect_area[0] = max(0, int(x - padding_w))
        rect_area[1] = min(orgimg.shape[1], int(y - padding_h))
        rect_area[2] = max(0, int(rect_area[2] + padding_w))
        rect_area[3] = min(orgimg.shape[0], int(rect_area[3] + padding_h))

        height_area = result['roi_height']
        landmarks = result['landmarks']
        result = result['plate_number']
        result_str += result + " "
        for i in range(4):  # 关键点
            cv2.circle(orgimg, (int(landmarks[i][0]), int(landmarks[i][1])), 5, clors[i], -1)
        cv2.rectangle(orgimg, (rect_area[0], rect_area[1]), (rect_area[2], rect_area[3]), (0, 0, 255), 2)  # 画框
        if len(result) >= 1:
            orgimg = cv2ImgAddText(orgimg, result, rect_area[0] - height_area, rect_area[1] - height_area - 10,
                                   (255, 0, 0), height_area)
    return orgimg


def cv2ImgAddText(img, text, left, top, textColor=(0, 255, 0), textSize=20):  
    """ 将识别结果画在图上 """
    if (isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    fontText = ImageFont.truetype("./data/platech.ttf", textSize, encoding="utf-8")

    draw.text((left, top), text, textColor, font=fontText)
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def post_precessing(dets, r, left, top, conf_thresh=0.3, iou_thresh=0.45, num_cls=2):
    """ Det检测后处理, 检测重叠框过滤 ,xywh2xyxyy并还原到原图尺寸"""

    choice = dets[:, :, 4] > conf_thresh
    dets = dets[choice]
    dets[:, 5:5 + num_cls] *= dets[:, 4:5]  # 5::7是2分类类别分数
    box = dets[:, :4]
    boxes = xywh2xyxy(box)
    score = np.max(dets[:, 5:5 + num_cls], axis=-1, keepdims=True)
    index = np.argmax(dets[:, 5:5 + num_cls], axis=-1).reshape(-1, 1)
    kpt_b = 5 + num_cls
    # yolov7关键有三个数，x,y,score，这里我们只需要x,y
    landmarks = dets[:, [kpt_b, kpt_b + 1, kpt_b + 3, kpt_b + 4, kpt_b + 6, kpt_b + 7, kpt_b + 9, kpt_b + 10]]
    output = np.concatenate((boxes, score, landmarks, index), axis=1)
    reserve_ = my_nms(output, iou_thresh)
    output = output[reserve_]
    output = restore_box(output, r, left, top)  # 返回原图上面的坐标
    return output


def get_split_merge(img):
    """双层车牌进行分割后识别"""
    h, w, c = img.shape
    img_upper = img[0:int(5 / 12 * h), :]
    img_lower = img[int(1 / 3 * h):, :]
    img_upper = cv2.resize(img_upper, (img_lower.shape[1], img_lower.shape[0]))
    new_img = np.hstack((img_upper, img_lower))
    return new_img


def order_points(pts):
    """ 关键点排列 按照（左上，右上，右下，左下）的顺序排列 """
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def xywh2xyxy(boxes): 
    """ xywh坐标变为 左上 ，右下坐标 x1,y1  x2,y2 """
    xywh = copy.deepcopy(boxes)
    xywh[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    xywh[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    xywh[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    xywh[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    return xywh


def my_nms(boxes, iou_thresh):
    # NMS
    index = np.argsort(boxes[:, 4])[::-1]
    keep = []
    while index.size > 0:
        i = index[0]
        keep.append(i)
        x1 = np.maximum(boxes[i, 0], boxes[index[1:], 0])
        y1 = np.maximum(boxes[i, 1], boxes[index[1:], 1])
        x2 = np.minimum(boxes[i, 2], boxes[index[1:], 2])
        y2 = np.minimum(boxes[i, 3], boxes[index[1:], 3])

        w = np.maximum(0, x2 - x1)
        h = np.maximum(0, y2 - y1)

        inter_area = w * h
        union_area = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1]) + (
                boxes[index[1:], 2] - boxes[index[1:], 0]) * (boxes[index[1:], 3] - boxes[index[1:], 1])
        iou = inter_area / (union_area - inter_area)
        idx = np.where(iou <= iou_thresh)[0]
        index = index[idx + 1]
    return keep


def restore_box(boxes, r, left, top):
    """ 返回原图上面的坐标 """
    boxes[:, [0, 2, 5, 7, 9, 11]] -= left
    boxes[:, [1, 3, 6, 8, 10, 12]] -= top
    boxes[:, [0, 2, 5, 7, 9, 11]] /= r
    boxes[:, [1, 3, 6, 8, 10, 12]] /= r
    return boxes


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
    """ 准备模型运行的feed_dict """
    feed_dict = dict()
    for input_name in input_names:
        feed_dict[input_name] = image

    return feed_dict


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


def npu_ocr_infer(character, lpimg):
    """ 车牌字符识别推理模块 """
    lpimg = resize_norm_img(lpimg)
    img_data = lpimg[np.newaxis, :, :, :]
    img_data = torch.from_numpy(img_data)
    img_data = np.array(img_data, dtype=np.float32)

    # -----------------om  infer----------------------
    t5 = time.time()
    result_om = modelocr.execute([img_data, ])
    t6 = time.time()
    npu_ocr_time = t6 - t5
    # print(f"npu_ocr_infer_time_per_lp:{npu_ocr_time}")

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


def creat_new_dir(dir_path):
    """ 创建空白文件夹 """
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    else:
        pass


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

    """ 20240116说明：使用.om 进行车牌截取/矫正字符识别 华为昇腾服务器 """

    # https://blog.csdn.net/qq_22764813/article/details/133787584?spm=1001.2101.3001.6650.2&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-2-133787584-blog-115270800.235%5Ev38%5Epc_relevant_sort_base3&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-2-133787584-blog-115270800.235%5Ev38%5Epc_relevant_sort_base3&utm_relevant_index=5
    # 华为:ONNX模型转换成om
    # atc --model=yolov7-lite-s.onnx --framework=5 --input_format=NCHW --input_shape="images:1,3,768,1280"  --soc_version=Ascend310P3 --output=yolov7lpdet --log=error
    # batchsize 动态
    # atc --model=ocr_rec111302.onnx --framework=5 --input_format=NCHW --input_shape="input:-1,3,48,168" --dynamic_batch_size="1,2,4,8"   --soc_version=Ascend310P3 --output=ocr_rec111302 --log=error
    # atc --model=ocr_rec111302.onnx --framework=5 --input_format=NCHW --input_shape="input:1,3,48,168"  --soc_version=Ascend310P3 --output=ocr_rec111302 --log=error

    # 华为 om格式模型推理代码
    parser = argparse.ArgumentParser()
    parser.add_argument('--detect_model', type=str, default='./weight/yolov7-lite-s.om', help='model.pt path(s)')  # 检测模型
    parser.add_argument('--character_dict_path', type=str, default='./weight/chinese_plate_dict.txt', help='ocr dict path(s)')  # 检测模型
    parser.add_argument('--rec_model', type=str, default='./weight/ocr_rec111302.om', help='model path(s)')  # 识别模型
    parser.add_argument('--image_path', type=str, default='./images/img100', help='source')
    parser.add_argument('--img_size', type=int, default=(768, 1280), help='inference size (pixels)')
    parser.add_argument('--output', type=str, default='./run',help='source')
    parser.add_argument('--drawed_image_path', type=str, default='./run', help='')
    opt = parser.parse_args()

    # 创建空文件夹
    creat_new_dir(opt.output)
    creat_new_dir(opt.drawed_image_path)
    
    # 是否调用GPU, 默认调用
    clors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]

    img_size = opt.img_size
    save_path = opt.output
    image_dir = opt.image_path

    # 遍历文件（图片）绝对路径
    file_list = []
    allFilePath(opt.image_path, file_list)
    # print(f"file_list:{file_list}")

    # ---- det om  infer------
    MODEL_PATH = opt.detect_model
    acl_resource = AclLiteResource()
    acl_resource.init()
    model = AclLiteModel(MODEL_PATH)
    # ---- npu OCR infer------
    OCR_MODEL_PATH_OM = opt.rec_model
    modelocr = AclLiteModel(OCR_MODEL_PATH_OM)

    # 加载字典
    character = om_character_dict_mapping(opt.character_dict_path, use_space_char=True)

    npu_lp_det_total_time = 0  # Det compute infer time
    npuDetPrePostTotalTime = 0

    npu_lp_ocr_total_time = 0  # OCR compute infer time
    npuOcrPrePostTotalTime = 0
    lp_ocr_number = 0
    count = 0  # 统计计算的图片数量
    
    for pic_ in file_list:
        count += 1
        print(count, ":", pic_, end="\n")
        img = cv2.imread(pic_)  # bgr
        img0 = copy.deepcopy(img)
        
        # 1-1: 车牌检测前处理
        # 图片预处理：1.图片缩放 2.通道变换 3.归一化; 返回的是尺寸resize、归一化、通道顺序变换后的图片, (H,W,C(bgr))-->（C（rgb）,H,W）-->（batch=1, C（rgb）,H,W）
        t3 = time.time()
        img, r, left, top = detect_pre_precessing(img, img_size)  # 输入图片进行前处理，不涉及 onnx
        
        # ******************************* detection START******************************** 
        # 1-2: npu det 
        # 进行目标检测，得到预测结果：长度19：x,y,w,h,obj_conf,cls1_conf,cls2_conf,x1,y1,坐标置信度,x2,y2,坐标置信度,x3,y3,坐标置信度,x4,y4,坐标置信度
        # ------- license plate NPU det 推理时间计算 ----------------
        t1 = time.time()
        om_det_result_list = model.execute([img, ])  # om:车牌检测
        t2 = time.time()
        npu_det_infer_time = t2 - t1
        npu_lp_det_total_time += npu_det_infer_time
        # print(f"NPU:detInferTime：{npu_det_infer_time}")
                
        # 1-3: 车牌检测结果后处理
        om_det_result_list = np.array(om_det_result_list)
        om_det_result_list2 = np.squeeze(om_det_result_list, axis=0)
        # 检测重叠框过滤 ,xywh2xyxyy并还原到原图尺寸， 返回NMS后的框
        # 返回N*14的结果：xyxy,conf=类别得分*框的框的置信度, 四个角点（左上点、右上点、右下点、左下点），角点框置信度得分
        detResult = post_precessing(om_det_result_list2, r, left, top, conf_thresh=0.3, iou_thresh=0.45, num_cls=2)  # OM 不涉及 onnx 非极大值抑制
        t4 = time.time()
        detPrePostAllTime = t4-t3
        npuDetPrePostTotalTime += detPrePostAllTime
        # ******************************* detection END******************************** 
                      
        #  ------------  车牌字符OCR  START--------------
        dict_list = []
        # detbox： 1*14的结果：xyxy,conf=类别得分*框的框的置信度, 四个角点（左上点、右上点、右下点、左下点），角点框置信度得分
        for detbox in detResult:
            result_dict = {}
            lpbbox = detbox[:4].tolist()  # xyxy
            land_marks = detbox[5:13].reshape(4, 2)  # corner 四个角点（左上点、右上点、右下点、左下点）
            
            # 车牌检测前后处理时间
            t6 = time.time()
            roi_img = four_point_transform(img0, land_marks)  # 截取车牌图像进行矫正
            layer = int(detbox[-1])
            score = detbox[4]
            if layer == 1:  # 0表示单层， 1代表是双层车牌
                roi_img = get_split_merge(roi_img)

            # 矫正后车牌字符识别
            info, character_conf, npu_ocr_time = npu_ocr_infer(character, roi_img)
            t7 = time.time()
            npu_lp_ocr_total_time += npu_ocr_time  #NPU
            npuOcrPrePostTime = t7-t6
            npuOcrPrePostTotalTime += npuOcrPrePostTime
            lp_ocr_number += 1
            # print(f"NPU-OCR:char:{info},recConf:{character_conf}, time:{npu_ocr_time}")

            result_dict['bbox_x1y1x2y2'] = lpbbox  # 2Dbox
            result_dict['landmarks'] = land_marks.tolist()
            result_dict['layer'] = layer
            result_dict['plate_number'] = info
            result_dict['plate_number_conf'] = character_conf  # onnx
            result_dict['roi_height'] = roi_img.shape[0]
            dict_list.append(result_dict)
        # print(f"dict_list:{dict_list}")

        # 车牌检测与字符识别结果绘制到图像上
        ori_img = draw_result(img0, dict_list)
        img_name = os.path.basename(pic_)
        save_img_path = os.path.join(save_path, img_name)
        cv2.imwrite(save_img_path, ori_img)

    print("-------------------- 华为昇腾 NPU 车牌检测/字符识别推理评估---------------------")
    print(f"NPUDetAvgTime:{npu_lp_det_total_time/count}; image number:{count},  TotalTime:{npu_lp_det_total_time}")
    print(f"NPUDetPrePostAvgTime:{npuDetPrePostTotalTime/count}; image number:{count},  TotalTime:{npuDetPrePostTotalTime}\n")
    print(f"NPUOCRAvgTime:{npu_lp_ocr_total_time/lp_ocr_number}; license plate number: {lp_ocr_number}, TotalTime:{npu_lp_ocr_total_time}")
    print(f"NPUOCRPrePostAvgTotalTime:{npuOcrPrePostTotalTime/lp_ocr_number}; license plate number: {lp_ocr_number}, TotalTime:{npuOcrPrePostTotalTime}")




