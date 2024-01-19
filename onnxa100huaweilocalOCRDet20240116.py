import copy
import argparse
from PIL import Image, ImageDraw, ImageFont
import time
import cv2
import numpy
import numpy as np
import torch
import onnxruntime
import math
import os

mean_value, std_value = ((0.588, 0.193))  # 识别模型均值标准差


def allFilePath(rootPath, allFIleList):
    """遍历文件"""
    # jwang 0
    fileList = os.listdir(rootPath)
    for temp in fileList:
        if os.path.isfile(os.path.join(rootPath, temp)):
            allFIleList.append(os.path.join(rootPath, temp))
        else:
            allFilePath(os.path.join(rootPath, temp), allFIleList)


def detect_pre_precessing(img, img_size):
    """ 检测前处理:  1.图片缩放 2.通道变换 3.归一化 """

    img, r, left, top = my_letter_box(img, img_size)  # 图片缩放与填充
    # cv2.imwrite("1.jpg",img2)
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

    # return the warped image
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
    # print(result_str)
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
    """ 关键点排列 按照（左上，右上，右下，左下）的顺序排列 """
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
    """返回原图上面的坐标 """
    boxes[:, [0, 2, 5, 7, 9, 11]] -= left
    boxes[:, [1, 3, 6, 8, 10, 12]] -= top
    boxes[:, [0, 2, 5, 7, 9, 11]] /= r
    boxes[:, [1, 3, 6, 8, 10, 12]] /= r
    return boxes


# ------------------------------------------------------ocr---------------------------------------------------------

def license_plate_mapping(OCR_onnx_file, character_dict_path, use_space_char=True):
    """
    基于onnx结果进行车牌字符对应
    Args:
        onnx_file: 检测模型路径
        character_dict_path: 字符识别字典
        use_space_char: 字符空格是否使用
    Returns:获取加载的OCR模型、输入/输出名称： input/output、加载的字典

    """

    providers = onnx_infer_type(onnx_use_gpt=False)  # 是否调用GPU, 默认调用
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

    # 获取加载的OCR模型、输入/输出名称： input/output、加载的字典
    return sess, input_names, output_names, character


def resize_norm_img(img, image_shape=[3, 48, 168]):
    """ 车牌图像resize、归一化、通道顺序变换 """
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


# 准备模型运行的feed_dict
def process(input_names, image):
    feed_dict = dict()
    for input_name in input_names:
        feed_dict[input_name] = image

    return feed_dict


def get_ignored_tokens():
    return [0]


def decode(character, text_index, text_prob=None, is_remove_duplicate=False):
    """

    Args:
        character: 处理后的字典
        text_index: 模型预测的最大索引
        text_prob: 模型预测的每个位置的置信度
        is_remove_duplicate: 去除重复的字符

    Returns: 车牌号+平均置信度（每个字符置信度相加/字符总数）

    """
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


def ocr_infer(sess, character, output_names, input_names, lp_bbox_data, boxlp):
    """
    Args:
        sess: onnx加载的字符识别模型
        character: 处理后的字典
        output_names: 模型输入名称
        input_names: 模型输出名称
        lp_bbox_data:cv读取的图片上截取车牌区域图
        boxlp:检测模型预测得到的结果:N*14, N=1
    """

    # img_onnx = cv2.imread(image_path)
    img_onnx = lp_bbox_data  # height, width, channel
    img_onnx = resize_norm_img(img_onnx)  # （3 48 168）
    onnx_indata = img_onnx[np.newaxis, :, :, :]
    onnx_indata = torch.from_numpy(onnx_indata)
    onnx_indata = np.array(onnx_indata, dtype=np.float32)
    feed_dict = process(input_names, onnx_indata)

    # ----------- ocr onnx infer------
    t5 = time.time()
    # onnx 模型推理：输入{input:[1,3,48,168]};输出：{}
    output_onnx = sess.run(output_names, feed_dict)  # 最内部的长度为85=blank+字典（83）+' '长度一致
    t6 = time.time()
    ocr_onnx_infer_time = t6 - t5
    # print(f"onnx_ocr_infer_time:{t6-t5}")
    output_onnx = numpy.asarray(output_onnx[0])  # （1，21，85）
    preds_idx = output_onnx.argmax(axis=2)
    preds_prob = output_onnx.max(axis=2)
    # 字符已去重复：车牌号+平均置信度（每个字符置信度相加/字符总数）
    post_result = decode(character, preds_idx, preds_prob, is_remove_duplicate=True)
    # print(f"post_result:{post_result}")
    if isinstance(post_result, dict):
        rec_info = dict()
        for key in post_result:
            if len(post_result[key][0]) >= 2:
                rec_info[key] = {
                    "label": post_result[key][0][0],
                    "score": float(post_result[key][0][1]),
                }
        # print(boxlp[1], rec_info)
    else:
        if len(post_result[0]) >= 2:
            # info = post_result[0][0] + "\t" + str(post_result[0][1])
            info = post_result[0][0]
            info_conf = post_result[0][1]
            # print(boxlp[1], info, info_conf)
            return info, info_conf, ocr_onnx_infer_time


def creat_new_dir(dir_path):
    """ 创建空白文件夹 """
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


if __name__ == "__main__":

    """ 20240116说明：使用.onnx 进行车牌截取/矫正字符识别 RTX3090/A100/华为昇腾服务器 """

    # https://blog.csdn.net/qq_22764813/article/details/133787584?spm=1001.2101.3001.6650.2&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-2-133787584-blog-115270800.235%5Ev38%5Epc_relevant_sort_base3&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-2-133787584-blog-115270800.235%5Ev38%5Epc_relevant_sort_base3&utm_relevant_index=5

    # 华为:ONNX模型转换成om
    # atc --model=yolov7-lite-s.onnx --framework=5 --input_format=NCHW --input_shape="images:1,3,768,1280"  --soc_version=Ascend310P3 --output=yolov7lpdet --log=error
    # batchsize 动态
    # atc --model=ocr_rec111302.onnx --framework=5 --input_format=NCHW --input_shape="input:-1,3,48,168" --dynamic_batch_size="1,2,4,8"   --soc_version=Ascend310P3 --output=ocr_rec111302 --log=error
    # atc --model=ocr_rec111302.onnx --framework=5 --input_format=NCHW --input_shape="input:1,3,48,168"  --soc_version=Ascend310P3 --output=ocr_rec111302 --log=error

    parser = argparse.ArgumentParser()
    # local RTX3090
    # parser.add_argument('--detect_model', type=str, default='/home/dell/桌面/huaweiDetOCR/weights/platedet/yolov7-lite-s.onnx', help='model.pt path(s)')  # 检测模型
    # parser.add_argument('--character_dict_path', type=str, default='/home/dell/桌面/huaweiDetOCR/weights/paddleocr/paddleocr20231113/chinese_plate_dict.txt',help='ocr dict path(s)')  # 检测模型
    # parser.add_argument('--rec_model', type=str, default='/home/dell/桌面/huaweiDetOCR/weights/paddleocr/paddleocr20231113/ocr_rec111302.onnx', help='model.pt path(s)')  # 识别模型
    # # parser.add_argument('--image_path', type=str, default='/home/dell/桌面/huaweiDetOCR/images/img100', help='source')
    # parser.add_argument('--image_path', type=str, default='/home/dell/桌面/huaweiDetOCR/images/img100', help='source')
    # parser.add_argument('--output', type=str, default='/home/dell/桌面/huaweiDetOCR/runs', help='source')
    # parser.add_argument('--drawed_image_path', type=str, default='/home/dell/桌面/huaweiDetOCR/runs/drawed', help='source')

    # 华为昇腾
    parser.add_argument('--detect_model', type=str, default='./weight/yolov7-lite-s.onnx', help='model.pt path(s)')  # 检测模型
    parser.add_argument('--character_dict_path', type=str, default='./weight/chinese_plate_dict.txt',help='ocr dict path(s)')  # 检测模型
    parser.add_argument('--rec_model', type=str, default='./weight/ocr_rec111302.onnx', help='model.pt path(s)')  # 识别模型
    # parser.add_argument('--image_path', type=str, default='/home/dell/桌面/huaweiDetOCR/images/img100', help='source')
    parser.add_argument('--image_path', type=str, default='./images/img100', help='source')
    parser.add_argument('--output', type=str, default='./runs', help='source')
    parser.add_argument('--drawed_image_path', type=str, default='./runs/drawed', help='source')

    # # A100
    # parser.add_argument('--detect_model', type=str, default='/home/dell/桌面/huaweiDetOCR/weights/platedet/yolov7-lite-s.onnx', help='model.pt path(s)')  # 检测模型
    # parser.add_argument('--character_dict_path', type=str, default='/home/dell/桌面/huaweiDetOCR/weights/paddleocr/paddleocr20231113/chinese_plate_dict.txt',help='ocr dict path(s)')  # 检测模型
    # parser.add_argument('--rec_model', type=str, default='/home/dell/桌面/huaweiDetOCR/weights/paddleocr/paddleocr20231113/ocr_rec111302.onnx', help='model.pt path(s)')  # 识别模型
    # # parser.add_argument('--image_path', type=str, default='/home/dell/桌面/huaweiDetOCR/images/img100', help='source')
    # parser.add_argument('--image_path', type=str, default='/home/dell/桌面/huaweiDetOCR/images/img100', help='source')
    # parser.add_argument('--output', type=str, default='/home/dell/桌面/huaweiDetOCR/runs', help='source')
    # parser.add_argument('--drawed_image_path', type=str, default='/home/dell/桌面/huaweiDetOCR/runs/drawed', help='source')

    parser.add_argument('--img_size', type=int, default=(768, 1280), help='inference size (pixels)')  # 640
    opt = parser.parse_args()

    # 创建空文件夹
    creat_new_dir(opt.output)
    creat_new_dir(opt.drawed_image_path)
    
    # 遍历文件（图片）绝对路径
    file_list = []
    allFilePath(opt.image_path, file_list)
    # print(f"file_list:{file_list}")

    clors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]

    img_size = opt.img_size
    save_path = opt.output
    image_dir = opt.image_path

    # 加载ONNX格式的检测/OCR模型
    use_gpu_infer = False
    #use_gpu_infer = True
    if use_gpu_infer:
        providers = onnx_infer_type(onnx_use_gpt=True)  # 调用GPU, 默认调用
    else:
        providers = onnx_infer_type(onnx_use_gpt=False)  # 不调用GPU
    session_detect = onnxruntime.InferenceSession(opt.detect_model, providers=providers)  # onnx: detection模型
    session_rec = onnxruntime.InferenceSession(opt.rec_model, providers=providers)  # onnx:车牌OCR模型

    onnx_det_total_time = 0     # Det compute: infer time
    onnxDetPrePostTotalTime = 0

    onnx_lp_ocr_total_time = 0  # OCR compute: infer time
    onnxOcrPrePostTotalTime = 0

    lp_ocr_number = 0
    count = 0                   # 统计检测图片数量

    begin = time.time()
    for pic_ in file_list:
        count += 1
        print(count, ":", pic_, end="\n")
        img = cv2.imread(pic_)  # bgr
        img2 = img
        img0 = copy.deepcopy(img)
        # ***************************  ONNX detection START **********************
        t3 = time.time()
        # 1-1 图片预处理：1.图片缩放 2.通道变换 3.归一化; 返回的是尺寸resize、归一化、通道顺序变换后的图片, (H,W,C(bgr))-->（C（rgb）,H,W）-->（batch=1, C（rgb）,H,W）
        img, r, left, top = detect_pre_precessing(img, img_size)  # 不涉及 onnx
        # 1-2：进行目标检测，预测结果：长度19：x,y,w,h,obj_conf,cls1_conf,cls2_conf,x1,y1,坐标置信度,x2,y2,坐标置信度,x3,y3,坐标置信度,x4,y4,坐标置信度
        t1 = time.time()
        # 1 60480 19
        detRes = session_detect.run([session_detect.get_outputs()[0].name], {session_detect.get_inputs()[0].name: img})[0]
        t2 = time.time()
        det_time_gap = t2-t1
        onnx_det_total_time += det_time_gap

        # 1-3：Det检测后处理, NMS重叠框过滤 ,xywh2xyxyy并还原到原图尺寸
        # 返回N*14的结果：xyxy,conf=类别得分*框的框的置信度, 四个角点（左上点、右上点、右下点、左下点），角点框置信度得分
        outputs = post_precessing(detRes, r, left, top, conf_thresh=0.3, iou_thresh=0.45, num_cls=2)  # ONNX不涉及 onnx 非极大值抑制
        t4 = time.time()
        detPrePostAllTime = t4-t3
        onnxDetPrePostTotalTime += detPrePostAllTime

        #  ***************************   ONNX detection END; 车牌字符OCR  START ***************************
        # 获取加载的OCR模型、输入/输出名称： input/output、加载的字典
        sess, input_names, output_names, character = license_plate_mapping(opt.rec_model, opt.character_dict_path, use_space_char=True)
        sess = session_rec
        dict_list = []
        # detbbox： 1*14的结果：xyxy,conf=类别得分*框的框的置信度, 四个角点（左上点、右上点、右下点、左下点），角点框置信度得分
        for detbbox in outputs:
            # lp_bbox_data：cv读取的图片上截取车牌区域图
            lp_bbox_data = img2[int(detbbox[1]):int(detbbox[3]), int(detbbox[0]):int(detbbox[2])]

            # 按照路径读取车牌小图
            # save cut small lp image
            # save_img_path = os.path.join(opt.drawed_image_path, str(detbbox[1])+'.jpg')
            # cv2.imwrite(save_img_path, lp_bbox_data)

            result_dict = {}
            rect = detbbox[:4].tolist()  # xyxy
            land_marks = detbbox[5:13].reshape(4, 2)  # corner 四个角点（左上点、右上点、右下点、左下点）

            t6 = time.time()
            roi_img = four_point_transform(img0, land_marks)  # 截取车牌图像进行矫正
            layer = int(detbbox[-1])
            score = detbbox[4]
            if layer == 1:  # 0表示单层， 1代表双层车牌
                roi_img = get_split_merge(roi_img)

            # RTX3090 读取小图进行车牌字符识别
            # result, lpconf, onnx_infer_time = ocr_infer(sess, character, output_names, input_names, save_img_path)
            # 未矫正车牌字符识别
            # result, lpconf, onnx_infer_time = ocr_infer(sess, character, output_names, input_names, lp_bbox_data, detbbox)
            # 矫正后车牌字符识别
            ocr_result, ocr_conf, onnx_infer_time = ocr_infer(sess, character, output_names, input_names, roi_img, detbbox)

            t7 = time.time()
            onnxOcrPrePostTime = t7 - t6
            onnx_lp_ocr_total_time += onnx_infer_time
            onnxOcrPrePostTotalTime += onnxOcrPrePostTime
            lp_ocr_number += 1

            result_dict['bbox_x1y1x2y2'] = rect
            result_dict['landmarks'] = land_marks.tolist()
            result_dict['layer'] = layer
            result_dict['plate_number'] = ocr_result
            result_dict['plate_number_conf'] = ocr_conf
            result_dict['roi_height'] = roi_img.shape[0]
            dict_list.append(result_dict)
        # print(f"dict_list:{dict_list}")
        # 检测识别结果绘制到图像上进行保存
        ori_img = draw_result(img0, dict_list)
        img_name = os.path.basename(pic_)
        save_img_path = os.path.join(save_path, img_name)
        cv2.imwrite(save_img_path, ori_img)

    print("-------------------- 华为昇腾 ONNX 车牌检测/字符识别推理评估---------------------")
    print(f"onnxDetAvgTime:{onnx_det_total_time / count}; image number:{count}, TotalTime:{onnx_det_total_time}")
    print(f"onnxDetPrePostAvgTime:{onnxDetPrePostTotalTime / count}; image number:{count}, TotalTime:{onnxDetPrePostTotalTime}\n")
    print(f"onnxOCRAvgTime:{onnx_lp_ocr_total_time / lp_ocr_number}; license plate number:{lp_ocr_number}, TotalTime:{onnx_lp_ocr_total_time}")
    print(f"onnxOCRPrePostAvgTotalTime:{onnxOcrPrePostTotalTime / lp_ocr_number}; license plate number:{lp_ocr_number}, TotalTime:{onnxOcrPrePostTotalTime}\n")



