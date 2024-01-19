import copy
import argparse
import time
import cv2
import numpy as np
import torch
import onnxruntime
import os

mean_value, std_value = ((0.588, 0.193))  # 识别模型均值标准差


def getAllFileAbsPath(rootPath, allFIleList):
    """ 遍历文件 ，得到绝对路径"""
    fileList = os.listdir(rootPath)
    for temp in fileList:
        if os.path.isfile(os.path.join(rootPath, temp)):
            allFIleList.append(os.path.join(rootPath, temp))
        else:
            getAllFileAbsPath(os.path.join(rootPath, temp), allFIleList)


def detect_pre_precessing(img, img_size):
    """ 检测前处理：1.图片缩放 2.通道变换  3.归一化 """
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

    return warped   # return the warped image


def draw_result(orgimg, dict_list):
    """ 将检测的关键点和检测框绘制到图片上 """
    for result in dict_list:
        rect_area = result['bbox_x1y1x2y2']
        x, y, w, h = rect_area[0], rect_area[1], rect_area[2] - rect_area[0], rect_area[3] - rect_area[1]
        padding_w = 0.05 * w
        padding_h = 0.11 * h
        rect_area[0] = max(0, int(x - padding_w))
        rect_area[1] = min(orgimg.shape[1], int(y - padding_h))
        rect_area[2] = max(0, int(rect_area[2] + padding_w))
        rect_area[3] = min(orgimg.shape[0], int(rect_area[3] + padding_h))

        landmarks = result['landmarks']
        for i in range(4):  # 关键点
            cv2.circle(orgimg, (int(landmarks[i][0]), int(landmarks[i][1])), 5, clors[i], -1)
        cv2.rectangle(orgimg, (rect_area[0], rect_area[1]), (rect_area[2], rect_area[3]), (0, 0, 255), 2)  # 画框

    return orgimg


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


def get_split_merge(img):  # 双层车牌进行分割后识别
    """ 双层车牌进行分割后识别 """
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


def xywh2xyxy(boxes):  # xywh坐标变为 左上 ，右下坐标 x1,y1  x2,y2
    """xywh坐标变为 左上 ，右下坐标 x1,y1  x2,y2"""
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
    """返回原图上面的坐标"""
    boxes[:, [0, 2, 5, 7, 9, 11]] -= left
    boxes[:, [1, 3, 6, 8, 10, 12]] -= top
    boxes[:, [0, 2, 5, 7, 9, 11]] /= r
    boxes[:, [1, 3, 6, 8, 10, 12]] /= r
    return boxes


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


if __name__ == "__main__":
    """  20240115说明： 使用ONNX 进行车牌检测 """

    # https://blog.csdn.net/qq_22764813/article/details/133787584?spm=1001.2101.3001.6650.2&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-2-133787584-blog-115270800.235%5Ev38%5Epc_relevant_sort_base3&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-2-133787584-blog-115270800.235%5Ev38%5Epc_relevant_sort_base3&utm_relevant_index=5

    # 华为:ONNX模型转换成om
    # atc --model=yolov7-lite-s.onnx --framework=5 --input_format=NCHW --input_shape="images:1,3,768,1280"  --soc_version=Ascend310P3 --output=yolov7lpdet --log=error
    # batchsize 动态
    # atc --model=ocr_rec111302.onnx --framework=5 --input_format=NCHW --input_shape="input:-1,3,48,168" --dynamic_batch_size="1,2,4,8"   --soc_version=Ascend310P3 --output=ocr_rec111302 --log=error
    # atc --model=ocr_rec111302.onnx --framework=5 --input_format=NCHW --input_shape="input:1,3,48,168"  --soc_version=Ascend310P3 --output=ocr_rec111302 --log=error

    # 华为官网参考示例
    parser = argparse.ArgumentParser()
    # # local
    # parser.add_argument('--detect_model', type=str, default='/home/dell/桌面/huaweiDetOCR/weights/platedet/yolov7-lite-s.onnx', help='model.pt path(s)')  # 检测模型
    # parser.add_argument('--image_path', type=str, default='/home/dell/桌面/huaweiDetOCR/images/img100', help='source')
    # # parser.add_argument('--image_path', type=str, default='/home/dell/桌面/huaweiDetOCR/images/img2', help='source')
    # parser.add_argument('--output', type=str, default='/home/dell/桌面/huaweiDetOCR/runs', help='source')
    # parser.add_argument('--drawed_image_path', type=str, default='/home/dell/桌面/huaweiDetOCR/runs/drawed', help='source')

    # A100
    # parser.add_argument('--detect_model', type=str, default='/home/jinbo/items/huaweionnxtest/testhuaweiOnnx/weight/plateDet/yolov7-lite-s.onnx', help='model.pt path(s)')  # 检测模型
    # parser.add_argument('--image_path', type=str, default='/home/jinbo/items/huaweionnxtest/testhuaweiOnnx/images/img2', help='source')
    # # parser.add_argument('--image_path', type=str, default='/home/dell/桌面/huaweiDetOCR/images/img2', help='source')
    # parser.add_argument('--output', type=str, default='/home/jinbo/items/huaweionnxtest/testhuaweiOnnx/runs', help='source')
    # parser.add_argument('--drawed_image_path', type=str, default='/home/jinbo/items/huaweionnxtest/testhuaweiOnnx/runs/drawed', help='source')

    # huawei
    parser.add_argument('--detect_model', type=str, default='./weight/yolov7-lite-s.onnx', help='model.pt path(s)')  # 检测模型
    parser.add_argument('--image_path', type=str, default='./images/img100', help='source')
    # parser.add_argument('--image_path', type=str, default='./images/img100', help='source')
    parser.add_argument('--output', type=str, default='./runs', help='source')
    parser.add_argument('--drawed_image_path', type=str, default='./runs/drawed', help='source')

    parser.add_argument('--img_size', type=int, default=(768, 1280), help='inference size (pixels)')  # 640

    opt = parser.parse_args()
    creat_new_dir(opt.output)
    creat_new_dir(opt.drawed_image_path)  # 创建保存结果的空文件夹


    # 遍历文件（图片）绝对路径
    file_list = []
    getAllFileAbsPath(opt.image_path, file_list)
    clors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]

    img_size = opt.img_size
    save_path = opt.output
    image_dir = opt.image_path

    # 加载ONNX格式的检测/OCR模型
    # use_gpu_infer = False
    use_gpu_infer = True
    if use_gpu_infer:
        providers = onnx_infer_type(onnx_use_gpt=True)  # 调用GPU, 默认调用
    else:
        providers = onnx_infer_type(onnx_use_gpt=False)  # 不调用GPU
    session_detect = onnxruntime.InferenceSession(opt.detect_model, providers=providers)  # onnx: detection模型

    detTotalTime = 0     # Det compute: infer time
    prePostDetRunAllTime = 0
    count = 0                   # 统计检测图片数量
    for pic_ in file_list:
        count += 1
        print(count, ":", pic_, end="\n")
        img = cv2.imread(pic_)  # # image read get : bgr
        img2 = img
        img0 = copy.deepcopy(img)

        t3 = time.time()
        # ***************************  ONNX detection START **********************
        # 1-1 图片预处理：1.图片缩放 2.通道变换 3.归一化; 返回的是尺寸resize、归一化、通道顺序变换后的图片,
        # (H,W,C(bgr))-->（C（rgb）,H,W）-->（batch=1, C（rgb）,H,W）
        img, r, left, top = detect_pre_precessing(img, img_size)  # 不涉及 onnx

        # 1-2：进行目标检测，预测结果：长度19：x,y,w,h,obj_conf,cls1_conf,cls2_conf,x1,y1,坐标置信度,x2,y2,坐标置信度,x3,y3,坐标置信度,x4,y4,坐标置信度
        t1 = time.time()
        detRes = session_detect.run([session_detect.get_outputs()[0].name], {session_detect.get_inputs()[0].name: img})[0]
        t2 = time.time()
        det_time_gap = t2-t1
        detTotalTime += det_time_gap

        # 1-3：Det检测后处理, NMS重叠框过滤 ,xywh2xyxyy并还原到原图尺寸
        # 返回N*14的结果：xyxy,conf=类别得分*框的框的置信度, 四个角点（左上点、右上点、右下点、左下点），角点框置信度得分
        outputs = post_precessing(detRes, r, left, top, conf_thresh=0.3, iou_thresh=0.45, num_cls=2)  # ONNX不涉及 onnx 非极大值抑制

        t4 = time.time()
        pre_post_det_run_time = t4 - t3
        prePostDetRunAllTime += pre_post_det_run_time

        #  ***************************   ONNX detection END， 以下为检测结果可视化保存  ***************************

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
            rect = detbbox[:4].tolist()                       # xyxy
            land_marks = detbbox[5:13].reshape(4, 2)          # corner 四个角点（左上点、右上点、右下点、左下点）
            roi_img = four_point_transform(img0, land_marks)  # 截取车牌图像进行矫正
            layer = int(detbbox[-1])
            score = detbbox[4]
            if layer == 1:     # 0表示单层， 1表示双层车牌
                roi_img = get_split_merge(roi_img)
            result_dict['bbox_x1y1x2y2'] = rect
            result_dict['landmarks'] = land_marks.tolist()
            result_dict['layer'] = layer
            result_dict['roi_height'] = roi_img.shape[0]
            dict_list.append(result_dict)

        # 检测识别结果绘制到图像上进行保存
        ori_img = draw_result(img0, dict_list)
        img_name = os.path.basename(pic_)
        save_img_path = os.path.join(save_path, img_name)
        cv2.imwrite(save_img_path, ori_img)


    # print("-------------------- RTX3090 ONNX GPU 车牌检测推理评估---------------------")
    print("-------------------- 华为昇腾 ONNX CPU 车牌检测推理评估---------------------")
    print(f"ONNXDetAvgTime:{detTotalTime / count}; imgN:{count}, TotalTime:{detTotalTime}")
    print(f"ONNXPrePosDetAvgTime:{prePostDetRunAllTime / count}; imgN:{count}, prePostDetRunAllTime:{prePostDetRunAllTime}\n")




