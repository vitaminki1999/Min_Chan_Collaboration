import argparse
import time
import os
from pathlib import Path
import cv2
import csv
import torch
import numpy as np

# Conclude setting / general reprocessing / plots / metrices / datasets
from utils.utils import \
    time_synchronized,select_device, increment_path,\
    scale_coords,xyxy2xywh,non_max_suppression,split_for_trace_model,\
    driving_area_mask,lane_line_mask,plot_one_box,show_seg_result,\
    AverageMeter,\
    LoadImages

######입력해야함##########입력해야함###########입력해야함############
video_number = 5
base_path = "C:\\Users\\jungmingi\\Desktop\\jung\\180_yolo\\Final_test\\"
##################################################################

def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='data/weights/yolopv2.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default=os.path.join(base_path, f"data\\sample{video_number}.mp4"), help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    return parser


def detect():
    # setting and directories
    source, weights,  save_txt, imgsz = opt.source, opt.weights,  opt.save_txt, opt.img_size
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images

    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    inf_time = AverageMeter()
    waste_time = AverageMeter()
    nms_time = AverageMeter()

    # Load model
    stride =32
    model  = torch.jit.load(weights)
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA
    model = model.to(device)

    if half:
        model.half()  # to FP16  
    model.eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()

    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0

        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        [pred,anchor_grid],seg,ll= model(img)
        t2 = time_synchronized()

        # waste time: the incompatibility of  torch.jit.trace causes extra time consumption in demo version 
        # but this problem will not appear in offical version 
        tw1 = time_synchronized()
        pred = split_for_trace_model(pred,anchor_grid)
        tw2 = time_synchronized()

        # Apply NMS
        t3 = time_synchronized()
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t4 = time_synchronized()

        da_seg_mask = driving_area_mask(seg) #도로
        ll_seg_mask = lane_line_mask(ll)    #라인

        # Process detections
        for i, det in enumerate(pred):  # detections per image
          
            p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    #s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img :  # Add bbox to image
                        plot_one_box(xyxy, im0, line_thickness=3)

            # Print time (inference)
            print(f'{s}Done. ({t2 - t1:.3f}s)')

            #추가 마스크 누적
            if 'accumulated_lane_mask' not in locals():
                accumulated_lane_mask = np.zeros_like(ll_seg_mask, dtype=np.int32)

            show_seg_result(im0, (np.zeros_like(da_seg_mask),ll_seg_mask), is_demo=True)

            #추가 마스크 누적
            #accumulated_lane_mask = np.logical_or(accumulated_lane_mask, ll_seg_mask).astype(np.uint8)
            accumulated_lane_mask += ll_seg_mask
            final_mask = (accumulated_lane_mask >= 20).astype(np.uint8)

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                    print(f" The image with the result is saved in: {save_path}")
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            #w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            #h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            w,h = im0.shape[1], im0.shape[0]
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)
            cv2.imwrite(f"final_mask_{video_number}.png", final_mask * 255)
    inf_time.update(t2-t1,img.size(0))
    nms_time.update(t4-t3,img.size(0))
    waste_time.update(tw2-tw1,img.size(0))
    print('inf : (%.4fs/frame)   nms : (%.4fs/frame)' % (inf_time.avg,nms_time.avg))
    print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    opt =  make_parser().parse_args()
    print(opt)

    with torch.no_grad():
            detect()

#########################################

# 1. 원본 이미지 로드 (차선이 흰색으로 있는 이진 이미지)
img = cv2.imread(f"final_mask_{video_number}.png", cv2.IMREAD_GRAYSCALE)  # 이미지는 흑백 이미지여야 합니다.

# 2. 이진화 (차선이 흰색, 배경이 검정색이면 이미 이진화가 되어있겠지만, 임계값을 설정할 수도 있습니다.)
_, binary_mask = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

# 3. 흰색 차선 부분의 좌표를 찾기 (흰색 차선은 255, 배경은 0)
# findContours로 흰색 영역의 경계를 찾습니다.
contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 4. 차선의 중심선을 1차 함수로 근사 (차선이 직선이라 가정)
lane_lines = []  # 차선의 직선들 저장

if video_number==7:
    num=100
else:
    num=300

for contour in contours:
    # 각 컨투어의 중심선 구하기 (여기서는 단순히 모든 점을 사용)
    if len(contour) > num:  # 충분한 점이 있을 때만 사용
        # 포인트들에서 1차 함수(직선)로 근사
        fit = np.polyfit(contour[:, 0, 0], contour[:, 0, 1], 1)  # 1차 함수로 근사: y = mx + b
        lane_lines.append(fit)  # 기울기(m)와 절편(b)을 저장

# 5. 직선을 그릴 새로운 이미지를 생성
img_with_lane = np.zeros_like(img)  # 원본 이미지 크기와 동일한 새 이미지

# 차선 함수 저장
temp = []

# 6. 각 1차 함수(직선)을 따라 직선을 그리기
for fit in lane_lines:
    m, b = fit  # 기울기(m)와 절편(b)
    temp.append([m,b])
    # x 값의 범위 (이미지 너비를 기준으로)
    x_vals = np.linspace(0, img.shape[1], 1000)  # x 값은 0부터 이미지 너비까지
    
    # y 값 계산: y = mx + b
    y_vals = m * x_vals + b
    
    # 차선의 직선 그리기
    for i in range(len(x_vals) - 1):
        cv2.line(img_with_lane, (int(x_vals[i]), int(y_vals[i])), (int(x_vals[i+1]), int(y_vals[i+1])), (255), 1)

# 7. 결과 이미지 보기
cv2.imshow("Lane Image with Fitted Lines", img_with_lane)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 8. 결과 이미지 저장

cv2.imwrite(os.path.join(base_path, f"output\\sample{video_number}\\line_functions_{video_number}.png"), img_with_lane)

print(temp)
print(img.shape[1], img.shape[0])

# CSV 파일에 기울기 및 절편 저장
csv_file_name = os.path.join(base_path, f"data_line\\line_functions_{video_number}.csv")  # 5, 7, 11 등 파일명 적절히 수정

with open(csv_file_name, "w", newline="") as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(["Slope", "Intercept"])  # 헤더 작성
    for line in temp:
        csv_writer.writerow(line)  # 데이터 작성