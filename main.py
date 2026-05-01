import cv2
import numpy as np

lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

def correct_perspective(img, src_points, dst_points, output_size):
    padding=50
    padded_size = (output_size[0] + 2*padding, output_size[1] + 2*padding)
    src = np.float32(src_points)
    dst = np.float32([[padding, padding], 
                      [output_size[0] + padding, padding],
                      [output_size[0] + padding, output_size[1] + padding],
                      [padding, output_size[1] + padding]])
    
    M = cv2.getPerspectiveTransform(src, dst)
    warped_image = cv2.warpPerspective(img, M, padded_size)
    return warped_image


def find_new_pts(cur_frame):
    p1,st,err = cv2.calcOpticalFlowPyrLK(old_frame, cur_frame, old_corners,None, **lk_params)
    if p1 is not None:
        return p1
    else:
        print('Не найдены точки')
        exit()

def calc_tilt(corners):
    top = np.linalg.norm(corners[0]-corners[1])
    right = np.linalg.norm(corners[1]-corners[2])
    bottom = np.linalg.norm(corners[2]-corners[3])
    left = np.linalg.norm(corners[3]-corners[0])
    tb_ratio = min(top,bottom)/max(top,bottom)
    lr_ratio = min(left,right)/max(left,right)
    tb_tilt = (1-tb_ratio)*90
    lr_tilt = (1-lr_ratio)*90
    return tb_tilt, lr_tilt

cap = cv2.VideoCapture('VIDE.mp4')
res, frame = cap.read()
detector = cv2.QRCodeDetector()
data,bbox,straight_qrcode = detector.detectAndDecode(frame)
if data is None:
    print('Not found!')
    exit()
destination_points = [[0, 0], [600, 0], [600, 600], [0, 600]]
output_dimensions = (600, 600)
old_corners = bbox[0]
old_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) 
while True:
    res,clear_frame = cap.read()
    frame = clear_frame.copy()
    data,bbox,straight_qrcode = detector.detectAndDecode(frame)
    if data and bbox is not None:
        color = (255,0,0)
        corners = bbox[0]
        text = data
    else:
        color= (0,0,255)
        corners = find_new_pts(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY))
    corrected_img = correct_perspective(frame, corners, destination_points, output_dimensions)
    for i in range (len(corners)):
        pt1 = tuple(corners[i].astype(int))
        pt2 = tuple(corners[(i+1)%len(corners)].astype(int))
        cv2.line(frame, pt1, pt2, color, 3)
    tb_tilt,lr_tilt = calc_tilt(corners)
    cv2.putText(frame, f'H_tilt = {tb_tilt:.1f}' , (10,30),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),3)
    cv2.putText(frame, f'V_tilt = {lr_tilt:.1f}' , (10,60),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),3)
    q_data,_,_ = detector.detectAndDecode(corrected_img)
    if data is None and q_data is not None:
        text = q_data
    cv2.putText(frame, f'Text = {text}' , (10,90),cv2.FONT_HERSHEY_SIMPLEX,1,color,3)
    cv2.imshow("Corrected Image", cv2.resize(corrected_img,None,fx = 0.7,fy = 0.7))
    cv2.imshow('QR Code', cv2.resize(frame,None,fx = 0.7,fy = 0.7))
    old_corners = corners
    old_frame = cv2.cvtColor(clear_frame,cv2.COLOR_BGR2GRAY)
    k = cv2.waitKey(1) & 0XFF
    if k == ord('q'):
        cv2.destroyAllWindows()
        exit()
