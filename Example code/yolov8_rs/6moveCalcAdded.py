import cv2
import sys
import os
import time
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO
import torch

#WORKS TO TEXT ATM

W = 640
H = 480

FOV_X = 57.33  # Horizontal field of view in degrees
FOV_Y = 43  # Vertical field of view in degrees

m=0
A=0
B=0
C=0

#SET OFFSET BASED ON DEPTH CAMERA POSITION - ACCOUNT FOR CAMERA RECESS
D = 0
E = 0
F = 0




def box_center_to_corner(boxes):
    """Convert from (center, width, height) to (upper-left, lower-right)."""
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    boxes = np.stack((x1, y1, x2, y2), axis=-1)
    return boxes

device_id = 0  # Change this to your desired device index
torch.cuda.set_device(device_id)

config = rs.config()
config.enable_stream(rs.stream.color, W, H, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, W, H, rs.format.z16, 30)

pipeline = rs.pipeline()
profile = pipeline.start(config)

align_to = rs.stream.color
align = rs.align(align_to)

model = YOLO(r"C:\Users\TDKin\Desktop\Models\yolov8l.pt")

while True:
    frames = pipeline.wait_for_frames()

    aligned_frames = align.process(frames)
    color_frame = aligned_frames.get_color_frame()
    depth_frame = aligned_frames.get_depth_frame()
    if not color_frame:
        continue

    color_image = np.asanyarray(color_frame.get_data())
    depth_image = np.asanyarray(depth_frame.get_data())
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.08), cv2.COLORMAP_JET)

    results = model(color_image)

    min_depth = np.inf
    min_box = None

    for r in results:
        boxes = r.boxes
        if len(boxes) > 0:
            for box in boxes:
                b = box.xyxy[0].to('cpu').detach().numpy().copy()  # get box coordinates in (top, left, bottom, right) format
                c = box.cls
                cv2.rectangle(depth_colormap, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 0, 255),
                              thickness = 2, lineType=cv2.LINE_4)
                cv2.putText(depth_colormap, text = model.names[int(c)], org=(int(b[0]), int(b[1])),
                            fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 0.7, color = (0, 0, 255),
                            thickness = 2, lineType=cv2.LINE_4)

                # Calculate the center area of the bounding box
                x1 = int(b[0] + (b[2] - b[0]) / 15)
                x2 = int(b[2] - (b[2] - b[0]) / 15)
                y1 = int(b[1] + (b[3] - b[1]) / 15)
                y2 = int(b[3] - (b[3] - b[1]) / 15)

                # Extract the depth values from the center area of the box
                center_depth_values = depth_image[y1:y2, x1:x2]

                # Calculate the median depth value
                median_depth = np.median(center_depth_values)

                if median_depth < min_depth:
                    min_depth = median_depth
                    min_box = b

                # Convert depth value to inches
                median_depth_in_inches = median_depth / 100

                print("Median depth in inches: ", median_depth_in_inches)
    if min_box is not None:
        # Calculate the center of the box
        center_x = (min_box[0] + min_box[2]) / 2.0
        center_y = (min_box[1] + min_box[3]) / 2.0

        # Calculate the x and y degree
        x_degree = (center_x / W - 0.5) * FOV_X
        y_degree = (center_y / H - 0.5) * FOV_Y

        # Convert depth value to inches
        min_depth_in_inches = min_depth / 100

        print("X degree: ", x_degree)
        print("Y degree: ", y_degree)
        print("Minimum depth in inches: ", min_depth_in_inches)

        #Calculate trig
        m = np.sqrt(x_degree * x_degree + y_degree * y_degree)

        A = median_depth_in_inches  * np.sin(np.deg2rad(x_degree))#left/right
        B = -y_degree #down/up
        C = np.sqrt(median_depth_in_inches**2 - m**2)  #backwards/forwards



        print(A)
        print(B)
        print(C)





    

    annotated_frame = results[0].plot()

    cv2.imshow("color_image", annotated_frame)
    cv2.imshow("depth_image", depth_colormap)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
