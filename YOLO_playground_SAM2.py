from ultralytics import YOLO, SAM
from PIL import Image
import numpy as np

# model = YOLO('yolomodel/COCO/yolov8x.pt')
model = SAM('yolomodel/Segmentation/sam2_l.pt')
# Display model information (optional)
model.info()

# source = 'http://images.cocodataset.org/val2017/000000039769.jpg'
# source = 'Images/Hamburg2.jpg' # God for bboxes=[500, 500, 800, 800]
source = 'Images/Hamburg4.jpg' # God for bboxes=[250, 300, 1000, 600]
# source = '../input/10997885.jpg'

# model.predict(source=source, save=True, line_width = 5)
# model.predict(source=source, save=False, show=True, imgsz = 640, line_width = 1) #Will only show a glimpse of the predicted image
#Predict only Door (164)
# model.predict(source=source, save=True, show=True, classes= 164, imgsz = 640, line_width = 1) #Will only show a glimpse of the predicted image

# Get tensor results
# results = model('../input/10997885.jpg')

# # detect only doors tensors
# results = model(source=source, classes =[164,587]) #Will only print box tensors, 587 is for 'Window', 164 is for 'Door'

results = model(source=source,
                save = True,
                bboxes=[250, 300, 1000, 600]
                )

#Will only print box tensors
for r in results:
    print(r.boxes)
    # print(r.masks) #No masks
    # print(r.keypoints) #no Keypoints



#keep predicted image open
for r in results:
    im_array = r.plot() #plot bgr numpy array of prediction
    im = Image.fromarray(im_array[..., ::-1]) #RGB PIL Image
    im.show() #Will open windows's default photo viewer

#################################################################################################

# Run inference with bboxes prompt
model = SAM('yolomodel/Segmentation/sam2_l.pt')
source = 'Images/Hamburg2.jpg'
results = model(source, bboxes=[100, 100, 200, 200])

# Run inference with single point
results = model(points=[900, 370], labels=[1], save = True)

# Run inference with multiple points
results = model(points=[[400, 370], [900, 370]], labels=[1, 1], save = True)

# Run inference with multiple points prompt per object
results = model(points=[[[400, 370], [900, 370]]], labels=[[1, 1]], save = True)

# Run inference with negative points prompt
results = model(points=[[[400, 370], [900, 370]]], labels=[[1, 0]], save = True)