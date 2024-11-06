from ultralytics import YOLO
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# model = YOLO.from_pretrained('jameslahm/yolov10n') straight from website
# model = YOLO('yolomodel/COCO/yolov8x.pt')
model = YOLO('yolomodel/Segmentation/yolov8x-seg.pt')
# model = YOLO('yolomodel/OpenImageV7/yolov10x.pt')
print(model.names) #Will print all the classes that the model can predict, 164 for 'Door'


# source = 'http://images.cocodataset.org/val2017/000000039769.jpg'
source = 'Images/Hamburg2.jpg'
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
                # save = True
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

# Open matplotlib window
plt.imshow
plt.imshow(im_array)
plt.show()
