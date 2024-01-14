__author__ = 'mkv-aql'

#Imports
import torch
from torchvision import models, transforms
from PIL import Image

#Use CUDA cores
torch.cuda.is_available() #To check if cuda cores are available

PATH = "Models/yolact_resnet50_54_800000.pth" #Change to correct path if needed, not pushed to github due to > 100MB

#Loading model
model = torch.load(PATH, map_location=torch.device('cpu'))
#model.eval() #set to evaluation mode

#Get and modify images size and normalize based on yolact model (in this case: 550x550)
image_transform = transforms.Compose([
    transforms.Resize((550, 550)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

image = Image.open('Images/Hamburg3.jpg')
image = image_transform(image).unsqueeze(0)  # Add batch dimension



#Inference
with torch.no_grad():  # No need to track gradients for inference
    predictions = model(image)


