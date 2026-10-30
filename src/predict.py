import sys
import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T

from dataset import CLASSES
from models import get_model

model_name = sys.argv[1]      # baseline | resnet18
img_path = sys.argv[2]        # path to your image

device = "cuda" if torch.cuda.is_available() else "cpu"
model = get_model(model_name).to(device)
model.load_state_dict(torch.load(f"../checkpoints/{model_name}.pt", map_location=device))
model.eval()

# 1. load image as grayscale
gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
if gray is None:
    sys.exit(f"Could not read image: {img_path}")

# 2. detect the face and crop it
cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))
if len(faces) == 0:
    print("No face detected, using the whole image (result may be poor)")
    face = gray
else:
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])   # largest face
    face = gray[y:y + h, x:x + w]

# 3. same preprocessing as validation/test
face = cv2.resize(face, (48, 48))
tf = T.Compose([T.ToPILImage(), T.ToTensor(), T.Normalize([0.5], [0.5])])
x = tf(face).unsqueeze(0).to(device)          # shape [1, 1, 48, 48]

# 4. predict
with torch.no_grad():
    probs = torch.softmax(model(x), dim=1)[0].cpu().numpy()

for i in np.argsort(-probs):
    print(f"{CLASSES[i]:10s} {probs[i]*100:5.1f}%")

cv2.imwrite("../results/last_face.png", cv2.resize(face, (192, 192)))
print("\nCropped face saved to ../results/last_face.png")