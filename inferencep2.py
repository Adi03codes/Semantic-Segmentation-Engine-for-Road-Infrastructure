
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import torch
from torchvision.transforms import Compose, Resize, ToTensor
from torchvision.models.segmentation import deeplabv3_resnet50
from PIL import Image
import numpy as np
import io
import uvicorn

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = deeplabv3_resnet50(pretrained=False, num_classes=6)
model.load_state_dict(torch.load("deeplabv3_roadseg.pth", map_location="cpu"))
model.eval()

transform = Compose([
    Resize((512, 512)),
    ToTensor()
])

@app.post("/segment")
async def segment_image(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)['out'][0].argmax(0).numpy().tolist()
    return {"segmentation_mask": output}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

