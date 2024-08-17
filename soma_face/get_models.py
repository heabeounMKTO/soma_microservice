import gdown
import os 

scrfd_output = "models/det_10g.onnx"
scrfd_model = 'https://drive.google.com/file/d/1syelTRVRVdoHfjAjMYVuDXYUJsFDTZXt/view?usp=sharing'

arcface_output = "models/arcfaceresnet100-8.onnx"
arcface_model = "https://drive.google.com/file/d/1oNxQiL35vopSN5wPRkdfWdh7cGxevvqU/view?usp=sharing"

yoloface_output = "models/yoloface_8n.onnx"
yoloface_model = "https://drive.google.com/file/d/1Z_Y3PvRnk__y-1zc2lT1z-uwLV9p1K4Z/view?usp=sharing"


if not os.path.exists(scrfd_output):
    print("downloading scrfd model")
    gdown.download(url=scrfd_model, output=scrfd_output, fuzzy=True, quiet=False)
else:
    print("scrfd model found.. skipping..")


if not os.path.exists(arcface_output):
    print("downloading arcface model")
    gdown.download(url=arcface_model, output=arcface_output, fuzzy=True, quiet=False)
else:
    print("arcface model found.. skipping..")


if not os.path.exists(yoloface_output):
    print("downloading yoloface model")
    gdown.download(url=yoloface_model, output=yoloface_output, fuzzy=True, quiet=False)
else:
    print("yoloface model found.. skipping..")

print("finished getting all model :]")

