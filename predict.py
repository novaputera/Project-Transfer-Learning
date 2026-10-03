import argparse
import numpy as np
import tensorflow as tf
from PIL import Image

parser = argparse.ArgumentParser(description="Prediksi gambar Nexus Omni")
parser.add_argument("--image", required=True, help="Path gambar")
args = parser.parse_args()

model = tf.keras.models.load_model("models/nexus_mobilenetv2.keras")
image = Image.open(args.image).convert("RGB").resize((224, 224))
x = np.expand_dims(np.array(image, dtype=np.float32), axis=0)
probability = float(model.predict(x, verbose=0)[0][0])
label = "obstacle" if probability >= 0.5 else "normal"

print("Hasil prediksi:", label)
print("Probabilitas obstacle:", f"{probability:.4f}")
