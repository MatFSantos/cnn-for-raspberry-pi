from tflite_runtime.interpreter import Interpreter
import numpy as np
from PIL import Image

MODEL_PATH = "cnn_mnist_float32_compat.tflite"
IMAGE_PATH = "mnist_image.png"

# carregar modelo
interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# carregar imagem em grayscale
img = Image.open(IMAGE_PATH).convert("L")
img = img.resize((28, 28))
img = np.array(img, dtype=np.float32) / 255.0

# formato: (1, 28, 28, 1)
img = img.reshape(1, 28, 28, 1)

# rodar inferencia
interpreter.set_tensor(input_details[0]['index'], img)
interpreter.invoke()

output = interpreter.get_tensor(output_details[0]['index'])

print("Predição:", np.argmax(output))