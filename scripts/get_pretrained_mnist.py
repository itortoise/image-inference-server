"""下载预训练 MNIST ONNX 模型。"""
import os
import urllib.request

# ONNX Model Zoo MNIST model
URL = "https://github.com/onnx/models/raw/main/validated/vision/classification/mnist/model/mnist-12.onnx"
OUTPUT = "models/mnist_pretrained/1/model.onnx"

os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)

print(f"Downloading pretrained MNIST model...")
print(f"  URL: {URL}")
try:
    urllib.request.urlretrieve(URL, OUTPUT)
    size = os.path.getsize(OUTPUT)
    print(f"  ✓ Downloaded: {size/1024:.1f} KB")
except Exception as e:
    print(f"  ❌ Failed: {e}")
    print("  Trying alternative URL...")
    # Try alternative
    ALT_URL = "https://media.githubusercontent.com/media/onnx/models/main/validated/vision/classification/mnist/model/mnist-12.onnx"
    try:
        urllib.request.urlretrieve(ALT_URL, OUTPUT)
        size = os.path.getsize(OUTPUT)
        print(f"  ✓ Downloaded from alternative: {size/1024:.1f} KB")
    except Exception as e2:
        print(f"  ❌ Also failed: {e2}")
