import json
import onnxruntime

labels_file = "steelball1aml/labels.json"
onnx_model_path = "steelball1aml/model.onnx"

providers = [
    ('CUDAExecutionProvider', {
        'device_id': 0,
        'arena_extend_strategy': 'kNextPowerOfTwo',
        'gpu_mem_limit': 2 * 1024 * 1024 * 1024,
        'cudnn_conv_algo_search': 'EXHAUSTIVE',
        'do_copy_in_default_stream': True,
    }),
    'CPUExecutionProvider',
]

with open(labels_file) as f:
    classes = json.load(f)
print(classes)
try:
    session = onnxruntime.InferenceSession(onnx_model_path, providers=providers)
    print("ONNX model loaded...")
except Exception as e: 
    print("Error loading ONNX file: ", str(e))