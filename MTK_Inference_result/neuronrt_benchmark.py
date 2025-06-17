from utils.neuronpilot.data import convert_to_binary, conert_to_numpy
from utils.neuronpilot import neuronrt
import argparse, time, warnings, shutil, os
import numpy as np
import tensorflow as tf
import psutil  , os

warnings.simplefilter('ignore')

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--tflite_model", type=str, help="Path to .tflite")
parser.add_argument("-d", "--device", type=str, default='mdla3.0', choices=['mdla3.0', 'mdla2.0', 'vpu'], help="Device to use")
parser.add_argument("-t", "--iteration", default=10, type=int, help="How many times to run inference")
args = parser.parse_args()

interpreter = neuronrt.Interpreter(model_path=args.tflite_model, device=args.device)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Create random input
inputs = np.random.rand(*input_details[0]['shape']).astype(input_details[0]['dtype'])

# Process info for memory tracking
process = psutil.Process(os.getpid())

# Set tensor
t1 = time.time()
interpreter.set_tensor(input_details[0]['index'], inputs)
t2 = time.time()

# Inference + memory tracking
mem_usages = []
for _ in range(args.iteration):
    mem_before = process.memory_info().rss
    interpreter.invoke()
    mem_after = process.memory_info().rss
    mem_usages.append(mem_after - mem_before)

t3 = time.time()
outputs = interpreter.get_tensor(output_details[0]['index'])

avg_mem_usage_kb = sum(mem_usages) / len(mem_usages) / 1024

# Report
print(f'Set tensor speed: {(t2 - t1) * 1000:.2f} ms')
print(f'Inference speed: {(t3 - t2) * 1000 / args.iteration:.2f} ms')
print(f'Get tensor speed: {(time.time() - t3) * 1000:.2f} ms')
print(f'Avg memory usage per inference: {avg_mem_usage_kb:.2f} KB')
