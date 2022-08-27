import argparse
import os
import struct
import sys

import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt

from PIL import Image

import ctypes

INPUT_H = 112
INPUT_W = 112
OUTPUT_SIZE = 512
INPUT_BLOB_NAME = "data"
OUTPUT_BLOB_NAME = "683"


BASE_PATH = "/opt/nvidia/deepstream/deepstream-6.1/sources/dst_python_apps"

engine_path = f"{BASE_PATH}/models/retinaface/retina_r50.engine"
img_path = f'{BASE_PATH}/data/fkhan001.jpg'
ctypes.cdll.LoadLibrary(name=f"{BASE_PATH}/models/retinaface/libRetinafaceDecoder.so")
gLogger = trt.Logger(trt.Logger.INFO)

def doInference(context, host_in, host_out, batchSize):
    engine = context.engine
    assert engine.num_bindings == 2

    devide_in = cuda.mem_alloc(host_in.nbytes)
    devide_out = cuda.mem_alloc(host_out.nbytes)
    bindings = [int(devide_in), int(devide_out)]
    stream = cuda.Stream()

    cuda.memcpy_htod_async(devide_in, host_in, stream)
    context.execute_async(bindings=bindings, stream_handle=stream.handle)
    cuda.memcpy_dtoh_async(host_out, devide_out, stream)
    stream.synchronize()

def _load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image).reshape(
        (im_height, im_width, 3)
        ).astype(np.uint8)

def _load_img(image_path, ih, iw):
    image = Image.open(image_path)
    image_resized = image.resize(
        size=(iw, ih),
        resample=Image.BILINEAR
    )
    img_np = _load_image_into_numpy_array(image_resized)
    # HWC -> CHW
    img_np = img_np.transpose((2, 0, 1))
    # Normalize to [-1.0, 1.0] interval (expected by model)
    # img_np = (2.0 / 255.0) * img_np - 1.0
    # img_np = (2.0 / 255.0) * img_np - 1.0
    img_np = img_np * 255
    return img_np
    # img_np = img_np.ravel()
    # return img_np


runtime = trt.Runtime(gLogger)
assert runtime

with open(engine_path, "rb") as f:
    engine = runtime.deserialize_cuda_engine(f.read())
assert engine

context = engine.create_execution_context()
assert context

data = _load_img(img_path, INPUT_H, INPUT_W)
host_in = cuda.pagelocked_empty(3*INPUT_H * INPUT_W, dtype=np.float32)
np.copyto(host_in, data.ravel())
host_out = cuda.pagelocked_empty(OUTPUT_SIZE, dtype=np.float32)

doInference(context, host_in, host_out, 1)

print(f'Output Shape: {host_out.shape}')
print(f'Output: {host_out}')