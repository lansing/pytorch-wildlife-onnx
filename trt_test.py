import tensorrt as trt
from cuda import cuda
import numpy as np
import ctypes


import logging

logger = logging.getLogger(__name__)


class HostDeviceMem(object):
    """Simple helper data class that's a little nicer to use than a 2-tuple."""

    def __init__(self, host_mem, device_mem, nbytes, size):
        self.host = host_mem
        err, self.host_dev = cuda.cuMemHostGetDevicePointer(self.host, 0)
        self.device = device_mem
        self.nbytes = nbytes
        self.size = size

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

    def __del__(self):
        cuda.cuMemFreeHost(self.host)
        cuda.cuMemFree(self.device)


class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem, nbytes, size, dtype):
        self.host_ptr = host_mem
        self.device = device_mem
        self.nbytes = nbytes
        self.size = size

        # Create a numpy array pointing to the allocated host memory
        # This allows np.copyto and direct slicing to work
        self.host = np.frombuffer(
            (ctypes.c_byte * nbytes).from_address(int(self.host_ptr)), dtype=dtype
        ).reshape(
            -1
        )  # Flattened view

    def __del__(self):
        cuda.cuMemFreeHost(self.host_ptr)
        cuda.cuMemFree(self.device)


class TrtLogger(trt.ILogger):
    def log(self, severity, msg):
        logger.log(self.getSeverity(severity), msg)

    def getSeverity(self, sev: trt.ILogger.Severity) -> int:
        if sev == trt.ILogger.VERBOSE:
            return logging.DEBUG
        elif sev == trt.ILogger.INFO:
            return logging.INFO
        elif sev == trt.ILogger.WARNING:
            return logging.WARNING
        elif sev == trt.ILogger.ERROR:
            return logging.ERROR
        elif sev == trt.ILogger.INTERNAL_ERROR:
            return logging.CRITICAL
        else:
            return logging.DEBUG


trt_logger = TrtLogger()

trt.init_libnvinfer_plugins(trt_logger, "")

model_path = "/home/max/exported_models/MDV6-yolov10-e_demo_export.engine"

with open(model_path, "rb") as f, trt.Runtime(trt_logger) as runtime:
    engine = runtime.deserialize_cuda_engine(f.read())

binding = engine[0]
binding_dims = engine.get_tensor_shape(binding)
binding_dtype = engine.get_tensor_dtype(binding)
input_shape = (binding_dims[2:], trt.nptype(binding_dtype))
context = engine.create_execution_context()


def _allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    output_idx = 0
    for binding in engine:
        binding_dims = engine.get_tensor_shape(binding)
        size = trt.volume(binding_dims)
        nbytes = size * engine.get_tensor_dtype(binding).itemsize
        err, host_mem = cuda.cuMemHostAlloc(
            nbytes, Flags=cuda.CU_MEMHOSTALLOC_DEVICEMAP
        )
        err, device_mem = cuda.cuMemAlloc(nbytes)
        bindings.append(int(device_mem))
        if binding == "Preprocessoronnx::Identity_0":
            print(f"Input has Shape {binding_dims}")
            inputs.append(HostDeviceMem(host_mem, device_mem, nbytes, size, dtype))
        else:
            # each grid has 3 anchors, each anchor generates a detection
            # output of 7 float32 values
            assert size % 7 == 0, f"output size was {size}"
            print(f"Output has Shape {binding_dims}")
            outputs.append(HostDeviceMem(host_mem, device_mem, nbytes, size, dtype))
            output_idx += 1
    return inputs, outputs, bindings


def _allocate_buffers(engine):
    inputs, outputs, bindings = [], [], []
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        dtype = trt.nptype(engine.get_tensor_dtype(name))
        shape = engine.get_tensor_shape(name)
        size = trt.volume(shape)
        nbytes = size * np.dtype(dtype).itemsize

        err, host_mem = cuda.cuMemHostAlloc(nbytes, cuda.CU_MEMHOSTALLOC_DEVICEMAP)
        err, device_mem = cuda.cuMemAlloc(nbytes)

        mem_obj = HostDeviceMem(host_mem, device_mem, nbytes, size, dtype)

        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            inputs.append(mem_obj)
        else:
            outputs.append(mem_obj)
        bindings.append(int(device_mem))

    return inputs, outputs, bindings


err, cu_ctx = cuda.cuCtxCreate(cuda.CUctx_flags.CU_CTX_MAP_HOST, 0)

err, stream = cuda.cuStreamCreate(0)

# TODOS
# patch is_input to look for my input name

(inputs, outputs, bindings) = _allocate_buffers(engine)


def detect_raw(tensor_input):
    # Input tensor has the shape of the [height, width, 3]
    # Output tensor of float32 of shape [20, 6] where:
    # O - class id
    # 1 - score
    # 2..5 - a value between 0 and 1 of the box: [top, left, bottom, right]

    # normalize
    if input_shape[-1] != trt.int8:
        tensor_input = tensor_input.astype(input_shape[-1])
        tensor_input /= 255.0

    inputs[0].host = np.ascontiguousarray(tensor_input.astype(input_shape[-1]))
    trt_outputs = _do_inference()

    # TODO postprocess


def _do_inference():
    """do_inference (for TensorRT 7.0+)
    This function is generalized for multiple inputs/outputs for full
    dimension networks.
    Inputs and outputs are expected to be lists of HostDeviceMem objects.
    """
    # Push CUDA Context
    cuda.cuCtxPushCurrent(cu_ctx)

    # Transfer input data to the GPU.
    [cuda.cuMemcpyHtoDAsync(inp.device, inp.host, inp.nbytes, stream) for inp in inputs]

    # Run inference.
    print("run inf")
    if not context.execute_v2(bindings):
        logger.warning("Execute returned false")

    # Transfer predictions back from the GPU.
    [
        cuda.cuMemcpyDtoHAsync(out.host, out.device, out.nbytes, stream)
        for out in outputs
    ]

    # Synchronize the stream
    cuda.cuStreamSynchronize(stream)

    # Pop CUDA Context
    cuda.cuCtxPopCurrent()

    # Return only the host outputs.
    return [
        np.array((ctypes.c_float * out.size).from_address(out.host), dtype=np.float32)
        for out in outputs
    ]


def detect_raw(tensor_input):
    # 1. Ensure input is float32 (or whatever your engine expects)
    tensor_input = tensor_input.astype(np.float32) / 255.0

    # 2. Set the input shape in the context (Crucial for V10/Dynamic engines)
    input_name = engine.get_tensor_name(0)  # Usually the first tensor
    context.set_input_shape(input_name, tensor_input.shape)

    # 3. Copy data to the pre-allocated host buffer
    # Note: Use np.copyto to keep the memory address consistent
    np.copyto(inputs[0].host, tensor_input.ravel())

    _do_inference()


def _do_inference():
    cuda.cuCtxPushCurrent(cu_ctx)

    # Transfer to Device
    for inp in inputs:
        cuda.cuMemcpyHtoDAsync(inp.device, inp.host, inp.nbytes, stream)

    # Use the newer address-based execution if on TRT 8.5+
    # Otherwise, ensure bindings list is exactly what the engine expects
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            context.set_tensor_address(name, int(inputs[0].device))
        else:
            context.set_tensor_address(name, int(outputs[0].device))

    # Run
    status = context.execute_async_v3(stream_handle=stream)
    if not status:
        print("Inference failed!")

    # Transfer Back
    for out in outputs:
        cuda.cuMemcpyDtoHAsync(out.host, out.device, out.nbytes, stream)

    cuda.cuStreamSynchronize(stream)
    cuda.cuCtxPopCurrent()


rand_input = np.zeros((1, 3, 640, 640))
detect_raw(rand_input)
