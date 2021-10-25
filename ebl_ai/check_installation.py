# pyre-ignore-all-errors
# Check Pytorch installation
import torch
import torchvision

print("Torch version: ", torch.__version__)
print("Is Cude Available: ", torch.cuda.is_available())
print("Torchvision version: ", torchvision.__version__)


# Check mmcv installation
import mmcv  # noqa: E402

print("MMCV version: ", mmcv.__version__)
from mmcv.ops import get_compiling_cuda_version, get_compiler_version  # noqa: E402

print("MMCV Cuda compiling version: ", get_compiling_cuda_version())
print("Compiler version: ", get_compiler_version())

# Check MMDetection installation
import mmdet  # noqa: E402

print("MMDET version: ", mmdet.__version__)
# Check mmocr installation
import mmocr  # noqa: E402

print("MMOCR version: ", mmocr.__version__)
