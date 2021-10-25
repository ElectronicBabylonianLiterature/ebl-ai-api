# Check Pytorch installation
import torch
import torchvision
import mmcv
from mmcv.ops import get_compiling_cuda_version, get_compiler_version
import mmdet
import mmocr

print("Torch version: ", torch.__version__)
print("Is Cude Available: ", torch.cuda.is_available())
print("Torchvision version: ", torchvision.__version__)  # pyre-ignore[16]

print("MMCV version: ", mmcv.__version__)


print("MMCV Cuda compiling version: ", get_compiling_cuda_version())
print("Compiler version: ", get_compiler_version())


print("MMDET version: ", mmdet.__version__)

print("MMOCR version: ", mmocr.__version__)
