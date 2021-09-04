# flake8: noqa
# Check Pytorch installation
import torch, torchvision


print(torch.__version__, torch.cuda.is_available())
print(torchvision.__version__)


# Check mmcv installation
import mmcv

print(mmcv.__version__)
from mmcv.ops import get_compiling_cuda_version, get_compiler_version

print(mmcv.__version__)
print(get_compiling_cuda_version())
print(get_compiler_version())

# Check MMDetection installation
import mmdet

print(mmdet.__version__)
# Check mmocr installation
import mmocr

print(mmocr.__version__)
