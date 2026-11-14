#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
conda env create -f environment_dcgan.yml
conda run -n dcgan python -c 'import torch, torchvision; from torchvision.ops import nms; assert torch.version.cuda == "12.1"; assert torch.cuda.is_available(); boxes = torch.tensor([[0., 0., 1., 1.]], device="cuda"); scores = torch.tensor([1.], device="cuda"); assert nms(boxes, scores, 0.5).tolist() == [0]; print(torch.__version__, torchvision.__version__, "CUDA GPUs:", torch.cuda.device_count())'
