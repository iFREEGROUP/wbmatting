"""
去除白色背景
Example:
    python wbmatting.py \
        --model torchscript.pth  \
        --src testdata/6900068804425.jpg \
        --device [cpu|cuda] \
        --output-dir output
"""
import torch
import argparse
from torchvision import transforms as T
from torchvision.transforms.functional import to_pil_image
from PIL import Image
from pathlib import Path


def run_model(model, p: Path, output_dir):
    src_image = Image.open(str(p)).convert("RGB")
    src = transforms(src_image).unsqueeze(0).to(precision).to(device)
    print(src.shape)

    bgr_image = Image.open(args.bgr).convert("RGB") if args.bgr is not None else Image.new(
        'RGB', (src_image.width, src_image.height), '#FFFFFF')
    bgr = transforms(bgr_image).unsqueeze(0).to(precision).to(device)
    pha, fgr = model(src, bgr)[:2]
    com = torch.cat([fgr * pha.ne(0), pha], dim=1)

    img = to_pil_image(com[0].cpu()).resize((src_image.width, src_image.height))
    name = p.name.replace(".jpg", '')
    img.save("{}/{}_com.png".format(output_dir, name))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='White Background Images Matting')

    parser.add_argument('--src', type=Path, required=True)
    parser.add_argument('--bgr', type=str, required=False)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--output-dir', type=str, default='.')

    args = parser.parse_args()

    device = torch.device(args.device)
    precision = torch.float32

    model = torch.jit.load(args.model)
    model.backbone_scale = 0.25
    model.refine_mode = 'sampling'
    model.refine_sample_pixels = 80_000

    model = model.to(device)
    transforms = T.Compose([
        T.Resize((800,800)),
        T.ToTensor(),
    ])

    src_path: Path = args.src

    if src_path.is_dir():
        for p in src_path.iterdir():
            run_model(model=model, p=p, output_dir=args.output_dir)
    else:
        run_model(model=model, p=src_path, output_dir=args.output_dir)
