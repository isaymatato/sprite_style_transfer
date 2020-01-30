import argparse
import os
import torch
import tqdm

from PIL import Image
from PIL.ImageOps import invert
from models import TransformerNet
from torch.autograd import Variable
from torchvision.utils import save_image
import torchvision.transforms.functional as F
from utils import *

def style_image(image_path,
                model_path):


    image = Image.open(image_path)
    width, height = image.size
    alpha = image.convert('RGBA').split()[-1]
    # @TODO - import the mean color...
    mean_color = Image.new("RGB", image.size, (124, 116, 103))

    rgb_image = image.convert('RGB')
    rgb_image.paste(mean_color, mask=invert(alpha))

    cuda_available = torch.cuda and torch.cuda.is_available()
    device = torch.device("cuda" if cuda_available else "cpu")

    model = torch.load(model_path, map_location=device)

    image_filename = os.path.basename(image_path)
    model_filename = os.path.basename(model_path)
    model_name = os.path.splitext(model_filename)[0]

    os.makedirs(f"images/outputs/{model_name}", exist_ok=True)

    transform = style_transform()

    # Define model and load model checkpoint
    transformer = TransformerNet().to(device)
    transformer.load_state_dict(model)
    transformer.eval()

    # Prepare input
    image_tensor = Variable(transform(rgb_image)).to(device)
    image_tensor = image_tensor.unsqueeze(0)

    # Stylize image
    with torch.no_grad():
        output_tensor = depro(transformer(image_tensor))
        # alt_image = denormalize(transformer(image_tensor)).cpu()

    stylized_image = F.to_pil_image(output_tensor) \
        .convert('RGBA') \
        .crop((0, 0, width, height))

    stylized_image.putalpha(alpha)
    # Save image
    stylized_image.save(f"images/outputs/{model_name}/{image_filename}", 'PNG')
    # save_image(alt_image, f"images/outputs/{model_name}/{image_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, required=True, help="Image path")
    parser.add_argument("--model_path", default="data/models/megaman8.pth", type=str, help="Model path")
    args = parser.parse_args()

    style_image(**vars(args))

