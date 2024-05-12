import spaces
import numpy as np
import torch
import argparse
from PIL import Image
import random

from pulid import attention_processor as attention
from pulid.pipeline import PuLIDPipeline
from pulid.utils import resize_numpy_image_long, seed_everything

torch.set_grad_enabled(False)

pipeline = PuLIDPipeline()

DEFAULT_NEGATIVE_PROMPT = (
    'flaws in the eyes, flaws in the face, flaws, lowres, non-HDRi, low quality, worst quality,'
    'artifacts noise, text, watermark, glitch, deformed, mutated, ugly, disfigured, hands, '
    'low resolution, partially rendered objects, deformed or partially rendered eyes, '
    'deformed, deformed eyeballs, cross-eyed, blurry'
)

# # realistic
# --prompt 'portrait, cinematic, wolf ears, white hair' --mode 'fidelity'
# # painting style
# --prompt 'portrait, impressionist painting, loose brushwork, vibrant color, light and shadow play' --mode 'fidelity'
# # papercut style
# --prompt 'portrait, flat papercut style, silhouette, clean cuts, paper, sharp edges, minimalist, color block, man' --mode 'fidelity'
# # 3d style
# --prompt 'woman, cartoon, solo, Popmart Blind Box, Super Mario, 3d' --mode 'fidelity'
# # anime style
# --prompt 'portrait, the legend of zelda, anime' --mode 'fidelity'
# # id mix
# --prompt 'portrait, superman' --mode 'fidelity' --id_mix

parser = argparse.ArgumentParser()
parser.add_argument("--save_dir", type=str)
parser.add_argument("--save_file_name", type=str)
# ID image (main)
parser.add_argument("--face_img", type=str, default='')
# Additional ID image (auxiliary)
parser.add_argument("--supp_face_img_1", type=str, default='')
# Additional ID image (auxiliary)
parser.add_argument("--supp_face_img_2", type=str, default='')
# Additional ID image (auxiliary)
parser.add_argument("--supp_face_img_3", type=str, default='')
# Prompt
parser.add_argument("--prompt", type=str, default='portrait, cinematic, wolf ears, white hair')
# Negative Prompt
parser.add_argument("--neg_prompt", type=str, default=DEFAULT_NEGATIVE_PROMPT)
# CFG, recommend value range [3, 7]
parser.add_argument("--cfg_scale", type=float, default=7)
# Num samples [1, 8]
parser.add_argument("--n_samples", type=int, default=1)
parser.add_argument("--seed", type=int, default=-1)
parser.add_argument("--steps", type=int, default=40)
# Height
parser.add_argument("--h", type=int, default=1216)
# Width
parser.add_argument("--w", type=int, default=832)
# ID scale [0.00, 5.00]
parser.add_argument("--id_scale", type=float, default=0.80)
# ['fidelity', 'extremely style']
parser.add_argument("--mode", type=str, default='fidelity', choices=['fidelity', 'extremely style'])
# ID Mix (if you want to mix two ID image, please turn this on, otherwise, turn this off)
parser.add_argument("--id_mix", action='store_true', default=False)
args = parser.parse_args()
print(args)


@spaces.GPU
def run(
    face_image,
    supp_image1,
    supp_image2,
    supp_image3,
    prompt,
    neg_prompt,
    scale,
    n_samples,
    seed,
    steps,
    H,
    W,
    id_scale,
    mode,
    id_mix
):
    id_image = np.array(Image.open(face_image).convert("RGB"))
    supp_images = [np.array(Image.open(name).convert("RGB")) for name in [supp_image1, supp_image2, supp_image3] if name != '']

    pipeline.debug_img_list = []
    if mode == 'fidelity':
        attention.NUM_ZERO = 8
        attention.ORTHO = False
        attention.ORTHO_v2 = True
    elif mode == 'extremely style':
        attention.NUM_ZERO = 16
        attention.ORTHO = True
        attention.ORTHO_v2 = False
    else:
        raise ValueError

    if id_image is not None:
        id_image = resize_numpy_image_long(id_image, 1024)
        id_embeddings = pipeline.get_id_embedding(id_image)
        for supp_id_image in supp_images:
            supp_id_image = resize_numpy_image_long(Image.open(supp_id_image).convert("RGB"), 1024)
            supp_id_embeddings = pipeline.get_id_embedding(supp_id_image)
            id_embeddings = torch.cat(
                (id_embeddings, supp_id_embeddings if id_mix else supp_id_embeddings[:, :5]), dim=1
            )
    else:
        id_embeddings = None

    if seed == -1:
        seed = random.randint(0, np.iinfo(np.int32).max)
    seed_everything(seed)
    ims = []
    for _ in range(n_samples):
        img = pipeline.inference(prompt, (1, H, W), neg_prompt, id_embeddings, id_scale, scale, steps)[0]
        ims.append(np.array(img))

    return ims


ims = run(
    args.face_img,
    args.supp_face_img_1,
    args.supp_face_img_2,
    args.supp_face_img_3,
    args.prompt,
    args.neg_prompt,
    args.cfg_scale,
    args.n_samples,
    args.seed,
    args.steps,
    args.h,
    args.w,
    args.id_scale,
    args.mode,
    args.id_mix
)

for i, img in enumerate(ims):
    Image.fromarray(img).save(f'{args.save_dir}/{args.save_file_name}_{i}.png')
