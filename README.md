#### Install:
```sh
!pip install -q diffusers==0.27.0
!pip install -q transformers==4.36.2
!pip install -q einops ftfy basicsr insightface onnxruntime-gpu accelerate timm apex opencv-python spaces==0.19.4
!pip install --use-pep517 facexlib
!pip install -q --pre xformers==0.0.20
!pip install -q torchvision==0.15.2+cu118 --index-url https://download.pytorch.org/whl/cu118
!pip install -q torch==2.0.1+cu118 --index-url https://download.pytorch.org/whl/cu118
!pip install -q --upgrade numpy
# !pip install hidiffusion

!sed -i 's/from torchvision.transforms.functional_tensor import rgb_to_grayscale/from torchvision.transforms.functional import rgb_to_grayscale/' /opt/conda/lib/python3.10/site-packages/basicsr/data/degradations.py
!wget https://huggingface.co/spaces/yanze/PuLID/resolve/main/eva_clip/bpe_simple_vocab_16e6.txt.gz -O {WORK_DIR}/PuLID_cli/eva_clip/bpe_simple_vocab_16e6.txt.gz
```

#### Run:
```sh
!python -W ignore::UserWarning: app.py --prompt 'portrait, flat papercut style, silhouette, clean cuts, paper, sharp edges, minimalist, color block, man' --neg_prompt 'long neck, wide jaw' --n_samples {n_samples} --save_dir '{OUTPUT_DIR}/PuLID_output' --save_file_name '{filename}' --face_img '{face_img}' --supp_face_img_1 '{supported_face_img_1}' --supp_face_img_2 '{supported_face_img_2}' # --seed 1234

```
