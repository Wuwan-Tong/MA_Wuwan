import warnings

import matplotlib.pyplot as plt
import torch
import torch.functional as F
import numpy as np
import requests
import torchvision
import open_clip
from PIL import Image
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from pytorch_grad_cam import GradCAM
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
import logging
from torch.backends import cudnn


cudnn.benchmark = False
img_name_list = ['C:/Users/INDA_HIWI/Desktop/dogball.jpg']

import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2




# Hook to capture the gradients and the activations
gradients = []
activations = []

def save_gradient(grad):
    gradients.append(grad)

def forward_hook(module, input, output):
    activations.append(output)
    output.register_hook(save_gradient)

# Register hook on the target layer
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k', precision='fp32')
model.train(True)
tokenizer=open_clip.get_tokenizer('ViT-B-32')
target_layer =model.visual.ln_post
hook = target_layer.register_forward_hook(forward_hook)

# Forward pass
caption = ['ball', 'dog']
text = tokenizer(caption)
img_path = 'C:/Users/INDA_HIWI/Desktop/dogball.jpg'
input_tensor = preprocess(Image.open(img_path)).unsqueeze(0)
image_features, text_features, logit_scale = model(input_tensor, text)
logits_per_image = logit_scale * image_features @ text_features.T
probs = logits_per_image.softmax(dim=-1).cpu().detach().numpy()

# Backward pass
target_class = probs.argmax()
model.zero_grad()
logits_per_image[:, target_class].backward()

# Get the gradients and activations
gradients = gradients[0].cpu().data.numpy()
activations = activations[0].cpu().data.numpy()

# Compute weights
weights = np.mean(gradients, axis=(0, 2))

# Compute Grad-CAM
# grad_cam=np.mean(gradients + activations, axis=0)
grad_cam = np.zeros(activations.shape[2:], dtype=np.float32)
activations = activations[0]
for i, w in enumerate(weights):
    grad_cam += w * activations[i]

grad_cam = np.maximum(grad_cam, 0)
grad_cam = cv2.resize(grad_cam, (224, 224))
grad_cam = grad_cam - grad_cam.min()
grad_cam = grad_cam / grad_cam.max()

# Visualize the heatmap
image = preprocess(Image.open(img_path)).permute(1, 2, 0)
image = np.array(image)

heatmap = cv2.applyColorMap(np.uint8(255 * grad_cam), cv2.COLORMAP_JET)
heatmap = np.float32(heatmap) / 255
overlay = heatmap + np.float32(image) / 255
overlay = overlay / overlay.max()

plt.imshow(overlay)
plt.show()

# Clean up hook
hook.remove()

# for img_name in img_name_list:
#
#     model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k', precision='fp32')
#     model.eval()
#     tokenizer=open_clip.get_tokenizer('ViT-B-32')
#     cam = GradCAM(model, target_layers=[model.visual.ln_post])
#
#     img_path = 'C:/Users/INDA_HIWI/Desktop/dogball.jpg'
#     input_tensor = preprocess(Image.open(img_path)).unsqueeze(0)
#     target_category = None
#     caption=['ball', 'dog']
#     text = tokenizer(caption)
#     grayscale_cam = cam(input_tensor=input_tensor, text=text, targets=model)
#     grayscale_cam = grayscale_cam[0, :]
#     ori_image=np.array(input_tensor.squeeze(0).permute(1, 2, 0))
#     visualization = show_cam_on_image(ori_image/np.max(ori_image), grayscale_cam)
#     Image.fromarray(visualization).show()


    # image_tensor = preprocess(Image.open(img_path))
    # caption = '' # todo
    # model = model.cuda()
    # image_tensor = image_tensor.cuda()
    # output = model(image_tensor, caption)
    #
    # normalized_masks = torch.nn.functional.softmax(output, dim=1).cpu()
    # sem_classes = ['backgeound', 'aaa']#todo
    # sem_class_to_idx = {cls:idx for (idx, cls) in enumerate(sem_classes)}
    #
    # plaque_category = sem_class_to_idx['aaa']
    # plaque_mask = normalized_masks[0, :, :, :].argmax(axis=0).detach().cpu().numpy
    # plaque_mask_uint8 = 255 * np.uint8(plaque_mask == plaque_category)
    # plaque_mask_float = np.float32(plaque_mask == plaque_category)
    #
    # both_images = np.hstack((np.array(Image.open(img_path)), np.repeat(plaque_mask_uint8[:, :, None], 3, axis=-1)))
    # Image.fromarray(both_images)
    #
    # target_layers = [model.visual.ln_post]
    # targets =
