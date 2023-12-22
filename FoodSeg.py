from mmseg.apis import inference_segmentor, init_segmentor
import torch
import numpy as np

def FoodSeg(image, 
            config_file : str = 'checkpoints/CCNet_ReLeM/ccnet_r50-d8_512x1024_80k.py',
            checkpoint_file : str = 'checkpoints/CCNet_ReLeM/iter_80000.pth'):

    # build the model from a config file and a checkpoint file
    model = init_segmentor(config_file, checkpoint_file, device='cuda:0')
    result = inference_segmentor(model, image)[0]
    del model
    torch.cuda.empty_cache()

    image = np.uint8(result * 255)

    return image