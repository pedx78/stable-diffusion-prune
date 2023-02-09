import torch
import safetensors.torch
import os.path
import sys


if __name__ == "__main__":
    checkpoint_file = sys.argv[1]
    if len(sys.argv) > 2:
        device = str(sys.argv[2])
    else:
        device = 'cpu'

    ext = os.path.splitext(checkpoint_file)

    if ext[-1] == ".safetensors":
        checkpoint = safetensors.torch.load_file(checkpoint_file, device=device)
    elif ext[-1] == ".ckpt":
        checkpoint = torch.load(checkpoint_file,  map_location=torch.device(device))
    else:
        raise Exception("Input file is not a valid checkpoint file")

    if 'state_dict' in checkpoint:
        checkpoint = checkpoint['state_dict']

    print(checkpoint['cond_stage_model.transformer.text_model.embeddings.position_ids'])
