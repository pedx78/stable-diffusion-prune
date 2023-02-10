import argparse
import torch
import os
import safetensors.torch

chckpoint_dict_replacements = {
    'cond_stage_model.transformer.embeddings.': 'cond_stage_model.transformer.text_model.embeddings.',
    'cond_stage_model.transformer.encoder.': 'cond_stage_model.transformer.text_model.encoder.',
    'cond_stage_model.transformer.final_layer_norm.': 'cond_stage_model.transformer.text_model.final_layer_norm.',
}

def transform_checkpoint_dict_key(k):
    for text, replacement in chckpoint_dict_replacements.items():
        if k.startswith(text):
            k = replacement + k[len(text):]
    return k

KEY = "cond_stage_model.transformer.text_model.embeddings.position_ids"

def load_model(filepath):

    print(f"loading ... {os.path.basename(filepath)}")
    _, extension = os.path.splitext(filepath)
    if extension.lower() == ".safetensors":
        pl_sd = safetensors.torch.load_file(filepath, device="cpu")
    else:
        pl_sd = torch.load(filepath, map_location="cpu")

    pl_sd = pl_sd.pop("state_dict", pl_sd)
    pl_sd.pop("state_dict", None)

    sd = {}
    for k, v in pl_sd.items():
        new_key = transform_checkpoint_dict_key(k)
        if new_key is not None:
            sd[new_key] = v
    pl_sd.clear()
    pl_sd.update(sd)
    return pl_sd

def dprint(str, flg=False):
    if flg:
        print(str)

if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--out', required=False)
    parser.add_argument('--verbose', default=False, action="store_true")
    args = parser.parse_args()

    if args.model is None:
        exit

    verbose = args.verbose

    sd_model = load_model(args.model)

    if KEY not in sd_model:
        print(f"This model dosen't have 'position_ids' key")
        exit
    tsr_current_key_data = sd_model[KEY]
    dprint("# current data is:", verbose)
    dprint(tsr_current_key_data, verbose)
    dprint(type(tsr_current_key_data), verbose)
    dprint(tsr_current_key_data.size(), verbose)
    dprint(tsr_current_key_data.dtype, verbose)

    dprint("# == if changed to torch.int64 ==", verbose)
    tsr_cast_to_int = tsr_current_key_data.to(torch.int64)
    dprint(type(tsr_cast_to_int), verbose)
    dprint(tsr_cast_to_int, verbose)
    dprint(tsr_cast_to_int.dtype, verbose)

    dprint("# change to:", verbose or args.out)
    tsr_int_index = torch.tensor([[
        0,1,2,3,4,5,6,7,8,9,
        10,11,12,13,14,15,16,17,18,19,
        20,21,22,23,24,25,26,27,28,29,
        30,31,32,33,34,35,36,37,38,39,
        40,41,42,43,44,45,46,47,48,49,
        50,51,52,53,54,55,56,57,58,59,
        60,61,62,63,64,65,66,67,68,69,
        70,71,72,73,74,75,76
        ]],
        dtype=torch.int64
        )

    dprint(tsr_int_index, verbose or args.out)
    dprint(type(tsr_int_index), verbose or args.out)
    dprint(tsr_int_index.dtype, verbose or args.out)

    tsr_eq = torch.eq(tsr_cast_to_int, tsr_int_index)
    print("#")
    print(tsr_eq)

    _all_number = []
    _false_number = []
    _missing_number = []
    for k in range(77):
        if not tsr_eq[0, k].item():
            _false_number.append(k)
        _all_number.append(tsr_cast_to_int[0, k].item())
    for i in range(77):
        if i not in _all_number:
            _missing_number.append(i)
    if len(_false_number) > 0:
        print(f"corrupt token indexes : {_false_number}")
    if len(_missing_number) > 0:
        print(f"missing token numbers : {_missing_number}")

    if args.out is not None and args.out != "":

        sd_model[KEY] = tsr_int_index

        output_file = args.out
        if not os.path.exists(output_file):
            print("Saving...")

            theta_0_sd = {"state_dict": sd_model}
            _, extension = os.path.splitext(output_file)
            if extension.lower() == ".safetensors":
                safetensors.torch.save_file(sd_model, output_file, metadata={"format": "pt"})
            else:
                output_file = output_file + ".ckpt" if ".ckpt" not in output_file else output_file
                torch.save(theta_0_sd, output_file)
    del sd_model
