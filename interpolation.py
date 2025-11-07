import torch
import os
import time

LLAMA_3_PATH = "llama3_base_weights.pt" 
LLAMA_2_PATH = "llama2_finetuned_weights.pt" 
OUTPUT_PATH = "interpolated_weights_alpha_0_6.pt"
ALPHA = 0.6

# Interpolation Function

def interpolate_llama_weights(path_a, path_b, alpha, output_path):

    print("Starting weight interpolation process")
    start_time = time.time()
    print(f"Loading state_dict B from: {path_b}")
    state_dict_b = torch.load(path_b, map_location='cpu')

    print(f"Loading state_dict A from: {path_a}")
    state_dict_a = torch.load(path_a, map_location='cpu')
    
    # check if the weight keys match
    if set(state_dict_a.keys()) != set(state_dict_b.keys()):
        print("Error: State dictionary keys do not match. Cannot interpolate.")
        return

    # interpolation
    print("Performing linear interpolation...")
    new_state_dict = {}
    
    # calc coefficients outside the loop
    coeff_a = 1.0 - alpha
    coeff_b = alpha

    for key in state_dict_a.keys():
        tensor_a = state_dict_a[key]
        tensor_b = state_dict_b[key]
        interpolated_tensor = coeff_a * tensor_a + coeff_b * tensor_b
        
        new_state_dict[key] = interpolated_tensor

    del state_dict_a
    del state_dict_b
    
    print(f"Saving new interpolated state_dict to: {output_path}")
    torch.save(new_state_dict, output_path)
    
    end_time = time.time()
    duration = end_time - start_time
    print(f"Interpolation and saving complete in {duration:.2f} seconds.")
    print(f"New file size: {os.path.getsize(output_path) / (1024**3):.2f} GB")
    
    return duration

if __name__ == "__main__":

