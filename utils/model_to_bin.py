import struct
import torch
from transformers import ViTForImageClassification

def save_vit_checkpoint(model, config, filepath):
    """
    Save ViT model weights and configuration into a .bin file.

    Args:
        model (torch.nn.Module): The Vision Transformer (ViT) model.
        config: The configuration of the model.
        filepath (str): Path to save the .bin file.
    """
    # Open file for writing in binary mode
    with open(filepath, "wb") as f:
        # Write magic number and version
        f.write(struct.pack("I", 20240326))  # Magic number
        f.write(struct.pack("I", 3))  # Version number (FP32)

        # Write model hyperparameters
        f.write(struct.pack("I", config.image_size))          # Image size
        f.write(struct.pack("I", config.hidden_size))         # Hidden size (C)
        f.write(struct.pack("I", config.num_hidden_layers))   # Number of layers (L)
        f.write(struct.pack("I", config.num_attention_heads)) # Attention heads
        f.write(struct.pack("I", config.intermediate_size))   # Feedforward size

        # Pad the header to 256 integers
        for _ in range(256 - 7):
            f.write(struct.pack("I", 0))

        total_bytes = 0
        state_dict = model.state_dict()

        # Define the parameter order
        param_order = []

        # Embedding parameters
        param_order.extend([
            'vit.embeddings.cls_token',
            'vit.embeddings.position_embeddings',
            'vit.embeddings.patch_embeddings.projection.weight',
            'vit.embeddings.patch_embeddings.projection.bias',
        ])

        # Loop over layers for specific parameters
        L = config.num_hidden_layers
        for i in range(L):
            # ln1w
            param_order.append(f'vit.encoder.layer.{i}.layernorm_before.weight')
        for i in range(L):
            # ln1b
            param_order.append(f'vit.encoder.layer.{i}.layernorm_before.bias')
        for i in range(L):
            # QKV weights
            param_order.append(f'vit.encoder.layer.{i}.attention.attention.query.weight')
            param_order.append(f'vit.encoder.layer.{i}.attention.attention.key.weight')
            param_order.append(f'vit.encoder.layer.{i}.attention.attention.value.weight')

        for i in range(L):
            # QKV biases
            param_order.append(f'vit.encoder.layer.{i}.attention.attention.query.bias')
            param_order.append(f'vit.encoder.layer.{i}.attention.attention.key.bias')
            param_order.append(f'vit.encoder.layer.{i}.attention.attention.value.bias')

        for i in range(L):
            # Attention output dense weights
            param_order.append(f'vit.encoder.layer.{i}.attention.output.dense.weight')
        for i in range(L):
            # Attention output dense biases
            param_order.append(f'vit.encoder.layer.{i}.attention.output.dense.bias')
        for i in range(L):
            # ln2w
            param_order.append(f'vit.encoder.layer.{i}.layernorm_after.weight')
        for i in range(L):
            # ln2b
            param_order.append(f'vit.encoder.layer.{i}.layernorm_after.bias')

        for i in range(L):
            # Intermediate dense weights
            param_order.append(f'vit.encoder.layer.{i}.intermediate.dense.weight')
        for i in range(L):
            # Intermediate dense biases
            param_order.append(f'vit.encoder.layer.{i}.intermediate.dense.bias')

        for i in range(L):
            # Output dense weights
            param_order.append(f'vit.encoder.layer.{i}.output.dense.weight')
        for i in range(L):
            # Output dense biases
            param_order.append(f'vit.encoder.layer.{i}.output.dense.bias')


        # Final LayerNorm and classifier parameters
        param_order.extend([
            'vit.layernorm.weight',
            'vit.layernorm.bias',
            'classifier.weight',
            'classifier.bias',
        ])

        # Write parameters to file
        for name in param_order:
            tensor = state_dict[name].to(torch.float32)
            f.write(tensor.numpy().tobytes())
            print(f"Saving tensor: {name}, shape: {tensor.shape}")
            total_bytes += tensor.numel() * 4

        print(f"Total bytes written: {total_bytes}")

if __name__ == "__main__":
    # Load the pretrained ViT model
    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224", cache_dir="./.cache"
    )

    # Access model configurations
    config = model.config
    print("Model configuration:", config)

    # Save the model weights to a .bin file
    save_vit_checkpoint(model, config, "vit_base_patch16_224.bin")
    print("Model saved as 'vit_base_patch16_224.bin'")