import torch

import katransformer
import timm
from fvcore.nn import FlopCountAnalysis, flop_count_str
from fvcore.nn.jit_handles import get_shape
from math import prod


def compute_flops(model_name, input_size, custom_op_name, device='cuda'):
    """
    Computes the FLOPs for a given TIMM model, including a customized operator, using FlopCountAnalysis.

    Args:
        model_name (str): The name of the TIMM model.
        input_size (tuple): The input size (height, width) of the model.
        custom_op_name (str): The name of the customized operator.
        custom_op_flops (int): The number of FLOPs for the customized operator.

    Returns:
        int: The total number of FLOPs.
    """

    # Load the TIMM model
    model = timm.create_model(model_name, pretrained=False)

    # Ensure the model is in evaluation mode to avoid unnecessary computations
    model.eval()
    model.to(device)

    # Create a dummy input tensor with the specified size
    input_tensor = torch.randn(1, 3, *input_size).to(device)

    # Set the FLOPs for the customized operator
    def _custom_op_flops_fn(inputs, outputs):
        # Assuming each operation in custom_op involves 'n' flops per input element
        n = 21  # This should be adjusted based on what the custom operation does
        
        input_shape = get_shape(inputs[0])
        total_elements = prod(input_shape)
        return total_elements * n


    # Use FlopCountAnalysis to compute the FLOPs
    analysis = FlopCountAnalysis(model, 
                                 input_tensor)
    analysis.set_op_handle(custom_op_name, _custom_op_flops_fn)
    
    print(flop_count_str(analysis))
    totoal_flops = analysis.total()
    # print totoal_flops in GigaFlops
    print("Total FLOPs: ", totoal_flops/1e9, "GFLOPs")
    total_params = sum(p.numel() for p in model.parameters())
    print("Total Params: ", total_params/1e6, "M")
    
    return totoal_flops / 1e9


    # return int(flops)

if __name__ == "__main__":
    # model_name = "vit_base_kat_mimetic_patch16_224"  # Replace with your desired model name
    model_name = 'kat_tiny_gelu_patch16_224'  # Replace with your desired model name
    input_size = (224, 224)  # Replace with your desired input size
    custom_op_name = "prim::PythonOp.rational_1dgroup"  # Replace with your customized operator name
    # custom_op_flops = 1000  # Replace with the actual FLOPs of your customized operator

    flops = compute_flops(model_name, input_size, custom_op_name)
    # print(f"FLOPs for {model_name}: {flops}")