import torch


def onnx_export(model, data_shape, data_dtype, device, output_file):
    print("\nExporting...\n")
    batch_size = 1
    _, data_channel, data_height, data_width = data_shape
    rand_input = torch.randn(batch_size,
                             data_channel,
                             data_height,
                             data_width,
                             requires_grad=True).to(dtype=data_dtype,
                                                    device=device)
    torch.onnx.export(model.cpu(),
                      rand_input.cpu(),
                      output_file,
                      export_params=True,
                      opset_version=11,
                      do_constant_folding=True,
                      input_names=['input'],
                      output_names=['output'],
                      dynamic_axes={
                          'input': {
                              0: 'batch_size'
                          },
                          'output': {
                              0: 'batch_size'
                          }
                      })
    model = model.to(device)
    print("\nModel exported!\n")
