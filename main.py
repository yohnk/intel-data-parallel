import torch
import intel_extension_for_pytorch as ipex
from torch.nn import Sequential, Linear, CrossEntropyLoss, DataParallel
from torch.nn.parallel import DistributedDataParallel
from torch.optim import SGD
from torch import nn


def data_parallel(module, input, device_ids, output_device=None):
    if not device_ids:
        return module(input)

    if output_device is None:
        output_device = device_ids[0]

    replicas = nn.parallel.replicate(module, device_ids)
    inputs = nn.parallel.scatter(input, device_ids)
    replicas = replicas[:len(inputs)]
    outputs = nn.parallel.parallel_apply(replicas, inputs)
    return nn.parallel.gather(outputs, output_device)


def main():
    use_dp_method = True

    linear = Linear(in_features=10, out_features=1, bias=False)
    print(f"Linear Layer dtype: {linear.weight.dtype}")

    model = Sequential(linear).to(torch.device("xpu:0"))
    if not use_dp_method:
        model = DataParallel(model).to(torch.device("xpu:0"))

    criterion = CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=0.001)

    x = torch.rand((100, 10), dtype=torch.float32).to(torch.device("xpu:0"))
    y = torch.rand((100, 1), dtype=torch.float32).to(torch.device("xpu:0"))
    print(f"X dtype: {x.dtype}")
    print(f"Y dtype: {y.dtype}")

    if use_dp_method:
        output = data_parallel(model, x, ["xpu:0", "xpu:1"], "xpu:0")
    else:
        output = model(x)

    loss = criterion(output, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print("Complete")


if __name__ == '__main__':
    main()
