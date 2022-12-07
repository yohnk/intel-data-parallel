import torch
import intel_extension_for_pytorch as ipex
from torch.nn import Sequential, Linear, CrossEntropyLoss, DataParallel
from torch.nn.parallel import DistributedDataParallel
from torch.optim import SGD


def main():
    linear = Linear(in_features=10, out_features=1, bias=False)
    print(f"Linear Layer dtype: {linear.weight.dtype}")

    model = Sequential(linear)
    model = DataParallel(model).to(torch.device("xpu:0"))

    criterion = CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=0.001)

    x = torch.rand((100, 10), dtype=torch.float32).to(torch.device("xpu:0"))
    y = torch.rand((100, 1), dtype=torch.float32).to(torch.device("xpu:0"))
    print(f"X dtype: {x.dtype}")
    print(f"Y dtype: {y.dtype}")

    output = model(x)
    loss = criterion(output, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print("Complete")


if __name__ == '__main__':
    main()
