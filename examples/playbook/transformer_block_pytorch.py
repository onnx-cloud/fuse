import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleAttn(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)

    def forward(self, x):
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        w = torch.matmul(q, k.transpose(-2, -1)) / (x.size(-1) ** 0.5)
        a = F.softmax(w, dim=-1)
        out = torch.matmul(a, v)
        return self.out(out)


if __name__ == "__main__":
    m = SimpleAttn(dim=64)
    example = torch.randn(2, 8, 64)
    torch.onnx.export(
        m,
        example,
        "onnx/playbook/transformer_block_pytorch.onnx",
        opset_version=14,
        input_names=["x"],
    )
    print("Exported -> onnx/playbook/transformer_block_pytorch.onnx")
