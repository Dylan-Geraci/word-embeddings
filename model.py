# model.py
import torch
import torch.nn as nn

class MLP(nn.Module):
    """
    Flexible MLP for classification.

    Constructor args:
      - input_dim: size of input vector
      - hidden_dims: list of ints (hidden layer neuron counts). Example: [128, 64]
      - num_classes: number of output classes
      - activation: nn.Module class for hidden activations (e.g., nn.ReLU)
      - dropout: float (dropout probability between hidden layers)
      - use_batchnorm: whether to use BatchNorm1d after linear layers
    """
    def __init__(self,
                 input_dim: int,
                 hidden_dims=(128, 64),
                 num_classes=7,
                 activation=nn.ReLU,
                 dropout=0.2,
                 use_batchnorm=True):
        super().__init__()

        layers = []
        prev = input_dim
        for i, h in enumerate(hidden_dims):
            layers.append(nn.Linear(prev, h))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(activation())
            if dropout and dropout > 0.0:
                layers.append(nn.Dropout(p=dropout))
            prev = h

        # Final classification layer -> logits for num_classes
        layers.append(nn.Linear(prev, num_classes))

        # Build sequential module
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # Expect x shape (batch, input_dim)
        return self.net(x)


# ORIGINAL SKELETON CODE
# import torch.nn as nn

# class MLP(nn.Module):
#     def __init__(self):
#         super().__init__()

#         # https://docs.pytorch.org/docs/stable/generated/torch.nn.Sequential.html
#         self.net = nn.Sequential(
#             # Insert layers here
#         )

#     def forward(self, x):
#         return self.net(x)