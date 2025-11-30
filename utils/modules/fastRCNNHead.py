import torch.nn as nn
import torch.nn.functional as F
class FastRCNNHead(nn.Module):
    """
    Two-FC head for classification + bbox regression
    """
    def __init__(self, in_channels: int, representation_size: int = 1024, num_classes: int = 2):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, representation_size)
        self.fc2 = nn.Linear(representation_size, representation_size)
        self.cls_score = nn.Linear(representation_size, num_classes)  # background + classes
        self.bbox_pred = nn.Linear(representation_size, num_classes * 4)  # 4 deltas per class

        for l in [self.fc1, self.fc2, self.cls_score, self.bbox_pred]:
            nn.init.normal_(l.weight, std=0.01)
            nn.init.constant_(l.bias, 0)

    def forward(self, x):
        # x: (N, C, H, W) pooled -> flatten
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)
        return scores, bbox_deltas