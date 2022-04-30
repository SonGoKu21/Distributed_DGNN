import torch
import torch.nn as nn

class Classifier(torch.nn.Module):
    def __init__(self, in_feature = None, hidden_feature = 100, out_feature = 2):
        super(Classifier, self).__init__()
        self.activation1 = torch.nn.ReLU()
        self.activation2 = torch.nn.Softmax(dim=1)
        self.activation3 = torch.nn.Sigmoid()

        assert in_feature, 'Unspecified number of input features'
        
        num_feats = in_feature

        self.layer_1 = torch.nn.Linear(num_feats, hidden_feature)
        self.layer_2 = torch.nn.Linear(hidden_feature, hidden_feature)
        self.layer_3 = torch.nn.Linear(hidden_feature, out_feature)

        self.norm = nn.BatchNorm1d(hidden_feature)

        # init the layer
        nn.init.xavier_uniform_(self.layer_1.weight)
        nn.init.constant_(self.layer_1.bias, 0)
        nn.init.xavier_uniform_(self.layer_2.weight)
        nn.init.constant_(self.layer_2.bias, 0)
        nn.init.xavier_uniform_(self.layer_3.weight)
        nn.init.constant_(self.layer_3.bias, 0)

    def forward(self, x):
        x = self.layer_1(x)
        x = self.norm(x)
        x = self.activation3(x)

        x = self.layer_2(x)
        x = self.norm(x)
        x = self.activation3(x)

        x = self.layer_3(x)
        x = self.activation2(x)

        return x