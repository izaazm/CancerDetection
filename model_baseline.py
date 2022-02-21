import torch
import torch.nn as nn
import torch.nn.functional as F


class cancer_classifier(nn.Module):
    def __init__(self):
        super(cancer_classifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 56, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(56, 256, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(256, 640, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(640, 640, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(640, 512, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(512, 420, kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(420, 320, kernel_size=3, stride=1, padding=1)
        self.conv8 = nn.Conv2d(320, 320, kernel_size=3, stride=1, padding=1)
        self.dropout025 = nn.Dropout2d(0.25)
        self.dropout050 = nn.Dropout2d(0.50)
        self.linear1 = nn.Linear(320*4*4, 1024)
        self.linear2 = nn.Linear(1024, 1024)
        self.linear3 = nn.Linear(1024, 2)
    
    def forward(self, x):
        out = F.max_pool2d(F.relu(self.conv1(x)), kernel_size=2, stride=2)  #cnn
        out = F.max_pool2d(F.relu(self.conv2(out)), kernel_size=2, stride=2)
        out = self.dropout025(out)                                          #dropout
        out = F.max_pool2d(F.relu(self.conv3(out)), kernel_size=2, stride=2)
        out = F.max_pool2d(F.relu(self.conv4(out)), kernel_size=2, stride=2)
        out = self.dropout025(out)                                          #dropout
        out = F.max_pool2d(F.relu(self.conv5(out)), kernel_size=2, stride=2)
        out = F.max_pool2d(F.relu(self.conv6(out)), kernel_size=2, stride=2)
        out = self.dropout025(out)                                          #dropout
        out = F.relu(self.conv7(out))
        out = F.relu(self.conv8(out))
        
        out = out.reshape([-1, 320*4*4])                                    #flatten
        out = self.dropout050(out)                                          #dropout
        out = F.relu(self.linear1(out))                                     #dnn
        out = F.relu(self.linear2(out))
        out = self.linear3(out)
        out = F.log_softmax(out, dim = 1)                                   #softmax
        return out


if __name__ == "__main__":
    batch = torch.zeros(64, 3, 256, 256)
    model = cancer_classifier()
    output = model(batch)
    print(output.size())