import torch
import torch.nn as nn

class encoder_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        
        self.conv1 = nn.Conv1d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=1,
            padding='same')
        
        self.relu = nn.ReLU()
        
        self.bn = nn.BatchNorm1d(self.out_channels)
        
        self.convres = nn.Conv1d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=1,
            stride=1,
            padding='same')
        
    def forward(self, x):
        out = x
        out = self.relu(self.bn1(self.conv1(out)))
        res = self.convres(x)
        out = torch.add(out,res)
        return out
        
class decoder_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        
        self.conv1 = nn.Conv1d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=1,
            padding='same')
        
        self.relu = nn.ReLU()
        
        self.bn = nn.BatchNorm1d(self.out_channels)
        
        self.convres = nn.Conv1d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=1,
            stride=1,
            padding='same')
        
    def forward(self, x):
        out = x
        out = self.relu(self.bn1(self.conv1(out)))
        res = self.convres(x)
        out = torch.add(out,res)
        return out
        


class SegMADe(nn.Module):
    
    def __init__ (self):
        super(SegMADe,self).__init__()
            
        self.encoder1 = encoder_block(1, 16, 5)
        self.drop1 = nn.Dropout(p = 0.2)
        self.pool1 = nn.MaxPool1d(2, stride=2)
        self.encoder2 = encoder_block(16, 32, 10)
        self.drop2 = nn.Dropout(p = 0.2)
        self.pool2 = nn.MaxPool1d(2, stride=2)
        self.encoder3 = encoder_block(32, 64, 20)
        self.drop3 = nn.Dropout(p = 0.2)
        self.pool3 = nn.MaxPool1d(2, stride=2)
        self.encoder4 = encoder_block(64, 128, 40)
        self.drop4 = nn.Dropout(p = 0.2)
        self.pool4 = nn.MaxPool1d(2, stride=2)
        self.encoder5 = encoder_block(128, 256, 80)
        self.pool5 = nn.MaxPool1d(2, stride=2)
        
        self.convbottom = nn.Conv1d(
            in_channels = 256,
            out_channels = 256,
            kernel_size = 3,
            stride = 1,
            padding = 'same')
        self.relubottom = nn.ReLU()
        self.bnbottom = nn.BatchNorm1d(256)
        self.dropbottom = nn.Dropout(p = 0.5)
        
        self.upsample1 = nn.Upsample(size=2)
        self.decoder1 = decoder_block(512, 256, 80)
        self.upsample2 = nn.Upsample(size=2)
        self.decoder2 = decoder_block(384, 128, 40)
        self.upsample3 = nn.Upsample(size=2)
        self.decoder3 = decoder_block(192, 64, 20)
        self.upsample4 = nn.Upsample(size=2)
        self.decoder4 = decoder_block(96, 32, 10)
        self.upsample5 = nn.Upsample(size=2)
        self.decoder5 = decoder_block(48, 16, 5)
        
        self.convout = nn.Conv1d(
            in_channels=16,
            out_channels=1,
            kernel_size = 1,
            stride=1,
            padding='same')
        self.sigmoid = nn.Sigmoid()
        
    def forword(self,x):
        
        out = x
        encoding_outs = []
        out = self.drop1(self.encoder1(out))
        encoder_outs.append(out)
        out = self.drop2(self.encoder2(self.pool1(out)))
        encoder_outs.append(out)
        out = self.drop3(self.encoder3(self.pool2(out)))
        encoder_outs.append(out)
        out = self.drop4(self.encoder4(self.pool3(out)))
        encoder_outs.append(out)
        out = self.drop5(self.encoder5(self.pool4(out)))
        encoder_outs.append(out)
        
        out = self.dropbottom(self.bnbottom(self.convbottom(out)))
        
        out = self.upsample1(out)
        out = torch.cat((out, encoding_outs[4]) , 1)
        out = self.upsample2(self.decoder1(out))
        out = torch.cat((out, encoding_outs[3]) , 1)
        out = self.upsample3(self.decoder2(out))
        out = torch.cat((out, encoding_outs[2]) , 1)
        out = self.upsample4(self.decoder3(out))
        out = torch.cat((out, encoding_outs[1]) , 1)
        out = self.upsample5(self.decoder4(out))
        out = torch.cat((out, encoding_outs[0]) , 1)
        out = self.decoder5(out)
        
        out = self.sigmoid(self.convout(out))
        
        return out

