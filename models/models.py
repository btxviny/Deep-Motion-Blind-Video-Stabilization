import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        ip_ = x
        return torch.add(self.conv_block(x), ip_)
        
        
class ENet(nn.Module):
    def __init__(self, in_channels=15, out_channels=3, residual_blocks=64):
        super(ENet, self).__init__()
        self.merge = torch.cat
        self.add = torch.add
        self.conv1 = nn.Sequential(
                        nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1), 
                        nn.ReLU())
        self.conv2 = nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), 
                nn.ReLU())
        #Residual blocks
        residuals = []
        for _ in range(residual_blocks):
            residuals.append(ResidualBlock(64))
        self.residuals = nn.Sequential(*residuals)
        
        #nearest neighbor upsample 
        self.seq = nn.Sequential(
                nn.Conv2d(64, 64, 3, 1, 1),
                nn.ReLU(),
                nn.Conv2d(64, 64, 3, 1, 1),
                nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(64, in_channels, kernel_size=3, stride=1, padding=1), nn.ReLU())
        self.conv4 = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1))
        total_trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total Trainable Parameters: {total_trainable_params}")

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.residuals(out)
        out = self.conv3(out)
        out = self.conv4(out) 

        return out


class Encoder(nn.Module):
    def __init__(self,in_features,out_features,stride = 2):
        super(Encoder,self).__init__()
        self.conv = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features,out_features,kernel_size=3, stride = stride, padding = 0),
            nn.LeakyReLU(0.2)
        )
        self.gate = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, out_features, kernel_size=3, stride= stride, padding=0),
            nn.Sigmoid()
        )
    def forward(self, x):
        x1 = self.conv(x)
        x2 = self.gate(x)
        return x1 * x2
    
class Decoder(nn.Module):
    def __init__(self,in_features,out_features, tanh = False):
        super(Decoder,self).__init__()
        self.conv = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features,in_features,kernel_size=3, stride = 1, padding = 0),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, out_features, kernel_size=3, stride=1, padding=0)
        )

        self.gate = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=0),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, out_features, kernel_size=3, stride=1, padding=0),
            nn.Sigmoid()
        )
        if tanh:
            self.activation = nn.Tanh()
        else:
            self.activation = nn.ReLU()
    def forward(self, x):
        x1 = self.conv(x)
        x1 = self.activation(x1)
        x2 = self.gate(x)
        return x1 * x2



class UNet(nn.Module):
    def __init__(self):
        super(UNet,self).__init__()
        self.enc0 = Encoder(15,64,stride = 1) #out: 256x256
        self.enc1 = Encoder(64,128)#out: 128x128
        self.enc2 = Encoder(128,256)#out: 64x64
        self.enc3 = Encoder(256,512)#out: 32x32
        self.enc4 = Encoder(512,512)#out: 16x16

        self.dec0 = Decoder(512,512) #32x32
        self.dec1 = Decoder(1024,256) #64x64
        self.dec2 = Decoder(512,128) #128x128
        self.dec3 = Decoder(256,64) #256x256
        self.dec4 = Decoder(64,3)
        
        self.initialize_weights()
        total_trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total UNet Parameters: {total_trainable_params}")
    
    def initialize_weights(self):
        # Initialize the weights for layers with intermediate gated convolutions using Xavier
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                # Xavier initialization for convolutions
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self,x):
        s0 = self.enc0(x)
        s1 = self.enc1(s0)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        s4 = self.enc4(s3)

        s5 = self.dec0(s4) #32x32
        s5 = F.interpolate(s5,scale_factor=2,mode='bilinear',align_corners=True)#64x64
        s6 = self.dec1(torch.cat([s5,s3],dim =1))
        s6 = F.interpolate(s6,scale_factor=2,mode='bilinear',align_corners=True)#128x128
        s7 = self.dec2(torch.cat([s6,s2],dim = 1))
        s7 = F.interpolate(s7,scale_factor=2,mode='bilinear',align_corners=True)#256x256
        s8 = self.dec3(torch.cat([s7,s1],dim=1))  
        s8 = F.interpolate(s8,scale_factor=2,mode='bilinear',align_corners=True)
        out = self.dec4(s8) 
        
        return out
    

class Critic(nn.Module):
    # Initializers
    def __init__(self, d=64):
        super(Critic, self).__init__()

        self.conv1 = nn.Conv2d(3, d, kernel_size=4, stride=2, padding=1)#128
        self.relu1 = nn.LeakyReLU(0.2)
        self.norm1 = nn.InstanceNorm2d(d)

        self.conv2 = nn.Conv2d(d, d * 2, kernel_size=4, stride=2, padding=1)#64
        self.relu2 = nn.LeakyReLU(0.2)
        self.norm2 = nn.InstanceNorm2d(d * 2)

        self.conv3 = nn.Conv2d(d * 2, d * 4, kernel_size=4, stride=2, padding=1)#32
        self.relu3 = nn.LeakyReLU(0.2)
        self.norm3 = nn.InstanceNorm2d(d * 4)

        self.conv4 = nn.Conv2d(d * 4, d * 8, kernel_size=4, stride=2, padding=1)#16
        self.relu4 = nn.LeakyReLU(0.2)
        self.norm4 = nn.InstanceNorm2d(d * 8)
        
        self.conv5 = nn.Conv2d(d * 8, d * 8, kernel_size=4, stride=2, padding=1)#8
        self.relu5 = nn.LeakyReLU(0.2)


        self.flatten = nn.Flatten()
        self.linear = nn.Linear(d * 8 * 8 * 8, 1)  # Adjust the linear layer input size accordingly

        # Weight initialization
        self._initialize_weights()

    # Weight initialization
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

    # Forward method
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.norm1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.norm2(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.norm3(x)

        x = self.conv4(x)
        x = self.relu4(x)
        x = self.norm4(x)

        x = self.conv5(x)
        x = self.relu5(x)

        x = self.flatten(x)
        x = self.linear(x)

        return x
    
class TempDiscriminator3D(nn.Module):
    def __init__(self, d=32,):
        super(TempDiscriminator3D, self).__init__()

        self.conv1 = nn.Conv3d(3, d, kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=(1, 1, 1))
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)
        self.dropout = nn.Dropout3d(0.4)

        self.conv2 = nn.Conv3d(d, d * 2, kernel_size=(3, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1))
        self.relu2 = nn.LeakyReLU(0.2, inplace=True)
        

        self.conv3 = nn.Conv3d(d * 2, d * 4, kernel_size=(3, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1))
        self.relu3 = nn.LeakyReLU(0.2, inplace=True)

        self.conv4 = nn.Conv3d(d * 4, d * 8, kernel_size=(3, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1))
        self.relu4 = nn.LeakyReLU(0.2, inplace=True)

        self.conv5 = nn.Conv3d(d * 8, d * 16, kernel_size=(3, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1))
        self.relu5 = nn.LeakyReLU(0.2, inplace=True)

        self.flatten = nn.Flatten()
        self.linear = nn.Linear(d * 256, 1)
        self.sigmoid = nn.Sigmoid()

        self._initialize_weights()
        total_trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total TempDiscriminator3D Trainable Parameters: {total_trainable_params}")

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.dropout(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.dropout(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.dropout(x)

        x = self.conv4(x)
        x = self.relu4(x)
        x = self.dropout(x)

        x = self.conv5(x)
        x = self.relu5(x)
        x = self.dropout(x)

        x = self.flatten(x)
        x = self.linear(x)
        x = self.sigmoid(x)
        return x