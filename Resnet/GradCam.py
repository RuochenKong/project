#!/usr/bin/env python
# coding: utf-8

# In[12]:


from resnet1d import Resnet34
import torch
import torch.nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from cusTestData import PPG_test


# In[13]:


class CamExtractor():

    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = []
        self.activation = []
        
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)
        
        
    def save_activation(self, module, input, output):
        self.activation.append(output.cpu().detach())
        
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = [grad_output[0].cpu().detach()] + self.gradients
    

    def __call__(self, x):
        self.gradients = []
        self.activation = []
        return self.model(x)
    


# In[33]:


class GradCam():

    def __init__(self, model, target_layer):
        self.model = model
        self.model.eval()
        self.extractor = CamExtractor(self.model, target_layer)
        self.target_layer = target_layer
        
    def resize(self,cam):
        target_interval = 7200
        num_interval = len(cam) - 1
        ratio = target_interval / num_interval
        
        resized_cam = np.zeros(7201)
        resized_cam[-1] = cam[-1]
        
        camdiff = []
        for i in range(num_interval):
            camdiff.append((cam[i+1] - cam[i])/ratio)
        
        add = 0
        camind = 0
        for i in range(target_interval):
            resized_cam[i] = cam[camind] + add
            
            newcamind = int(i/ratio)
            if newcamind != camind:
                add = 0
                camind = newcamind
            else:
                add += camdiff[camind]
        
        return resized_cam

    def __call__(self, input_sig, target=None):
        
        output = self.extractor(input_sig)
        if target is None:
            target = np.argmax(output.cpu().data.numpy())
        self.model.zero_grad()
        onehot = torch.zeros((1,2)).to('cuda')
        onehot[0][target] = 1
        output.backward(gradient=onehot, retain_graph=True)
        
        activations = self.extractor.activation[-1].data.numpy()[0, :]
        grads = self.extractor.gradients[-1].data.numpy()[0, :]
        weights = np.mean(grads,axis = 1)
        
        cam = activations.T.dot(weights)
        cam -= np.min(cam)
        cam /= np.max(cam) + 1e-10
        return self.resize(cam)
        


# In[34]:


model = Resnet34().to('cuda')
path = "../data/trainedModel.pth"
model.load_state_dict(torch.load(path))
data = PPG_test('ppgqual/mimic/test.txt','mat',10)


# In[35]:


label = open('ppgqual/mimic/predict.txt').read().strip().split(' ')
for i in range(len(label)):
    label[i] = int(label[i])
label = np.array(label)
ind = np.where(label == 1)[0]


# In[39]:


def check(ind):
    sig = torch.Tensor(data[ind]).to('cuda')
    sig = torch.reshape(sig, (1, 1, 7201))
    print(label[ind])
    gradcam = GradCam(model, model.stage2)(sig)
    plt.figure(figsize=(20,2), dpi=80)
    plt.plot(data[ind].flatten())
    plt.show()
    plt.figure(figsize=(20,2), dpi=80)
    plt.plot(gradcam)
    plt.show()
    return gradcam


# In[44]:


gc = check(ind[9])


# In[ ]:




