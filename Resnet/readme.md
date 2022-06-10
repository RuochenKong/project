### Model - Resnet34

`resnet1d.py`\
input shape (N,1,7201)\
output shape (2,)

### DataSet & DataLoader
`cusDataset.py` -- data with binary labels\
`cusTestDataset.py` -- data with different types and lengths (no labels)

### Training
`train_qual.py`\
Loss: Binary Cross Entropy\
Optimizer: Adam

### GradCam
`GradCam.py`
