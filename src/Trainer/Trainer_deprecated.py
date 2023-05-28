
# Se está utilizando para las optimizaciones el notebook "TrainerSegmentation"

import IPython
from torchvision import transforms
from ax.service.managed_loop import optimize
from tqdm import tqdm
import torch
import torch.nn as nn
import torch
import torch.utils.data as data
import pandas as pd
import torchvision.transforms as transforms
from collections import defaultdict
import sys
from ax.utils.notebook.plotting import render, init_notebook_plotting
sys.path.append("..")
from Helper.ImagesCustomDataset import ImagesCustomDataset
from Helper.LossFunction import dice_loss, iou_loss, tversky_index, dice, iou
from Net.UNet.UNet import UNet

from ax.plot.contour import plot_contour
init_notebook_plotting()
BATCH_SIZE = 64

url_imagen = "C:/Users/ortiz/OneDrive/Escritorio/TFG DOCUMENTS/Segmentacion_Imagenes/Imagen_Seccionado_128x128"
url_label = "C:/Users/ortiz/OneDrive/Escritorio/TFG DOCUMENTS/Segmentacion_Imagenes/Label_Seccionado_128x128"
url_imagen_test= "C:/Users/ortiz/OneDrive/Escritorio/TFG DOCUMENTS/Segmentacion_Imagenes/Imagenes_test_128x128"
url_label_test = "C:/Users/ortiz/OneDrive/Escritorio/TFG DOCUMENTS/Segmentacion_Imagenes/Label_test_128x128"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
            transforms.ToTensor()
          ])



dataset = ImagesCustomDataset(url_imagen,url_label, transform=transform)
train_dataset,train_dataset_eval = torch.utils.data.random_split(dataset,[11082,5000])

# encapsulate data into dataloader form

dataloaders = {
    'train': data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True),
    'val': data.DataLoader(dataset=train_dataset_eval, batch_size=2*BATCH_SIZE, shuffle=True)
}

def train_segmentation_ax(model, lr:float, momentum:float, alpha:float, beta:float, num_epochs=25, mode=1)->nn.Module:
  optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)  
  for epoch in range(num_epochs):
      for inputs, labels in tqdm(dataloaders['train']):
          model.train()
          inputs = inputs.to(device)
          labels = labels.to(device)
          optimizer.zero_grad()
          metrics = defaultdict(float)
          outputs = model(inputs)
          if mode==1:
             loss = dice_loss(outputs, labels, metrics)
          elif mode==2:
             loss = iou_loss(outputs,labels, metrics)
          else:
             loss= tversky_index(outputs,labels,alpha,beta)

          loss.backward()
          optimizer.step()
  return model

def evaluate_segmentation_ax(model,mode=1) -> float:
  total_loss = .0 
  with torch.no_grad():
    for idx, (inputs, labels) in enumerate(tqdm(dataloaders['val'])):
        model.eval()
        inputs = inputs.to(device)
        labels = labels.to(device)        
        metrics = defaultdict(float)
        outputs = model(inputs)
        if mode==1:
          loss = dice(outputs, labels)
        elif mode==2:
          loss = iou(outputs,labels)
        else:
          loss= (dice(outputs, labels)+iou(outputs, labels))/2
          
        total_loss += loss

  return (total_loss / (idx+1)).item()
dtype = torch.float


#OPTIMIZER

###################MODEL1###################
def train_evaluate(parameterization):
    print(parameterization)
    net =UNet(2,1,parameterization.get('p')) #crear modelo con el parametro p
    # Con cambiar el moded a 2 y a 3 se cambia la loss function
    net = train_segmentation_ax(net.cuda(), parameterization.get('lr'), parameterization.get('momentum'),
                                parameterization.get('alpha'), parameterization.get('beta'),  num_epochs=10, mode=1)
    return evaluate_segmentation_ax(
        model=net.cuda(),
        mode=1
    )
best_parameters, values, experiment, model = optimize(
    parameters=[
        {"name": "p", "type": "range", "bounds": [0.0, 1.0]},
        {"name": "lr", "type": "range", "bounds": [1e-6, 0.4], "log_scale": True},
        {"name": "momentum", "type": "range", "bounds": [0.0, 1.0]},
        {"name": "alpha", "type": "range", "bounds": [0.0, 1.0]},
        {"name": "beta", "type": "range", "bounds": [0.0, 1.0]}
    ],
    evaluation_function=train_evaluate,
    objective_name='accuracy',total_trials=20
)
dict_best_parameters={
    'p':[best_parameters.get('p')],
    'lr':[best_parameters.get('lr')],
    'momentum':[best_parameters.get('momentum')]
}
render(plot_contour(model=model, param_x='lr', param_y='momentum', metric_name='p'))
df = pd.DataFrame(dict_best_parameters)
df.to_csv('best_parameters_model.csv')

