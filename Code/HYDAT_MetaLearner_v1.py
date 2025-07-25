# -*- coding: utf-8 -*-
"""
This is code for developing a probabilistic meta learner for fusing TimesFM and
fine-tuned-regional LSTM model outputs

In this version, Gaussian output layer is used

This code can be used for FT for both streamflow and water level prediction. This
can be set using the var_int='q' or 'wl'

The input data to the models is obtained using:
    https://github.com/sinajahangir/Foundation-Models/tree/main
    HYDAT_Prediction_v..

The example provided here is associated with gauge 02GA014 in Ontario, CA
"""
#%%
#import necessary libraries
import numpy as np
import torch
import pandas as pd
import torch.nn as nn
import os
import random
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader,ConcatDataset
from matplotlib import rcParams
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
#%%
#set plot settings
params = {'legend.fontsize': 'large',
         'axes.labelsize': 'x-large',
         'axes.titlesize':'large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
pylab.rcParams.update(params)
rcParams['font.family'] = 'Calibri'
rcParams['axes.labelweight'] = 'bold'
rcParams['axes.titleweight'] = 'bold'
#%%
#change directory
os.chdir(r'F:\msjahang\HYSETS\Data\Flood') #change this
#%%
#set variable of ineterest: water level (wl) or streamflow (q)
var_int='wl'
#%%
#read input dat to the models
#Regional LSTM output
df_train_lstm=pd.read_csv(f'Prediction_Train_{var_int}_02GA014.csv').reset_index(drop=True)
df_test_lstm=pd.read_csv(f'Prediction_Test_{var_int}_02GA014.csv').reset_index(drop=True)


#TimesFM
# Subset the portion that overlaps with the simulation model
df_train_tfm=pd.read_csv(f'TimesFM_{var_int}_Train_02GA014.csv').iloc[-len(df_train_lstm):,1:].reset_index(drop=True)
df_test_tfm=pd.read_csv(f'TimesFM_{var_int}_Test_02GA014.csv').iloc[-len(df_test_lstm):,1:].reset_index(drop=True)
#%%
# Data integration
df_train_tfm['LSTM']=df_train_lstm['Prediction'].reset_index(drop=True)
df_test_tfm['LSTM']=df_test_lstm['Prediction'].reset_index(drop=True)
#%%
def split_sequence_multi_train(sequence_x,sequence_y, n_steps_in, n_steps_out,mode='seq'):
    """
    written by:SJ
    sequence_x=features; 2D array
    sequence_y=target; 2D array
    n_steps_in=IL(lookbak period);int
    n_steps_out=forecast horizon;int
    mode:either single (many to one) or seq (many to many).
    This function creates an output in shape of (sample,IL,feature) for x and
    (sample,n_steps_out) for y
    """
    X, y = list(), list()
    k=0
    sequence_x=np.copy(np.asarray(sequence_x))
    sequence_y=np.copy(np.asarray(sequence_y))
    for _ in range(len(sequence_x)):
		# find the end of this pattern
        end_ix = k + n_steps_in
        out_end_ix = end_ix + n_steps_out
		# check if we are beyond the sequence
        if out_end_ix > len(sequence_x):
            break
		# gather input and output parts of the pattern
        seq_x = sequence_x[k:end_ix]
        #mode single is used for one output
        if n_steps_out==0:
            seq_y= sequence_y[end_ix-1:out_end_ix]
        elif mode=='single':
            seq_y= sequence_y[out_end_ix-1]
        else:
            seq_y= sequence_y[end_ix:out_end_ix]
        X.append(seq_x)
        y.append(seq_y.flatten())
        k=k+1

    XX,YY= np.asarray(X), np.asarray(y)
    if (n_steps_out==0 or n_steps_out==1):
        YY=YY.reshape((len(XX),1))
    return XX,YY
#%%
class TimeSeriesDataset(Dataset):
    """
    A dataset class for regression tasks on time series data.
    
    Each sample consists of a single feature vector (row from data)
    and a corresponding scalar target.
    """
    def __init__(self, data, targets):
        """
        Args:
            data (np.ndarray or torch.Tensor): A 2D array of shape [num_samples, num_features].
            targets (np.ndarray or torch.Tensor): A 1D array of target values of shape [num_samples].
        """
        self.data = torch.tensor(data, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)

        assert len(self.data) == len(self.targets), "Data and targets must have the same number of samples."

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sequence = self.data[idx]
        target = self.targets[idx]
        return sequence, target

#%%
#data preperation
columns=df_train_tfm.columns.to_list()
columns.remove('date')
columns.remove('obs')


batch_size = 256

# Generate n random integers: n number of catchments for FT
random_numbers = random.sample(range(1), 1)
kk=0
#populate data loader
for ii in random_numbers:
  data_all=np.asarray(df_train_tfm.loc[:, columns])
  targets=np.asarray(df_train_tfm['obs']).reshape((-1,1))

  dataset_temp = TimeSeriesDataset(data_all[:int(0.9*len(data_all)),:], targets[:int(0.9*len(data_all)),:].reshape((-1,1)))
  dataset_val_temp=TimeSeriesDataset(data_all[int(0.9*len(data_all)):,:], targets[int(0.9*len(data_all)):,:].reshape((-1,1)))

  if kk==0:
    dataset = dataset_temp
    dataset_val=dataset_val_temp
  else:
    dataset = ConcatDataset([dataset, dataset_temp])
    dataset_val = ConcatDataset([dataset_val, dataset_val_temp])
  kk=kk+1
dataloader_train = DataLoader(dataset, batch_size=batch_size, shuffle=True)
dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=True)
#%%
#DL models
import torch.nn.functional as F
import torch.distributions as dist
class GAMeta(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1, dropout_prob=0.1):
        super(GAMeta, self).__init__()

        self.linear_hid=nn.Linear(input_size,hidden_size)
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout = nn.Dropout(dropout_prob)
        self.linear_mu = nn.Linear(hidden_size, output_size)
        self.linear_std = nn.Linear(hidden_size, output_size)
    

    def forward(self, x ,return_dist=False, return_params=False):
        
        out = self.linear_hid(x)
        out=F.relu(out)
        mu= self.linear_mu(out)
        log_std=self.linear_std(out)
        
        
        sigma = F.softplus(log_std) + 1e-6
        sigma = sigma * (0.1 + 0.2 * torch.sigmoid(mu.abs().mean()))

        if return_dist:
            return dist.Normal(mu, sigma)
        if return_params:
            return mu, sigma
        return mu
#%%
#set settings fro training the meta model

#set seed for reproducability
seed=1113
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # For CUDA
np.random.seed(seed)  # For NumPy
random.seed(seed)  # For Python's random module
torch.backends.cudnn.deterministic = True  # Ensures deterministic behavior

# Model parameters
input_size = 4  # Dynamic
hidden_size =1028
output_size = 1
dropout_prob = 0.1
#learning rate has a substantial impact if few catchments are used
lr=1e-3
# Create model
model= GAMeta(input_size, hidden_size)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
max_epochs_trained = 100
patience = 6
best_val_loss = float('inf')
early_stop_counter = 0
best_model_state = None
best_model_list=[]
epochs_trained = 0
optimizer = optim.Adam(model.parameters(),lr=lr)
scheduler = ReduceLROnPlateau(
        optimizer, mode='min', patience=4, factor=0.1, min_lr=1e-6)
#deterministic loss function
loss_fn =nn.MSELoss()
#%%
#training loop
for epoch in range(max_epochs_trained):
    model.train()
    epoch_loss = 0.0
    print('new epoch')
    print(epoch)
    for X_batch, Y_batch in dataloader_train:
        # Move data to the same device as the model
        X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
        optimizer.zero_grad()
        # Forward pass (returns distribution)
        pred_dist = model(X_batch, return_dist=True)
            
            # Negative log likelihood loss
        nll_loss = -pred_dist.log_prob(Y_batch).mean()
        
        nll_loss.backward()
        optimizer.step()
        epoch_loss += nll_loss.item()

    # Compute average training loss
    epoch_loss /= len(dataloader_train.dataset)

    # Validation every 10 epochs
    if epoch % 2 == 0:
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for XX_batch, YY_batch in dataloader_val:
                # Move data to the same device as the model
                XX_batch, YY_batch = XX_batch.to(device), YY_batch.to(device)
                # Forward pass for validation
                med = model(XX_batch)
                y_pred=med
                y_pred=y_pred.to(device)
                loss = loss_fn(y_pred,YY_batch)
                val_loss += loss.item()

        # Compute average validation loss
        val_loss /= len(dataloader_val.dataset)

        # Early Stopping Logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
            best_model_state = model.state_dict()
            best_model_list.append(best_model_state)
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch}!")
                break

        print(f"Epoch {epoch}: Train Loss: {epoch_loss:.5f}, Val Loss: {val_loss:.5f}")
    scheduler.step(val_loss)
    print(scheduler.get_last_lr())
    epochs_trained += 1
    # Step the scheduler
    if epochs_trained >= max_epochs_trained:
        print(f"Reached maximum of {max_epochs_trained} training epochs.")
        break
# Restore the best model
if best_model_state:
    model.load_state_dict(best_model_list[-1])
    print("Best model restored!")
    torch.save(best_model_list[-1], 'Model_MetaGA_Final_%1.4f_%s'%(lr,var_int))
#%%
save_name=f'Meta_GA_{hidden_size}_%1.4f_%s.csv'%(lr,var_int)
n_samples=100
for ii in range(0,1):
  



  temp_xx=np.asarray(df_test_tfm.loc[:, columns])
  temp_yy=np.asarray(df_test_tfm['obs']).reshape((-1,1))
  
  X_test=torch.tensor(temp_xx,dtype=torch.float32)
  Y_test=torch.tensor(temp_yy,dtype=torch.float32)

  model.eval()


  test_X = X_test.to(device).float()
  test_y = Y_test.to(device).float()
  
  predictions = []
  # Disable gradient calculation since we're in evaluation mode
  with torch.no_grad():
        for _ in range(n_samples):
            pred =model(test_X,return_dist=True).sample()
            pred = pred.cpu().numpy().ravel()
            predictions.append(pred)
  predictions = np.stack(predictions, axis=0)  # shape: (n_samples, n_obs)
    
    # Rescale
  predictions = predictions
  y_test = temp_yy
    # Compute quantiles
  lower = np.percentile(predictions, 2.5, axis=0)
  median = np.percentile(predictions, 50, axis=0)
  upper = np.percentile(predictions, 97.5, axis=0)
  

  torch.cuda.empty_cache()

  df_temp = pd.DataFrame({
        'date': df_test_tfm.iloc[-len(median):]['date'],
        'obs': y_test.ravel(),
        'lower': lower,
        'med': median,
        'upper': upper,
        'basin_id': ii
    })

  if ii == 0:
      df_temp.to_csv(save_name, index=False)
      df_all=df_temp
  else:
      df_existing = pd.read_csv(save_name)
      df_all = pd.concat([df_existing, df_temp], axis=0).reset_index(drop=True)
      df_all.to_csv(save_name, index=False)
#%%
def CC(x, y):
    return np.corrcoef(x.flatten(), y.flatten())[0, 1]

def KGE(prediction, observation):
    prediction = np.reshape(prediction, (-1, 1))
    observation = np.reshape(observation, (-1, 1))
    nas = np.logical_or(np.isnan(prediction), np.isnan(observation))
    pred = np.copy(prediction[~nas])
    obs = np.copy(observation[~nas])
    r = CC(pred, obs)
    beta = np.nanmean(pred) / np.nanmean(obs)
    gamma = (np.nanstd(pred) / np.nanstd(obs)) / beta
    kge = 1 - np.sqrt((r - 1) ** 2 + (beta - 1) ** 2 + (gamma - 1) ** 2)
    return kge

def NSE(Pr, Y):
    Pr = np.reshape(Pr, (-1, 1))
    Y = np.reshape(Y, (-1, 1))
    error = Y - Pr
    nse = 1 - (np.nansum((error) ** 2)) / np.nansum((Y - np.nanmean(Y)) ** 2)
    return nse
def compute_picp_quantiles_np(y_true, y_lower, y_upper):
    """
    Computes Prediction Interval Coverage Probability (PICP) using predicted quantiles.

    Args:
        y_true (np.ndarray): True target values, shape (n,) or (n, 1)
        y_lower (np.ndarray): Lower bound of the prediction interval, shape (n,)
        y_upper (np.ndarray): Upper bound of the prediction interval, shape (n,)

    Returns:
        float: PICP value in [0, 1]
    """
    y_true = np.squeeze(y_true)
    y_lower = np.squeeze(y_lower)
    y_upper = np.squeeze(y_upper)

    inside = np.logical_and(y_true >= y_lower, y_true <= y_upper)
    return np.mean(inside)
#%%
#plot and save figure 
fig, axi = plt.subplots(1, 1, figsize=(6, 6),dpi=400,sharex=True)
for ii in range(0,1):
    pred=np.asarray(df_all[df_all['basin_id']==ii].loc[:,'med']).reshape((-1,1))
    pred[pred<0]=0
    obs=np.asarray(df_all[df_all['basin_id']==ii].loc[:,'obs']).reshape((-1,1))
    low=np.asarray(df_all[df_all['basin_id']==ii].loc[:,'lower']).reshape((-1,1))
    low[low<0]=0
    upper=np.asarray(df_all[df_all['basin_id']==ii].loc[:,'upper']).reshape((-1,1))
    upper[upper<0]=0
    
    
    
    
    # Generate datetime index ending on 2023-12-30"
    end_date = pd.to_datetime("2023-12-30")
    n = len(obs)
    dates = pd.date_range(end=end_date, periods=n, freq='D')
    # Compute metrics
    nse_val = NSE(pred, obs)
    kge_val = KGE(pred, obs)
    picp=compute_picp_quantiles_np(obs,low,upper)*100

    # Plot
    
    axi.fill_between(x=dates,y1=low.ravel(),y2=upper.ravel(),color='blue',label='PI')
    axi.plot(dates,obs, color='red', lw=0.5, label='Observation')
    
    axi.set_ylabel(var_int)
    axi.set_title(f'NSE = {nse_val:.2f}, KGE = {kge_val:.2f}, PICP={picp:.2f}')

    # Legend only for the first subplot
    if ii == 0:
        leg = axi.legend()
        leg.get_frame().set_edgecolor('black')
        leg.get_frame().set_linestyle('--')
    axi.set_xlabel('Date')
plt.xticks(rotation=45)    
plt.tight_layout()
plt.savefig(f'Meta_GA_{hidden_size}_%1.4f_%s.png'%(lr,var_int))
    



