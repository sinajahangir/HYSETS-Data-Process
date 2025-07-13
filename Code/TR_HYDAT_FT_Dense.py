# -*- coding: utf-8 -*-
"""
This is code for fine-tuning (FT) a regioal long short-term memory netwrok (LSTM)
model trained on >390 catchments in CONUS on a limited number of catchments in Canada

This code can be used for FT for both streamflow and water level prediction. This
can be set using the var_int='q' or 'wl'

The input data to the models is obtained using:
DL_InputGenerator_HYSETS_v

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
#train data
csv_path_train='Alldata_Train_st_02GA014.csv'
df_train = pd.read_csv(csv_path_train)

#mean and std of all variables
mean_=np.asarray(df_train.iloc[:,1:].mean())
std_=np.asarray(df_train.iloc[:,1:].std())

#mean and std of the target
    #This will be used to retransform the data
mean_y=df_train[var_int].mean()
std_y=df_train[var_int].std()
#transfrom data
df_train_tr=df_train.iloc[:,1:]-mean_
df_train_tr=df_train_tr/(std_)
df_train_tr=df_train_tr.drop(columns=['basin_id'])
#here, as we only have one catchment, we can use a default constant value for all static features
df_train_tr.iloc[:,7:]=1
df_train_tr['basin_id'] = df_train['basin_id']
#test data
csv_path_test='Alldata_Test_st_02GA014.csv'
df_test = pd.read_csv(csv_path_test)
#transfrom data
df_test_tr=df_test.iloc[:,1:]-mean_
df_test_tr=df_test_tr/(std_)
df_test_tr=df_test_tr.drop(columns=['basin_id'])
df_test_tr.iloc[:,7:]=1
df_test_tr['basin_id'] = df_test['basin_id']
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
#Custom Dataset for LSTM Input
class TimeSeriesDataset(Dataset):
    """
    Custom dataset for handling large 2D arrays and converting them to LSTM-ready 3D sequences.
    """
    def __init__(self, data, targets, seq_length):
        """
        Args:
            data (np.ndarray or torch.Tensor): The 2D array of shape [num_samples, num_features].
            targets (np.ndarray or torch.Tensor): The 1D array of target values.
            seq_length (int): The length of the sequence for LSTM.
        """
        self.data = torch.tensor(data, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length + 1

    def __getitem__(self, idx):
        sequence = self.data[idx: idx + self.seq_length,:]
        target = self.targets[idx + self.seq_length - 1]  # Align target with the end of the sequence
        return sequence, target
#%%
#data preperation
columns=df_train_tr.columns.to_list()
if var_int=='wl':
    columns.remove('q')
else:
    columns.remove('wl')
    
columns.remove('basin_id')
columns.remove(var_int)


seq_length = 365
batch_size = 256

# Generate n random integers: n number of catchments for FT
random_numbers = random.sample(range(1), 1)
kk=0
#populate data loader
for ii in random_numbers:
  data_all=np.asarray(df_train_tr[df_train['basin_id']==ii].loc[:, columns])
  targets=np.asarray(df_train_tr[df_train['basin_id']==ii][var_int]).reshape((-1,1))

  dataset_temp = TimeSeriesDataset(data_all[:int(0.9*len(data_all)),:], targets[:int(0.9*len(data_all)),:].reshape((-1,1)), seq_length)
  dataset_val_temp=TimeSeriesDataset(data_all[int(0.9*len(data_all)):,:], targets[int(0.9*len(data_all)):,:].reshape((-1,1)), seq_length)

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

#original model
class ALLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_prob=0.1):
        super(ALLSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout_prob)
        self.linear_med = nn.Linear(hidden_size, output_size)
        self.linear_up = nn.Linear(hidden_size, output_size)
        self.linear_down = nn.Linear(hidden_size, output_size)
        self.linear_sigma = nn.Linear(hidden_size, output_size)  # Scale parameter
        #self.relu=nn.ReLU()

    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        lstm_out, _ = self.lstm(x)
        # lstm_out shape: (batch_size, sequence_length, hidden_size)
        out = self.dropout(lstm_out[:, -1, :]) # take the last time step output
        # out shape: (batch_size, hidden_size)
        med = self.linear_med(out)
        up = self.linear_up(out)
        down = self.linear_down(out)
        # out shape: (batch_size, output_size)
        return down, med, up
#FT LSTM model
class TransferLSTMModel(nn.Module):
    def __init__(self, allstm_model, input_size, hidden_size, output_size=1, dropout_prob=0.1):
        super(TransferLSTMModel, self).__init__()
        # Load only the pretrained LSTM layer
        self.lstm = allstm_model.lstm
        
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout = nn.Dropout(dropout_prob)
        self.linear = nn.Linear(hidden_size, output_size)
    

    def forward(self, x, return_weights=False):
        # Run input through pretrained LSTM
        lstm_out, (hidden, _) = self.lstm(x)
        # lstm_out shape: (batch_size, sequence_length, hidden_size)
        out = self.dropout(lstm_out[:, -1, :]) # take the last time step output
        out = self.linear(out)
        return out
'''
# As can be seen, the original model has a different structure compared to the FT model
We are only using the pre-trained model's LSTM cell
'''

#%%
#set settings fro training the FT model

#set seed for reproducability
seed=1113
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # For CUDA
np.random.seed(seed)  # For NumPy
random.seed(seed)  # For Python's random module
torch.backends.cudnn.deterministic = True  # Ensures deterministic behavior

# Model parameters
input_size = 30  # Dynamic+static
hidden_size = 256
num_layers = 1
output_size = 1
dropout_prob = 0.4
#learning rate has a substantial impact if few catchments are used
lr=5e-3
# Create model
model_ = ALLSTMModel(input_size, hidden_size, num_layers, output_size,dropout_prob)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_.to(device)
#load pretrained model
model_.load_state_dict(torch.load('ModelALLSTM_Final',weights_only=True))
model= TransferLSTMModel(model_,input_size, hidden_size, output_size,dropout_prob)
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
        med= model(X_batch)
        y_pred=med
        y_pred=y_pred.to(device)
        loss = loss_fn(y_pred,Y_batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    # Compute average training loss
    epoch_loss /= len(dataloader_train.dataset)

    # Validation every 2 epochs
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
torch.save(best_model_list[-1], 'Model_Dense_Final_%1.4f_%s'%(lr,var_int))
#%%
save_name=f'Dense_HYDAT_{seed}_%1.4f_%s.csv'%(lr,var_int)
n_samples=1 #this is for probabilistic DL. Set 1 for deterministic modeling
#loop through the catchments. Only one in this example
for ii in range(0,1):
  



  temp_xx=np.asarray(df_test_tr[df_test['basin_id']==ii].loc[:, columns])
  temp_yy=np.asarray(df_test_tr[df_test['basin_id']==ii]['q']).reshape((-1,1))
  xx_,yy_=split_sequence_multi_train(temp_xx,temp_yy,365,0,mode='seq')
  X_test=torch.tensor(xx_,dtype=torch.float32)
  Y_test=torch.tensor(yy_,dtype=torch.float32)

  model.eval()


  test_X = X_test.to(device).float()
  test_y = Y_test.to(device).float()
  
  predictions = []
  # Disable gradient calculation since we're in evaluation mode
  with torch.no_grad():
        for _ in range(n_samples):
            pred = model(test_X)
            pred = pred.cpu().numpy().ravel()
            predictions.append(pred)
  predictions = np.stack(predictions, axis=0)  # shape: (n_samples, n_obs)
    
    # Rescale
  predictions = predictions * std_y + mean_y
  y_test = yy_ * std_y + mean_y

  #makes no difference if n_sample==1
  #this can also be mean
  median = np.percentile(predictions, 50, axis=0)
  

  torch.cuda.empty_cache()

  df_temp = pd.DataFrame({
        'obs': y_test.ravel(),
        'pred': median,
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
# Your custom functions for perfromance analysis
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
#%%
#plot and save figure 
fig, axi = plt.subplots(1, 1, figsize=(6, 6),dpi=400,sharex=True)
for ii in range(0,1):
    pred=np.asarray(df_all[df_all['basin_id']==ii].loc[:,'pred']).reshape((-1,1))
    obs=np.asarray(df_all[df_all['basin_id']==ii].loc[:,'obs']).reshape((-1,1))
    
    
    
    
    # Generate datetime index ending on 2018-12-31
    end_date = pd.to_datetime("2023-12-30")
    n = len(obs)
    dates = pd.date_range(end=end_date, periods=n, freq='D')
    # Compute metrics
    nse_val = NSE(pred, obs)
    kge_val = KGE(pred, obs)

    # Plot
    
    axi.plot(dates,obs, color='red', lw=0.5, label='Observation')
    axi.plot(dates,pred, color='blue', lw=1, label='Prediction')
    
    axi.set_ylabel(var_int)
    axi.set_title(f'NSE = {nse_val:.2f}, KGE = {kge_val:.2f}')

    # Legend only for the first subplot
    if ii == 0:
        leg = axi.legend()
        leg.get_frame().set_edgecolor('black')
        leg.get_frame().set_linestyle('--')
    axi.set_xlabel('Date')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f'Dense_HYDAT_{seed}_%1.4f_%s.png'%(lr,var_int))
        

    



