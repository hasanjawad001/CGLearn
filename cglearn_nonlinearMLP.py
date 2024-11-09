######################################################################################################
## Consistent Gradient-based Learning (CGLearn) - nonlinear MLP
######################################################################################################

## import libraries 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, silhouette_score
from sklearn.utils import resample
import pickle
import torch.autograd as autograd

## functions
def get_data(tn=0):
    '''
        Generates and returns data/envrionments based on different experimental case scenarios.
    '''
    
    ## reading data
    seed = 42 + tn 
    df = pd.read_csv(curdir + 'datasets/nonlinear/yacht_hydrodynamics.data', delim_whitespace=True, header=None)
    data = df.iloc[:,:-1].values
    target = df.iloc[:, -1].values
    X = data
    y = target.reshape(-1, 1)  

    def determine_clusters(X, min_clusters=3, max_clusters=10):
        '''
            Determines the optimal number of clusters based on silhoutte scores.
            Here the optimal #clusters is determined by using range of 3 to 10.
        '''
        silhouette_scores = []
        for n_clusters in range(min_clusters, max_clusters + 1):
            kmeans = KMeans(n_clusters=n_clusters, random_state=seed)
            labels = kmeans.fit_predict(X)
            silhouette_scores.append(silhouette_score(X, labels))
        optimal_clusters = np.argmax(silhouette_scores) + min_clusters  
        return optimal_clusters

    ## standardization
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    ## using K-Means to create environments based on the optimal #environments
    optimal_clusters = determine_clusters(X_scaled)
    kmeans = KMeans(n_clusters=optimal_clusters, random_state=seed)
    envs = kmeans.fit_predict(X_scaled)

    ## data to environments
    environments_X, environments_y = [], []
    for i in range(optimal_clusters):
        env_X = X_scaled[envs == i]
        env_y = y[envs == i]
        environments_X.append(env_X)
        environments_y.append(env_y)

    ## generating all possible cases -
    ##     where one environment is considered as the test environment,
    ##     and the rest are considered as for training.
    ##     every environment is considered as test environment for once.
    ans = []
    for i in range(optimal_clusters):        
        environments_X_tensors, environments_y_tensors, X_test, y_test = [], [], None, None
        env_no = 0
        for env_X, env_y in zip(environments_X, environments_y):
            if env_no == i:
                X_test, y_test = torch.tensor(env_X, dtype=torch.float32), torch.tensor(env_y, dtype=torch.float32)
            else:
                environments_X_tensors.append(torch.tensor(env_X, dtype=torch.float32))
                environments_y_tensors.append(torch.tensor(env_y, dtype=torch.float32))                
            env_no +=1
        ans.append((environments_X_tensors, environments_y_tensors, X_test, y_test))
    return ans

def train_model_cgl(
    model, X_train_envs, y_train_envs, 
    X_val, y_val, 
    learning_rate, epochs, l2_regularizer_weight, crp=0, method=1
):
    '''
        Function to train model using the consistent gradient-based learning (CGLearn) strategy
    '''
    
    model.to(device)                        
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        list_grads = []  
        feature_l2_norms = []            
        
        for X_train, y_train in zip(X_train_envs, y_train_envs):
            X_train = X_train.to(device) 
            y_train = y_train.to(device)                                                 
            outputs = model(X_train)
            loss = torch.sqrt(criterion(outputs, y_train))                
            l2_norm = torch.tensor(0., device=device)
            for param in model.parameters():
                l2_norm += torch.norm(param, p=2) ** 2
            final_loss = loss + (l2_regularizer_weight * l2_norm)

            optimizer.zero_grad()
            final_loss.backward()            
            train_loss += loss.item()                
            grads = []
            for param in model.parameters():
                grads.append(param.grad.clone().flatten())
            all_grads = torch.cat(grads)
            list_grads.append(all_grads)
            l2_norms = torch.norm(model.fc1.weight.grad, dim=0, p=2).detach() ## calculating the L2-norm of grads for each features for the first hidden layer (h1)
            feature_l2_norms.append(l2_norms)                

        train_loss = train_loss/len(X_train_envs)
        list_grads = torch.stack(list_grads)             
        net_grads = torch.mean(list_grads, dim=0)

        feature_l2_norms = torch.stack(feature_l2_norms)
        mean_norms = torch.mean(feature_l2_norms, dim=0)
        std_norms = torch.std(feature_l2_norms, dim=0) + 1e-8  
        cr = torch.abs(mean_norms) / std_norms ## consistency ratio (this is to be compared with consistency threshold 'ct' to determine gradient consistency)            

        if method == 1:
            ct = np.percentile(cr.cpu().numpy(), crp)                
            consistency_mask = torch.where(cr >= ct, torch.tensor(1., device=device), torch.tensor(0., device=device))
            consistency_mask = consistency_mask.repeat(hidden_dim, 1)  
        else:
            raise Error('Error: Method not recognized!')

        optimizer.zero_grad()      
        start_index = 0
        for name, param in model.named_parameters():
            param_numel = param.numel() 
            mean_grad = net_grads[start_index : start_index + param_numel].view_as(param)                
            if method == 1 and 'fc1.weight' in name: ## imposing consistency check on first hidden layer (h1)
                masked_mean_grad = mean_grad * consistency_mask
                param.grad = masked_mean_grad
            elif method == 1: ## the other layer's weights are updated as usual (does not depend on consistency)
                param.grad = mean_grad                                
            start_index += param_numel
        optimizer.step()   


    model.eval()
    with torch.no_grad():
        X_train = torch.cat(X_train_envs, dim=0).to(device) 
        y_train = torch.cat(y_train_envs, dim=0).to(device) 
        train_outputs = model(X_train)
        train_loss = torch.sqrt(criterion(train_outputs, y_train))
        X_val = X_val.to(device) 
        y_val = y_val.to(device)                                    
        val_outputs = model(X_val)
        val_loss = torch.sqrt(criterion(val_outputs, y_val))      
        
    return model, val_loss.item(), train_loss.item(), cr

def evaluate_model(model, X_test, y_test):
    '''
        Function to evaluate the trained model on test environment.
    '''
    model.to(device) 
    X_test = X_test.to(device) 
    y_test = y_test.to(device) 
    criterion = nn.MSELoss()
    model.eval()
    with torch.no_grad():
        predictions = model(X_test)
        test_loss = torch.sqrt(criterion(predictions, y_test))
    return test_loss.item()

## model - nonlinear
class MLP(nn.Module):
    def __init__(self, input_size, hidden_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_dim)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = nn.ReLU()
        self.output = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.output(x)
        return x

if __name__=='__main__':
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")    
    curdir = ''
    lr = 1e-2 ## learning rate
    epochs = 100 
    hidden_dim = 60 
    l2_regularizer_weight = 0
    tn = 0 
    p = 50 ## used to select 'ct' (consistency threshold)

    lt, lv = [], []
    ans = get_data(tn=tn)
    for caseno, case in enumerate(ans):
        environments_X_tensors, environments_y_tensors, X_test, y_test = case
        model = MLP(environments_X_tensors[0].shape[1], hidden_dim).to(device)
        model, val_loss, train_loss, cr = train_model_cgl(
            model, environments_X_tensors, environments_y_tensors, 
            X_test, y_test, 
            lr, epochs, l2_regularizer_weight, crp=p, method=1
        )
        X_train_merged = torch.cat(environments_X_tensors, dim=0).to(device) 
        y_train_merged = torch.cat(environments_y_tensors, dim=0).to(device)    
        train_loss = evaluate_model(model, X_train_merged, y_train_merged)
        X_test = X_test.to(device) 
        y_test = y_test.to(device)                            
        val_loss = evaluate_model(model, X_test, y_test)
        lt.append(train_loss)
        lv.append(val_loss)

    mean_lt, mean_lv = np.mean(lt), np.mean(lv)
    print(mean_lt, mean_lv)