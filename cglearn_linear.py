######################################################################################################
## Consistent Gradient-based Learning (CGLearn) - linear
######################################################################################################
 
import math
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F

## linear implementation of CGLearn
class CGLBase(object):

    def __init__(self, environments, lr, n_iterations, v_method):
        
        dim_x = environments[0][0].size(1) ## feature dimension        
        validation_env = environments[-1] 
        training_envs = environments[:-1]
        
        ## search for best params
        best_w = None
        best_loss = float('inf')
        best_threshold = None
        possible_thresholds = [0.25, 1, 4]  ## consistency thresholds, needs to be modified based on search space

        for threshold in possible_thresholds:
            w = torch.nn.Parameter(torch.Tensor(dim_x, 1))
            # std_dev = math.sqrt(2.0 / (dim_x + 1))  # He initialization        
            std_dev = math.sqrt(1.0 / (dim_x + 1))  # Xavier initialization
            torch.nn.init.normal_(w, mean=0.0, std=std_dev) 

            optimizer = torch.optim.Adam([w], lr=lr)
            loss = torch.nn.MSELoss()
            
            for iteration in range(n_iterations):
                optimizer.zero_grad()            
                
                ## gradient calculation
                list_grads = []
                for x_e , y_e in training_envs: 
                    error_e = loss(x_e @ w, y_e) 
                    grads = torch.autograd.grad(error_e, w, create_graph=True)[0]
                    list_grads.append(grads)
                list_grads = torch.stack(list_grads)

                ## gradient consistency check
                mean_grads = torch.mean(list_grads, dim=0)
                std_grads = torch.std(list_grads, dim=0)
                ## adding a small value to the denominator to avoid division by zero case
                std_grads += 1e-8                
                cr = mean_grads / std_grads ## consistency ratio
                update_mask = torch.abs(cr) > threshold ## consistency mask                 
                if v_method == 0:
                    grad_w = mean_grads * update_mask.float()   
                else:
                    raise Exception('Error: v_method value is not valid!')               
                with torch.no_grad():
                    w.grad = grad_w
                optimizer.step()                        
            
            ## evaluation on validation env
            with torch.no_grad():
                x_val, y_val = validation_env
                val_loss = loss(x_val @ w, y_val)
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_w = w.clone()
                    best_threshold = threshold                
        self.w = best_w.view(-1, 1)

    def solution(self):
        return self.w
    
class CGLearn(CGLBase):
    def __init__(self, environments, lr, n_iterations):
        super(CGLearn, self).__init__(environments, lr, n_iterations, v_method=0)

## functions
def load_environments(case, n_envs):
    '''
        function to read datasets and load as environments.
    '''
    environments = []
    for env_idx in range(1, n_envs + 1):
        file_name = f"{curdir}datasets/linear/{case}_env{env_idx}.csv"
        df = pd.read_csv(file_name, header=None)
        data = df.values
        features = torch.tensor(data[:, :-1], dtype=torch.float32)
        target = torch.tensor(data[:, -1:], dtype=torch.float32)
        environments.append((features, target))
    return environments

def load_solution(case):
    '''
        function to read the solution
    '''
    solution_file_name = f"{curdir}datasets/linear/{case}_solution.pth"
    solution = torch.load(solution_file_name)
    return solution

if __name__=='__main__':
    
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)

    ## experimental variables
    curdir = ''
    n_envs = 3
    n_iterations = 10000
    lr = 1e-3 

    ## loading envrionments + solution with linear CGLearn
    case = 'FOU'
    environments = load_environments(case, n_envs)
    w_true, scramble_true = load_solution(case)[0].view(-1), load_solution(case)[1]
    model = CGLearn(environments, lr, n_iterations)
    w_pred = (scramble_true @ model.solution()).view(-1)

    ## calculate causal and noncausal errors 
    ## causal variables are where w_true is 1 and noncausal are where w_true is 0
    causal_mask = w_true != 0
    if causal_mask.any():
        error_causal = F.mse_loss(w_pred[causal_mask], w_true[causal_mask]).item()
    else:
        error_causal = 0
    noncausal_mask = w_true == 0
    if noncausal_mask.any():
        error_noncausal = F.mse_loss(w_pred[noncausal_mask], w_true[noncausal_mask]).item()
    else:
        error_noncausal = 0

    ##
    print(error_causal, error_noncausal)