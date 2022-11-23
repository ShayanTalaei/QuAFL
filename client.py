import torch
from copy import deepcopy
from quantizer import *
import numpy as np
from model_manager import *
from utils import *
        
class Client: 

    server_in_averaging = True
    
    def __init__(self, index, model, optimizer, criterion, dataloader, quantizer, gpu_id, fast):
        self.index = index
        self.gpu_id = gpu_id
        self.model = deepcopy(model).cuda(self.gpu_id)
        self.dataloader = dataloader
        self.quantizer  = quantizer
        self.dataloader_iterator = None
        
        ## For simulating time-based runs
        self.time = 0
        self.mean_step_time = 2 if fast else 8 
        self.next_step_time = self.get_run_time()
        
        ##Initialize based on the dataset
        self.criterion = criterion
        self.optimizer = optimizer(self.model.parameters(), lr=0.001)
        
        self.unseen_steps = 0
        self.gradient_dict = {}
        self.zero_gradient_dict()
    
    def take_step(self, step_count, lr, accumulate_grad=False):
        training_mode = self.model.training
        self.model.train()
        for g in self.optimizer.param_groups:
            g['lr'] = lr
        
        taken_steps = 0
        run_time = 0
        while(taken_steps < step_count):
            if(self.dataloader_iterator == None):
                self.dataloader_iterator = iter(self.dataloader)
            else:
                try:
                    data = next(self.dataloader_iterator)
                    Xb, yb = data
                    Xb, yb = Xb.cuda(self.gpu_id), yb.cuda(self.gpu_id)
                    # zero the parameter gradients
                    self.optimizer.zero_grad()

                    # forward + backward + optimize
                    outputs = self.model(Xb)
                    loss = self.criterion(outputs, yb)
                    loss.backward()
                    if accumulate_grad:
                        self.accumulate_on_gradient_dict(lr)
                    self.optimizer.step()
                    # self.optimizer.zero_grad()
            
                    taken_steps += 1
                    self.unseen_steps += 1
                    self.time += self.next_step_time
                    run_time += self.next_step_time
                    self.next_step_time = float(self.get_run_time())
                    # del Xb, yb, outputs, loss
                except StopIteration:
                    # del self.dataloader_iterator
                    self.dataloader_iterator = iter(self.dataloader)
        self.model.train(training_mode)
        return run_time, taken_steps
    
    def run_until(self, lr, time, max_steps):
        taken_steps = 0
        total_run_time = 0
        while(taken_steps < max_steps and 
              self.time + self.next_step_time < time):
            run_time, step_count = self.take_step(1, lr)
            assert step_count == 1
            total_run_time += run_time
            taken_steps += 1
        return total_run_time, taken_steps
    
    def compute_gradient(self):
        Xb, Yb = None, None
        while(Xb == None):
            if(self.dataloader_iterator == None):
                self.dataloader_iterator = iter(self.dataloader)
            else:
                try:
                    data = next(self.dataloader_iterator)
                    Xb, yb = data
                    Xb, yb = Xb.cuda(self.gpu_id), yb.cuda(self.gpu_id)
                    # zero the parameter gradients
                    self.optimizer.zero_grad()

                    # forward + backward + optimize
                    outputs = self.model(Xb)
                    loss = self.criterion(outputs, yb)
                    loss.backward()
                    
                except StopIteration:
                    self.dataloader_iterator = iter(self.dataloader)
        
    def get_model(self, quantized):
        params = torch.nn.utils.parameters_to_vector(self.model.parameters())
        if quantized:
            return self.quantizer.encode(params)
        else:
            return params
    
    def get_model_SD(self, quantized):
        copy_SD = self.model.state_dict()
        if(quantized):
            for key in copy_SD:
                copy_SD[key] = self.quantizer.encode(copy_SD[key]).detach()
        return copy_SD
            
    def load_SD(self, new_state_dict, quantized):
        current_SD = self.model.state_dict()
        for key in current_SD:
            new_value = new_state_dict[key]
            if quantized:
                new_value = self.quantizer.decode(new_value, current_SD[key])
            current_SD[key] = new_value
        self.model.load_state_dict(current_SD)
    
    def average_with_server_SD(self, server_state_dict, server_model_ratio):
        with torch.no_grad():
            p, q = server_model_ratio, 1 - server_model_ratio
            current_SD = self.model.state_dict()
    #         server_state_dict = put_state_dict_on_gpu(server_state_dict, self.gpu_id)
            for key in current_SD:
                decoded_server_SD = self.quantizer.decode(server_state_dict[key], current_SD[key])
                decoded_server_SD = decoded_server_SD
                current_SD[key]   = p*decoded_server_SD + q*current_SD[key]
            self.model.load_state_dict(current_SD)
        
    def get_run_time(self):
        run_time = np.random.exponential(self.mean_step_time) 
        return run_time

############################   Gradient_dict methods   ############################
    ### gradient_dicts are dictionaries from model named-parameters to their gradient values.
    ### For non-learnable parameters such as batch-norm "num", "mean", and "var", their actual 
    ### values are stored in the gradient_dicts.

    def get_and_zero_gradient_dict(self, quantized):
        copy_GD = deepcopy(self.gradient_dict)
        ## There are some statistical parameters without gradient like batch normalization parameters
        ## for them we send their actual value
        model_SD = deepcopy(self.model.state_dict())
        for key in model_SD:
            if 'num' in key or 'var' in key or 'mean' in key: ## Statistical parameters
                copy_GD[key] = model_SD[key]
        self.zero_gradient_dict()
        if(quantized):
            for key in copy_GD:
                copy_GD[key] = self.quantizer.encode(copy_GD[key])
        return copy_GD
    
    def accumulate_on_gradient_dict(self, lr):
        for key, param in self.model.named_parameters():
            self.gradient_dict[key] += lr*param.grad
    
    def zero_gradient_dict(self):
        self.gradient_dict = {k:torch.zeros_like(v) for k,v in self.model.named_parameters()}
    
    def get_model_dictionary(self, quantized, mode = "state"):
        ## This method returns either "state"_dict or "gradient"_dict of the client's model,
        ## identified by mode
        
        if mode == "state":
            return self.get_model_SD(quantized)
        elif mode == "gradient":
            return self.get_and_zero_gradient_dict(quantized)
        else:
            print("Requested mode is invalid!")
            return None
        
################################   DEPRECATED   ################################      
    ### These functions work with tensor form of model.parameters(), however non-learnable parameters
    ### such as batch-norm "num", "mean", and "var" are not learnable hence not present in the tensors. 
    ### This fact causes problem in averaging server parameters. Therefore, we use state_dicts,  
    ### gradient_dicts in the functions above to pass the parameters between clients and server.  

    def average_with_server(self, quantized_params, server_model_ratio):
        params = torch.nn.utils.parameters_to_vector(self.model.parameters())
        decoded_params = self.quantizer.decode(quantized_params, params)
        p, q = server_model_ratio, 1 - server_model_ratio
        new_params = p * decoded_params + q * torch.nn.utils.parameters_to_vector(self.model.parameters())
        torch.nn.utils.vector_to_parameters(new_params, self.model.parameters())
        
    def set_model(self, new_params, quantized):
        params = torch.nn.utils.parameters_to_vector(self.model.parameters())
        new_params = torch.clone(new_params)
        if quantized:
            new_params = self.quantizer.decode(new_params, params)
        torch.nn.utils.vector_to_parameters(new_params, self.model.parameters())