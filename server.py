import torch
from copy import deepcopy
from utils import *

class Server:
    
    server_interaction_time = 1
    
    def __init__(self, model, criterion, quantizer, gpu_id):
        self.gpu_id = gpu_id
        self.model = deepcopy(model).cuda(self.gpu_id)
        self.criterion = criterion
        self.quantizer = quantizer
        self.received_dicts = []   ## State (gradient) dicts received from clients
        
        ## For simulation statistics
        self.interaction_count = 0 ## Number of interactions (sever steps)
        self.seen_local_steps  = 0 ## Total number of local steps seen by the server
        self.aggregated_local_steps = 0 ## The number of SGD steps which have move the server model.
        self.time = 0
    
    def average_received_SDs(self, server_model_ratio):
        with torch.no_grad():
            ## In case that server receives state dicts of clients and do the averaging with them

            received_SD_count = len(self.received_dicts)
            if(received_SD_count == 0):
                return

            current_SD = self.model.state_dict()
            p, q = server_model_ratio, 1 - server_model_ratio
            for key in current_SD:
                temp_value = torch.zeros_like(current_SD[key]).float()
                for state_dict in self.received_dicts:
    #                 state_dict = put_state_dict_on_gpu(state_dict, self.gpu_id)
                    temp_value += self.quantizer.decode(state_dict[key], current_SD[key])
                temp_value /= received_SD_count*(1.0)
                current_SD[key] = p*current_SD[key] + q*temp_value
            self.model.load_state_dict(current_SD)
            self.received_dicts = []
    
    def get_model_SD(self, quantized):
        ## Get the state dict of server's model, quintized if requested
        
        copy_server_SD = deepcopy(self.model.state_dict())
        if(quantized):
            for key in copy_server_SD:
                copy_server_SD[key] = self.quantizer.encode(copy_server_SD[key])
        return copy_server_SD

    def apply_received_GDs(self, lr=1):
        ## In case that server receives updates (accumulated gradients) of clients
        ## and apply the gradient step on its model
        
        received_dicts_count = len(self.received_dicts)
        if(received_dicts_count == 0):
            return
        
        current_SD = self.model.state_dict()
        for key in current_SD:
            temp_value = torch.zeros_like(current_SD[key]).float()
            for gradient_dict in self.received_dicts:
                # gradient_dict = put_state_dict_on_gpu(state_dict, self.gpu_index)
                temp_value += self.quantizer.decode(gradient_dict[key], current_SD[key])
            temp_value /= received_dicts_count*(1.0)
            if 'num' in key or 'var' in key or 'mean' in key: ## It's a statistical parameter that should be averaged.
                p = 1 / (received_dicts_count + 1)
                current_SD[key] = p*current_SD[key] + (1-p)*temp_value
            else: ## It's a differentiable parameter which it's gradient from the client has been sent
                current_SD[key] -= temp_value*lr
        self.model.load_state_dict(current_SD)
        self.received_dicts = []