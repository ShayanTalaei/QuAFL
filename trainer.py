import pandas as pd
import torch
from torch import Tensor
from typing import Tuple, List, Callable
import random
import warnings
import math
warnings.filterwarnings("ignore")

import pdb
import time 

from quantizer import *
from server import *
from client import *
from dataset_manager import *
from model_manager import *
from utils import *

from threading import Thread

class Trainer:
    
    SLOW_CLIENTS_RATIO = None

    def __init__(self, algorithm, dataset_name, client_count, train_sets_list, test_set, 
                 local_step, group_count, quantizer, initial_model, log_period, 
                 gpu_ids, server_averaging=True, client_averaging=True, **kwargs):
        
        self.algorithm = algorithm
        self.dataset_name = dataset_name
        self.gpu_ids = gpu_ids
        self.initial_model = initial_model
        batch_size = get_batch_size(dataset_name)
        self.optimizer = get_optimizer(dataset_name)
        self.criterion = get_criterion(dataset_name)
        
        self.server = Server(initial_model, self.criterion, quantizer, gpu_id=gpu_ids[0])
        self.client_count = client_count ## This is N, as the number of agents/clients.
        self.clients = []
        
        self.LOCAL_STEP  = local_step  ## This is K, as the number of steps that a model 
                                       ## should take to get ready for another interaction.
        
        self.GROUP_COUNT = group_count ## This is S, as the number of models that server 
                                       ## should interact with them at each step.
        
        self.log_period = log_period
        test_batch_size = 1000 if dataset_name in ["cifar 10", "celeba"] else 2000
        self.test_loader = data.DataLoader(test_set, batch_size = test_batch_size, 
                                           shuffle = True, num_workers=6)
        if len(train_sets_list) == 1:
            self.train_loader = data.DataLoader(train_sets_list[0], batch_size = test_batch_size, 
                                               shuffle = True, num_workers=6)
        else:
            self.train_loader = None
        self.history = []
        self.last_tested = 0
        self.setup_clients(train_sets_list, batch_size, quantizer, initial_model, gpu_ids)
        
        self.server_averaging = server_averaging
        self.client_averaging = client_averaging
        
        self.total_transferred_bits = 0
        self.b = kwargs.get('bits', 32)
        
        
    def setup_clients(self, train_sets_list, batch_size, quantizer, initial_model, gpu_ids):
        shared_dataset =  len(train_sets_list) == 1
        gpu_count = len(self.gpu_ids)
        slow_client_count = math.ceil(Trainer.SLOW_CLIENTS_RATIO * self.client_count)
        for i in range(self.client_count):
            if shared_dataset:
                sampler = torch.utils.data.distributed.DistributedSampler(train_sets_list[0], self.client_count, i, shuffle = True)
                dataloader_i = data.DataLoader(train_sets_list[0], batch_size=batch_size, 
                                           num_workers=2, sampler=sampler) #shuffle=True,
            else:
                dataloader_i = data.DataLoader(train_sets_list[i], batch_size=batch_size, 
                                           num_workers=2, shuffle=True)
            
            fast = i > (slow_client_count-1)
            client_i = Client(index = i, model = initial_model, optimizer = self.optimizer,
                              criterion = self.criterion, dataloader = dataloader_i,
                              quantizer = quantizer, gpu_id = self.gpu_ids[i%gpu_count], fast = fast)
            self.clients.append(client_i)
            print(f"Client {i+1} is added to the population as a {'fast' if fast else 'slow'} client.")
        print(f"There are {slow_client_count} slow clients and {self.client_count - slow_client_count} fast clients.")
    
    def train_client(self, client, lr, p, server_SD, client_dictionary_mode, server_model_ratio_on_client):
        max_steps = self.LOCAL_STEP
        run_time, taken_steps = client.run_until(lr* (1 / (1 - p) if self.server_averaging else 1),
                                                 self.server.time, max_steps)
        client_dict = client.get_model_dictionary(quantized = True, mode=client_dictionary_mode)
        self.send_dict_to_server(client_dict)
        self.server.seen_local_steps += taken_steps

        client.average_with_server_SD(server_SD, server_model_ratio_on_client)
        client.time = self.server.time
        
    def train_quantized_fl(self, lr, time_limit, decreasing):
        real_time = time.time()
        self.diverged = False
        p = 1 / (self.GROUP_COUNT + 1) ## Server ratio in averaging
        print(self.client_averaging, self.server_averaging)
        server_model_ratio_on_client = p if self.client_averaging else 1
        server_model_ratio_on_server = p if self.server_averaging else 0
        client_dictionary_mode = "state" #if server_averaging else "gradient"
        self.test()
        lr_factor = 1
        while(self.server.time < time_limit):
            # pdb.set_trace()
            ## The following lines are for time-based lr-scheduler, to decrease 
            ## the LR during the training you can uncomment them.
            if(self.server.time > time_limit * 0.6 and lr_factor == 1 and decreasing):
                lr_factor = 0.5
                lr *= 0.5
                print(f"The lr is {lr} from now!")
            elif(self.server.time > time_limit * 0.8 and lr_factor == 0.5 and decreasing):
                lr_factor = 0.1
                lr *= 0.2
                print(f"The lr is {lr} from now!")
            
            if(self.server.time - self.last_tested >= self.log_period or 
               (self.server.time - self.last_tested >= 10 and self.server.time <= 100)):
#                 pdb.set_trace()
                self.test()
                # torch.cuda.empty_cache()
                print(f"Real time: {time.time() - real_time}")
                real_time = time.time()

            interaction_group = random.sample(self.clients, self.GROUP_COUNT)
            server_SD = self.server.get_model_SD(quantized = True)
#             threads = []
            for client in interaction_group:
                ### The following lines are for multithread implementation of a single server step, to switch back to sequential form
                ### comment out the thread-related lines and uncomment the commented lines in the for loop.
#                 new_thread = Thread(target=self.train_client,args=(client, lr, p, server_SD, client_dictionary_mode, server_model_ratio_on_client))
#                 new_thread.start()
#                 threads.append(new_thread)
                max_steps = self.LOCAL_STEP ##random.randint(1, self.LOCAL_STEP)
                run_time, taken_steps = client.run_until(lr* (1 / (1 - p) if self.server_averaging else 1),
                                                         self.server.time, max_steps)
                client_dict = client.get_model_dictionary(quantized = True, mode=client_dictionary_mode)
                # run_time, taken_steps = client.take_step(self.LOCAL_STEP, lr, accumulate_grad=True)
                # client_dict = client.get_and_zero_gradient_dict(quantized=True)
                self.send_dict_to_server(client_dict)
                self.server.seen_local_steps += taken_steps
                self.server.aggregated_local_steps += taken_steps * (1 - server_model_ratio_on_server)/self.GROUP_COUNT
                self.total_transferred_bits += 2*self.b
                
                client.average_with_server_SD(server_SD, server_model_ratio_on_client)
                client.time = self.server.time
#             for thread in threads:
#                 thread.join()
#             if server_averaging:
            self.server.average_received_SDs(server_model_ratio = server_model_ratio_on_server)
#             else:
#                 self.server.apply_received_GDs()
            # self.server.apply_received_GDs(lr=1)
            self.server.interaction_count += 1
            self.server.time += Server.server_interaction_time
            self.server.time += Trainer.server_waiting_time
        self.test()
        
        return self.history
    
    def train_Fed_Avg(self, lr, time_limit, decreasing):
        self.diverged = False
        self.test()
        lr_factor = 1
        while(self.server.time < time_limit):
            ### The following lines are for time-based lr-scheduler, to decrease 
            ### the LR during the training you can uncomment them.
            if(self.server.time > time_limit * 0.6 and lr_factor == 1 and decreasing):
                lr_factor = 0.5
                lr *= 0.5
                print(f"The lr is {lr} from now!")
            elif(self.server.time > time_limit * 0.8 and lr_factor == 0.5 and decreasing):
                lr_factor = 0.1
                lr *= 0.2
                print(f"The lr is {lr} from now!")
            if(self.server.time - self.last_tested >= self.log_period or 
               (self.server.time - self.last_tested >= 10 and self.server.time <= 100)):
                self.test()

            server_SD = self.server.get_model_SD(quantized = True)
            interaction_group = random.sample(self.clients, self.GROUP_COUNT)
            run_times = []
            for client in interaction_group:
                client.load_SD(server_SD, quantized = True)
                run_time, taken_steps = client.take_step(self.LOCAL_STEP, lr, accumulate_grad=True)
                run_times.append(run_time)
                # client_SD = client.get_model_SD(quantized = True)
                client_SD = client.get_and_zero_gradient_dict(quantized=True)
                self.send_dict_to_server(client_SD)
                self.server.seen_local_steps += taken_steps
                self.total_transferred_bits += 2*self.b
            self.server.time += max(run_times)
            self.server.time += Server.server_interaction_time
#             self.server.model = deepcopy(interaction_group[0].model)
            # self.server.average_received_SDs(server_model_ratio = 0)
            self.server.apply_received_GDs(lr=1)
            self.server.interaction_count += 1
        self.test()

        return self.history 
    
    def train_FedBuff(self, lr, time_limit, decreasing):
        self.diverged = False
        self.test()
        lr_factor = 1
        participations = [0]*self.client_count
        while(self.server.time < time_limit):
            ### The following lines are for time-based lr-scheduler, to decrease 
            ### the LR during the training you can uncomment them.
            if(self.server.time > time_limit * 0.6 and lr_factor == 1 and decreasing):
                lr_factor = 0.5
                lr *= 0.5
                print(f"The lr is {lr} from now!")
            elif(self.server.time > time_limit * 0.8 and lr_factor == 0.5 and decreasing):
                lr_factor = 0.1
                lr *= 0.2
                print(f"The lr is {lr} from now!")
            if(self.server.time - self.last_tested >= self.log_period or 
               (self.server.time - self.last_tested >= 10 and self.server.time <= 100)):
                self.test()
                
            server_SD = self.server.get_model_SD(quantized = True)
            available_clients = [client for client in self.clients if client.unseen_steps < self.LOCAL_STEP]
            for client in available_clients:
                run_time, taken_steps = client.take_step(self.LOCAL_STEP, lr*self.GROUP_COUNT/self.client_count, accumulate_grad=True)
                self.server.seen_local_steps += taken_steps
                
            sorted_arrived_clients = sorted(self.clients, key=lambda x: x.time)
            early_arrived_clients = sorted_arrived_clients[:self.GROUP_COUNT]
            # print([(client.index, float("{0:.3f}".format(client.time))) for client in sorted_arrived_clients])
            for client in early_arrived_clients:
                self.server.time = max(self.server.time, client.time)
                participations[client.index] += 1
                grad_dict = client.get_and_zero_gradient_dict(quantized=True)
                self.server.aggregated_local_steps += client.unseen_steps / self.GROUP_COUNT
                client.unseen_steps = 0
                client.load_SD(server_SD, quantized=True)
                client.time = self.server.time
                self.send_dict_to_server(grad_dict)
                self.total_transferred_bits += 2*self.b
            self.server.time += Server.server_interaction_time
            self.server.interaction_count += 1
            self.server.apply_received_GDs(lr=1)
        self.test()
        print(f"Participations: {participations}.")
        return self.history
                
        
    def test(self):
        loss, acc = 0, 0
        model = self.server.model
        result = evaluate_on_dataloader(model, self.dataset_name, self.test_loader)
        result['Time'] = float(self.server.time)
        result['Server steps'] = int(self.server.interaction_count)
        result['Local steps'] = int(self.server.seen_local_steps)
        result['Transferred bits'] = int(self.total_transferred_bits)
        result['Aggregated local steps'] = int(self.server.aggregated_local_steps)
        # if self.train_loader != None:
        #     train_result = evaluate_on_dataloader(model, self.dataset_name, self.test_loader)
        #     result["Train loss"] = float(train_result["Loss"])
        #     result["Train accuracy"] = float(train_result["Accuracy"])
            
        
        self.last_tested = self.server.time
        time = self.server.time
        loss, acc = result["Loss"], result["Accuracy"]
        result["Loss"], result["Accuracy"] = float(loss), float(acc)
        print('Train: Step: {:5.0f} Val-Loss: {:.4f}  Val-Acc: {:.2f} Time: {:6.2f} Local Steps: {:5.0f}'.format(self.server.interaction_count, loss, acc, time, self.server.seen_local_steps))
        self.history.append(result)
        
    def send_dict_to_server(self, client_dict):
        self.server.received_dicts.append(client_dict)
    
    def train(self, lr, time_limit, decreasing):
        if self.algorithm == "quantized_fl":
            return self.train_quantized_fl(lr, time_limit, decreasing)
        elif self.algorithm == "Fed_Avg":
            return self.train_Fed_Avg(lr, time_limit, decreasing)
        elif self.algorithm == "FedBuff":
            return self.train_FedBuff(lr, time_limit, decreasing)