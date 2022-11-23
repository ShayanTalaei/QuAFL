from quantizer import *
from server import *
from client import *
from dataset_manager import *
from model_manager import *
from trainer import *
from math import log2

def run(setups, dataset_name, log_period, **kwargs):
    trainers = [] 
    logs = {} 
    for case in setups:
        logs[case] = {}
    Trainer.SLOW_CLIENTS_RATIO = kwargs.get("slow_client_ratio", 0.3)
    print(f"Slow client ratio is {Trainer.SLOW_CLIENTS_RATIO}.")
    decreasing = kwargs.get('decreasing', False)
    print(f"Setups run with {'decreasing' if decreasing else 'constant'} lr.")
    train_sets_list, test_set = get_datasets(dataset_name, **kwargs)
    initial_model = kwargs.get("model", get_model(dataset_name))
    for case in setups:
        torch.cuda.empty_cache()
        print(f"--- {str(case)} ---")
        start = time.time()

        ## Setting run parameters
        case_params = setups[case]
        algorithm    = case_params['algorithm']
        client_count = case_params['client count']
        local_step   = case_params['local step']
        group_count  = case_params['group count']
        time_limit   = case_params['time_limit']
        lr           = case_params['lr']
        gpu_ids      = case_params['gpu_ids']

        if 'swt' in case_params.keys():
            Trainer.server_waiting_time = case_params['swt']
        if 'sit' in case_params.keys():
            Server.server_interaction_time = case_params['sit']

        server_averaging = case_params['server_averaging'] if 'server_averaging' in case_params.keys() else True
        client_averaging = case_params['client_averaging'] if 'client_averaging' in case_params.keys() else True

        quantization_params = case_params["quantizer"]
        if quantization_params['method'] == 'identity':
            bits = 32
        elif quantization_params['method'] == 'lattice':
            bits = quantization_params['quant_q']
        elif quantization_params['method'] == 'qsgd':
            bits = int(log2(quantization_params['k'])) 
            
        quantizer = get_quantizer(**quantization_params)


        trainer = Trainer(  algorithm = algorithm,
                            dataset_name = dataset_name,
                            client_count = client_count,
                            train_sets_list = train_sets_list,
                            test_set = test_set,
                            local_step = local_step,
                            group_count = group_count,
                            quantizer = quantizer,
                            initial_model = initial_model, 
                            log_period = log_period,
                            gpu_ids    = gpu_ids,
                            server_averaging = server_averaging,
                            client_averaging = client_averaging,
                            bits = bits)

        ## Keeping all the trainers is so memory consuming, so comment out next line if you don't need them.
#         trainers.append(trainer)   

        history = trainer.train(lr=lr, time_limit=time_limit, decreasing=decreasing)

        for key in history[0].keys():
            logs[case][key] = [x[key] for x in history]
        end = time.time()
        print(f"Finished in {end - start}")
            
    return logs, trainers 
