import h5py    
import numpy as np    
import logging

import torch
import torch.nn as nn

import numpy as np
from torch.optim.lr_scheduler import ExponentialLR

import os

import wandb

import time


from src.dataset_processing import create_dataloaders, create_dataloader_taxi
from src.train_loop import train_model, train_model_with_RL, train_model_routed
from src.models.ConvLSTM import ConvLSTM
from src.models.ConvSelfAttention import ConvSelfAttention
from src.models.BiDepth import BidirectionalDeepShallow, DeepShallowConvLSTM2
from src.models.AdaptiveDepth import AdDepth, QLearningAgent
from src.models.DepthRouter import AdDepthRouted


def model_num_params(model):
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {total_params}")
    return total_params

def main():

    mulit_run = True
    taxi = True

    # Setup logging
    logging.basicConfig(filename='training_log.log', level=logging.INFO, format='%(asctime)s %(message)s')

    file_name = "../data/weather/train_test_2015_2020.h5"
    gpu_number = 0
    checkpoint_path = "checkpoint_ShallowDeep.pt"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,5,6"

    channels = 16 if taxi else 8
    window_size = 7 if taxi else 6
    kernel_size = (3,3)
    input_dim = channels
    in_channels = channels
    out_channels = channels
    hidden_dim = [out_channels]
    num_layers = len(hidden_dim)
    
    batch_size = 128 if taxi else 16
    learning_rate = 0.005
    max_depth = 10 if taxi else 6
    min_depth = 1
    initial_depth = 5
    attention = False
    Rl_start_step = 10
    dynamic_layer_counts = None # When loading
    loss_spike_threshold = False
    # loss_spike_threshold = 1.1

    pooling = None
    convdeconv = None

    # Define the number of training epochs
    num_epochs = 500 if taxi else 50
    validation_interval = 5 if taxi else int(10000/ batch_size)# Now Refers to batch count for images validation_interval+batch =100000
    patience = 20
    use_wandb = True

    max_val_loss = 2
    discrete_states = 25

    # --- AdDepthRouted (differentiable router, AdDepth v2) hyperparameters ---
    lambda_budget = 0.01        # compute-budget penalty; sweep {0, 1e-3, 1e-2, 1e-1}
    tau_start, tau_end = 2.0, 0.5  # Gumbel-softmax temperature annealing

    # --- Ablation "random + rollback" (no Q-learning) ---
    # Run train_model_with_RL with an agent whose actions are always random:
    #   QLearningAgent(window_size, init_loss, exploration_rate=1.0, exploration_decay=1.0)
    # Only the rollback mechanism remains; if this matches AdDepth, the
    # Q-learning component adds nothing.


    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print( 'Mac MPS is avilable' )
    else:
        device = torch.device(f"cuda:{gpu_number}" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available() : 
            print( 'Cuda is avilable' )

    if not taxi:
        with h5py.File(file_name, "r") as f:
            print("Keys: %s" % f.keys())

            data_dict = {}
            for k, v in f.items():
                data_dict[k] = np.array(v)


    if mulit_run == False:

        if taxi:
            train_loader, val_loader, test_loader = create_dataloader_taxi(batch_size)

        else: # Weather
            train_loader, val_loader, test_loader = create_dataloaders(
                data_dict['Precipitation'], 
                frame_number=channels, 
                window_size=window_size, 
                batch_size=batch_size, 
                train_split=0.85, 
                val_split=0.05
            )

        for data in train_loader:
            break

        # Create an instance of the CNNRNN
        # model, model_arch = (ConvLSTM(input_dim, hidden_dim, kernel_size, num_layers, in_channels, out_channels, data[0].shape[-2:], window_size).to(device), "ConvLSTM")
        # model, model_arch = (ConvSelfAttention(input_dim, hidden_dim, kernel_size, num_layers, in_channels, out_channels, data[0].shape[-2:]  , window_size, pooling = pooling, convdeconv=convdeconv ).to(device),  "ConvAttention")
        ShallowDeep = False
        model, model_arch = (DeepShallowConvLSTM2(input_dim, hidden_dim, kernel_size, num_layers, in_channels, out_channels, data[0].shape[-2:], window_size, initial_layers = max_depth, reverse = ShallowDeep , attention = attention, pooling=pooling, convdeconv=convdeconv).to(device),  "ShallowDeep" if ShallowDeep else "DeepShallow")
        # model, model_arch = (BidirectionalDeepShallow(input_dim, hidden_dim, kernel_size, num_layers, in_channels, out_channels,  data[0].shape[-2:] ,window_size, initial_layers = max_depth, attention=attention, pooling=pooling, convdeconv=convdeconv).to(device),  "BiDepth")
        # model, model_arch = (AdDepth(input_dim, hidden_dim, kernel_size, num_layers, in_channels, out_channels, data[0].shape[-2:] ,window_size, initial_layers = max_depth , attention = attention, pooling=pooling, convdeconv=convdeconv).to(device), "AdDepth")

        # start a new wandb run to track this script
        if use_wandb:
            wandb.init(
                # set the wandb project where this run will be logged
                # project="nyc-taxi",
                project="nyc-taxi" if taxi else "WeatherData",

                # track hyperparameters and run metadata
                config={
                "learning_rate": learning_rate,
                "architecture": model_arch,
                "dataset": "taxi" if taxi else "Weather",
                "epochs": num_epochs,
                "batch_size": batch_size,
                "channels": channels,
                "hidden_dim" : str(hidden_dim),
                "window_size": window_size,
                "max_depth": max_depth,
                "pooling": pooling,
                "convdeconv": convdeconv,
                "validation_interval" : validation_interval,
                "patience" : patience,
                "attention" : attention,
                'model_parameters_count': model_num_params(model),
                # "loss_spike_threshold": loss_spike_threshold,
                # "allow_val_change":True
                }
            )
            columns = ["treshhold" , "loss" , "best_layer" , "epoch" ,"time"]
            table = wandb.Table(columns=columns)


        # Check if multiple GPUs are available
        # if torch.cuda.device_count() > 1:
        #     print("Using", torch.cuda.device_count(), "GPUs.")
        #     # Wrap the model for multi-GPU usage using DataParallel
        #     model = nn.DataParallel(model)
        #     device = 'cuda'

        # Load model Weights:
        # dynamic_layer_counts = [6, 6, 5, 3, 6, 4]
        # model.update_dynamic_layer_counts(dynamic_layer_counts)
        # model.load_state_dict(torch.load("checkpoint.pt"))

        # Define the loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.AdamW(params = model.parameters() ,  lr = learning_rate)

        # Define the learning rate decay scheduler
        scheduler = ExponentialLR(optimizer, gamma=0.95)
        scheduler = None

        # train_model returns (model, train_losses, val_losses, best_val_loss, last_epoch)
        model, train_losses, val_losses, _, _ = train_model(
            model,
            criterion,
            optimizer,
            train_loader,
            val_loader,
            test_loader,
            num_epochs,
            validation_interval,
            patience,
            logging,
            device,
            scheduler= scheduler,
            print_losses=True,
            loss_logging = wandb if use_wandb else None,
            checkpoint_path = checkpoint_path,
        )


        # model, train_losses, val_losses, _  = train_model_with_RL(
        #     model,
        #     criterion,
        #     optimizer,
        #     train_loader,
        #     val_loader,
        #     test_loader,
        #     num_epochs,
        #     validation_interval,
        #     patience,
        #     q_learning_agent,  # Instance of QLearningAgent
        #     window_size,
        #     max_depth,
        #     initial_depth,
        #     logging,
        #     device,
        #     min_depth = min_depth,
        #     Rl_start_step = Rl_start_step,
        #     current_layers= dynamic_layer_counts,
        #     scheduler= scheduler,
        #     loss_spike_threshold = loss_spike_threshold,
        #     print_losses=True,
        #     loss_logging = wandb if use_wandb else None,
        #     checkpoint_path = checkpoint_path,
        # )

        if use_wandb:
            wandb.finish()


    # loss_spike_thresholds = [1 , 1.01 , 1.02 , 1.03 , 1.05, 1.1, 1.15 , 1.2, 1.3 , 1.4 , 1.5, 2.0,1 , 1.01 , 1.02 , 1.03 , 1.05, 1.1, 1.15 , 1.2, 1.3 , 1.4 , 1.5, 2.0,1 , 1.01 , 1.02 , 1.03 , 1.05, 1.1, 1.15 , 1.2, 1.3 , 1.4 , 1.5, 2.0,1 , 1.01 , 1.02 , 1.03 , 1.05, 1.1, 1.15 , 1.2, 1.3 , 1.4 , 1.5, 2.0,1 , 1.01 , 1.02 , 1.03 , 1.05, 1.1, 1.15 , 1.2, 1.3 , 1.4 , 1.5, 2.0,1 , 1.01 , 1.02 , 1.03 , 1.05, 1.1, 1.15 , 1.2, 1.3 , 1.4 , 1.5, 2.0]
    # attention_mechanism = [False]
    if mulit_run:
    
        # hidden_dims = [[out_channels] , [64, out_channels] , [16,16,out_channels] ]
        max_depths = [75, 100, 150, 200]

        results = []

        for max_depth in max_depths:
            print(f"Training with hidden_dim: {str(hidden_dim)}")
            num_layers = len(hidden_dim)


            if taxi:
                train_loader, val_loader, test_loader = create_dataloader_taxi(batch_size)

            else:
                train_loader, val_loader, test_loader = create_dataloaders(
                    data_dict['Precipitation'], 
                    frame_number=channels, 
                    window_size=window_size, 
                    batch_size=batch_size, 
                    train_split=0.85, 
                    val_split=0.05
                )


            for data in train_loader:
                break
                
            # Reset the model and optimizer for each run
            # model, model_arch = (ConvLSTM(input_dim, hidden_dim, kernel_size, num_layers, in_channels, out_channels, data[0].shape[-2:], window_size).to(device), "ConvLSTM")
            # model, model_arch = (ConvSelfAttention(input_dim, hidden_dim, kernel_size, num_layers, in_channels, out_channels, data[0].shape[-2:]  , window_size, pooling = pooling, convdeconv=convdeconv ).to(device),  "ConvAttention")
            # ShallowDeep = False
            # model, model_arch = (DeepShallowConvLSTM2(input_dim, hidden_dim, kernel_size, num_layers, in_channels, out_channels, data[0].shape[-2:], window_size, initial_layers = max_depth, reverse = ShallowDeep , attention = attention, pooling=pooling, convdeconv=convdeconv).to(device),  "ShallowDeep" if ShallowDeep else "DeepShallow")
            model, model_arch = (BidirectionalDeepShallow(input_dim, hidden_dim, kernel_size, num_layers, in_channels, out_channels,  data[0].shape[-2:] ,window_size, initial_layers = max_depth, attention=attention, pooling=pooling, convdeconv=convdeconv).to(device),  "BiDepth")
            # model, model_arch = (AdDepth(input_dim, hidden_dim, kernel_size, num_layers, in_channels, out_channels, data[0].shape[-2:] ,window_size, initial_layers = max_depth , attention = attention, pooling=pooling, convdeconv=convdeconv).to(device), "AdDepth")
            # model, model_arch = (AdDepthRouted(input_dim, hidden_dim, kernel_size, num_layers, in_channels, out_channels, data[0].shape[-2:], window_size, max_depth=max_depth, min_depth=min_depth, attention=attention, pooling=pooling, convdeconv=convdeconv).to(device), "AdDepthRouted")
            criterion = nn.MSELoss()
            optimizer = torch.optim.AdamW(params=model.parameters(), lr=learning_rate)
            scheduler = None

            # Update wandb config
            # if use_wandb:
            #     wandb.config.update({"attention": attention},allow_val_change=True )
            if use_wandb:
                wandb.init(
                    # set the wandb project where this run will be logged
                    project="nyc-taxi" if taxi else "WeatherData" ,

                    # track hyperparameters and run metadata
                    config={
                    "learning_rate": learning_rate,
                    "architecture": model_arch,
                    "dataset": "taxi" if taxi else "Weather",
                    "epochs": num_epochs,
                    "batch_size": batch_size,
                    "channels": channels,
                    "window_size": window_size,
                    "max_depth": max_depth,
                    "pooling": pooling,
                    "convdeconv": convdeconv,
                    "attention" : attention,
                    'hidden_dim' : str(hidden_dim),
                    'model_parameters_count': model_num_params(model),
                    'validation_interval' : validation_interval,
                    "patience" : patience,
                    "loss_spike_threshold": loss_spike_threshold,
                    }
                )

            columns = ["treshhold" , "loss" , "best_layer" , "epoch" ,"time"]
            table = wandb.Table(columns=columns)

            start_time = time.time()

            model, train_losses, val_losses, val_loss_min, epoch  = train_model(
                model,
                criterion,
                optimizer,
                train_loader,
                val_loader,
                test_loader,
                num_epochs,
                validation_interval,
                patience,
                logging,
                device,
                scheduler= scheduler,
                print_losses=True,
                loss_logging = wandb if use_wandb else None,
                checkpoint_path = checkpoint_path,
            )
            best_layer = None

    
            # model, train_losses, val_losses, val_loss_min, best_layer, epoch = train_model_with_RL(
            #     model,
            #     criterion,
            #     optimizer,
            #     train_loader,
            #     val_loader,
            #     test_loader,
            #     num_epochs,
            #     validation_interval,
            #     patience,
            #     q_learning_agent,  # Instance of QLearningAgent
            #     window_size,
            #     max_depth,
            #     initial_depth,
            #     logging,
            #     device,
            #     min_depth = min_depth,
            #     Rl_start_step = Rl_start_step,
            #     current_layers= dynamic_layer_counts,
            #     scheduler= scheduler,
            #     loss_spike_threshold = loss_spike_threshold,
            #     print_losses=True,
            #     loss_logging = wandb if use_wandb else None,
            #     checkpoint_path = checkpoint_path,
            #     max_val_loss = 4,
            # )

            # --- AdDepthRouted training (use with the AdDepthRouted model above) ---
            # model, train_losses, val_losses, val_loss_min, depth_profile, epoch = train_model_routed(
            #     model,
            #     criterion,
            #     optimizer,
            #     train_loader,
            #     val_loader,
            #     test_loader,
            #     num_epochs,
            #     validation_interval,
            #     patience,
            #     logging,
            #     device,
            #     lambda_budget = lambda_budget,
            #     tau_start = tau_start,
            #     tau_end = tau_end,
            #     scheduler = scheduler,
            #     print_losses = True,
            #     loss_logging = wandb if use_wandb else None,
            #     checkpoint_path = checkpoint_path,
            # )
            # best_layer = depth_profile



            end_time = time.time()
            train_time = end_time - start_time

            best_val_loss = val_loss_min
            results.append({
                "loss_spike_threshold": loss_spike_threshold,
                "best_val_loss": best_val_loss,
                "train_time": train_time
            })

            if use_wandb:
                wandb.log({"train_time": train_time})
                wandb.log({"the_loss_spike_threshold": loss_spike_threshold})
                # table.add_data(loss_spike_threshold, best_val_loss , str(best_layer), epoch, train_time)

            print(f"Finished training with loss_spike_threshold: {loss_spike_threshold}")
            print(f"Best validation loss: {best_val_loss}")
            print(f"Training time: {train_time} seconds")
            print("--------------------")

        # Print and log the final results
        print("\nFinal Results:")
        for result in results:
            print(f"Loss Spike Threshold: {result['loss_spike_threshold']}, Best Val Loss: {result['best_val_loss']}, Train Time: {result['train_time']}s")
            logging.info(f"Loss Spike Threshold: {result['loss_spike_threshold']}, Best Val Loss: {result['best_val_loss']}, Train Time: {result['train_time']}s")

        if use_wandb:
            wandb.log({"final_results": str(results)})
            # wandb.log({"final_results_table_new": table})
            wandb.finish()

if __name__ == "__main__":
    main()

