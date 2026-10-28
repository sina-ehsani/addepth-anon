# custom weighted loss function:
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import wandb


def _amp_ctx(device, use_amp):
    """bf16 autocast on CUDA when use_amp; otherwise a no-op context.

    bf16 needs no GradScaler and keeps fp32 master weights, so only the
    forward/loss ops run in reduced precision. Reported test MSE
    (`evaluate_mse`) stays in fp32 for comparability.
    """
    if use_amp and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()

# import logging
import matplotlib.pyplot as plt
import os

from .models.AdaptiveDepth import translate_action_to_dynamic_layer_counts, get_initial_state, calculate_reward


def evaluate_mse(model, data_loader, device):
    """Evaluate a model on a loader using PLAIN MSE.

    Reported test numbers must be plain MSE regardless of the (possibly
    weighted) criterion used for training, otherwise they are not comparable
    across models.
    """
    mse_criterion = nn.MSELoss(reduction="sum")
    model.eval()
    total_sq_error = 0.0
    total_elements = 0
    with torch.no_grad():
        for data in data_loader:
            if len(data) == 3:
                inputs1, inputs2, labels = data
                inputs1, inputs2 = inputs1.to(device).float(), inputs2.to(device).float()
            else:
                inputs1, labels = data
                inputs1 = inputs1.to(device).float()
            labels = labels.to(device).float()
            outputs = model(inputs1, inputs2) if len(data) == 3 else model(inputs1)
            total_sq_error += mse_criterion(outputs, labels).item()
            total_elements += labels.numel()
    return total_sq_error / max(total_elements, 1)




def model_num_params(model):
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {total_params}")
    return total_params


class WeightedMSELoss(nn.Module):
    def __init__(self, weight=10.0):
        super(WeightedMSELoss, self).__init__()
        self.weight = weight

    def forward(self, output, target):
        # Create a mask with 'weight' for non-zero values and 1 for zero values
        mask = target.clone()
        mask[mask != 0] = self.weight
        mask[mask == 0] = 1.0

        # Compute the element-wise MSE loss
        loss = (output - target) ** 2

        # Weight the losses
        weighted_loss = loss * mask

        # Return the mean loss
        return weighted_loss.mean()

def plot_losses(train_losses, val_losses, filename='loss_plot.png'):
    plt.figure(figsize=(10, 5))
    plt.plot([x[0] for x in train_losses], [x[1] for x in train_losses], label='Training Loss')
    plt.plot([x[0] for x in val_losses], [x[1] for x in val_losses], label='Validation Loss')
    plt.xlabel('Effective Batch Number')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Losses')
    plt.savefig(filename)
    plt.close()

def train_model(
    model,
    criterion,
    optimizer,
    train_loader,
    val_loader,
    test_loader,
    num_epochs,
    validation_interval,  # Now refers to number of training batches
    patience,
    logging,
    device,
    scheduler=None,
    print_losses=True,
    loss_logging = None,
    checkpoint_path = "checkpoint.pt",
    use_amp=False,
    grad_clip=None,          # max grad-norm (OpenSTL-style); None = off
    sched_per_batch=False,   # step scheduler per batch (OneCycleLR) vs per epoch
):
    train_losses = []
    val_losses = []
    effective_batch_count = 0

    # Initialize the early_stopping object
    early_stopping = EarlyStopping(patience=patience, verbose=print_losses, path=checkpoint_path)

    logging.info('----- NEW TRAINING -------')
    logging.info(f"Number of parameters: {model_num_params(model)}")
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for batch_index, data in enumerate(train_loader):
            effective_batch_count = epoch * len(train_loader) + batch_index

            if len(data) == 3:
                inputs1, inputs2, labels = data
                inputs1, inputs2 = inputs1.to(device).float(), inputs2.to(device).float()
            else:
                inputs1, labels = data
                inputs1 = inputs1.to(device).float()

            labels = labels.to(device).float()
            optimizer.zero_grad()
            with _amp_ctx(device, use_amp):
                outputs = model(inputs1, inputs2) if len(data) == 3 else model(inputs1)
                loss = criterion(outputs, labels)
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            if scheduler is not None and sched_per_batch:
                scheduler.step()

            running_loss += loss.item() * inputs1.size(0)

            if (effective_batch_count + 1) % validation_interval == 0:
                # Log training loss
                epoch_loss = running_loss / validation_interval / inputs1.size(0)
                train_losses.append((effective_batch_count, epoch_loss))
                logging.info(f'Train Loss at batch {effective_batch_count}: {epoch_loss:.4f}')
                if loss_logging:
                    loss_logging.log({"train_loss": epoch_loss, "epoch": epoch, "effective_batch_count": effective_batch_count})

                if print_losses:
                    print(f'Epoch {epoch+1}/{num_epochs}, Effective Batch {effective_batch_count}: Train Loss: {epoch_loss:.4f}')

                # Validate
                model.eval()
                val_running_loss = 0.0

                for val_data in val_loader:
                    if len(val_data) == 3:
                        inputs1, inputs2, labels = val_data
                        inputs1, inputs2 = inputs1.to(device).float(), inputs2.to(device).float()
                    else:
                        inputs1, labels = val_data
                        inputs1 = inputs1.to(device).float()

                    labels = labels.to(device).float()

                    with torch.no_grad(), _amp_ctx(device, use_amp):
                        outputs = model(inputs1, inputs2) if len(val_data) == 3 else model(inputs1)
                        val_loss = criterion(outputs, labels)
                        val_running_loss += val_loss.item() * inputs1.size(0)

                val_epoch_loss = val_running_loss / len(val_loader.dataset)
                val_losses.append((effective_batch_count, val_epoch_loss))
                logging.info(f'Validation Loss at batch {effective_batch_count}: {val_epoch_loss:.4f}')
                if loss_logging:
                    loss_logging.log({"validation_loss": val_epoch_loss, "epoch": epoch, "effective_batch_count": effective_batch_count})


                if print_losses:
                    print(f'Validation Loss: {val_epoch_loss:.4f}')

                early_stopping(val_epoch_loss, model)

                if early_stopping.early_stop:
                    print("Early stopping")
                    break

                model.train()

                # Reset running loss for the next validation interval
                running_loss = 0.0

                # Update and save loss plot
                plot_losses(train_losses, val_losses)

        if scheduler is not None and not sched_per_batch:
            scheduler.step()

        if early_stopping.early_stop:
            break

    # Load the last checkpoint with the best model
    model.load_state_dict(torch.load(checkpoint_path, weights_only=True))

    logging.info(f"Epoch {epoch + 1}/{num_epochs}")
    logging.info(f"Best Validation Loss: {early_stopping.val_loss_min:.4f}")

    if test_loader:
        # Test performance is reported as PLAIN MSE, independent of the
        # training criterion (which may be weighted).
        test_epoch_loss = evaluate_mse(model, test_loader, device)

        logging.info(f'test MSE : {test_epoch_loss:.4f}')
        if loss_logging:
            loss_logging.log({"test_mse": test_epoch_loss})


    return model, train_losses, val_losses, early_stopping.val_loss_min, epoch+1




def train_model_with_RL(
    model,
    criterion,
    optimizer,
    train_loader,
    test_loader,
    val_loader,
    num_epochs,
    validation_interval,
    patience,
    q_learning_agent,  # Instance of QLearningAgent
    window_size,
    max_depth,
    initial_depth,
    logging,
    device,
    min_depth = 1,
    Rl_start_step = 3,
    current_layers = None,
    scheduler=None,
    grad_clip=None,          # unused RL path; kept for signature parity with train_model
    sched_per_batch=False,
    loss_spike_threshold= 1.5,  # New parameter to define what constitutes a "huge spike"
    print_losses=True,
    loss_logging = None,
    checkpoint_path = "checkpoint.pt",
    max_val_loss = 4,
    use_amp=False,
):
    train_losses = []
    val_losses = []
    effective_batch_count = 0


    if current_layers:
        action , _ = get_initial_state(initial_depth, window_size)
    else: 
        action, current_layers = get_initial_state(initial_depth, window_size)

    # Initialize the early_stopping object
    early_stopping = EarlyStopping(patience=patience, verbose=print_losses, path = checkpoint_path)

    logging.info('----- NEW TRAINING - RL -------')
    logging.info(f"Number of parameters: {model_num_params(model)}")

    for epoch in tqdm(range(num_epochs), desc="Epoch"):

        # training phase
        model.train()
        running_loss = 0.0

        for batch_index, data in enumerate(train_loader):
            # Update your model's dynamic_layer_counts here
            model.update_dynamic_layer_counts(current_layers)
            
            effective_batch_count = epoch * len(train_loader) + batch_index

            # check if the loader provides two inputs or just one
            if len(data) == 3:
                inputs1, inputs2, labels = data
                inputs1, inputs2 = inputs1.to(device).float(), inputs2.to(device).float()
            else:
                inputs1, labels = data
                inputs1 = inputs1.to(device).float()

            labels = labels.to(device).float()
            optimizer.zero_grad()
            with _amp_ctx(device, use_amp):
                outputs = model(inputs1, inputs2) if len(data) == 3 else model(inputs1)
                loss = criterion(outputs, labels)
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            if scheduler is not None and sched_per_batch:
                scheduler.step()

            running_loss += loss.item() * inputs1.size(0)

            if (effective_batch_count + 1) % validation_interval == 0:

                if loss_logging:
                    loss_logging.log({"current_layers": current_layers, "epoch": epoch, "effective_batch_count": effective_batch_count})
                logging.info(f'current_layers: {current_layers}')


                # Log training loss
                epoch_loss = running_loss / validation_interval / inputs1.size(0)
                train_losses.append((effective_batch_count, epoch_loss))
                logging.info(f'Train Loss at batch {effective_batch_count}: {epoch_loss:.4f}')
                if loss_logging:
                    loss_logging.log({"train_loss": epoch_loss, "epoch": epoch, "effective_batch_count": effective_batch_count})

                if print_losses:
                    print(f'Epoch {epoch+1}/{num_epochs}, Effective Batch {effective_batch_count}: Train Loss: {epoch_loss:.4f}')

                # Validate
                model.eval()
                val_running_loss = 0.0

                for val_data in val_loader:
                    if len(val_data) == 3:
                        inputs1, inputs2, labels = val_data
                        inputs1, inputs2 = inputs1.to(device).float(), inputs2.to(device).float()
                    else:
                        inputs1, labels = val_data
                        inputs1 = inputs1.to(device).float()

                    labels = labels.to(device).float()
                    with torch.no_grad(), _amp_ctx(device, use_amp):
                        outputs = model(inputs1, inputs2) if len(val_data) == 3 else model(inputs1)
                        val_loss = criterion(outputs, labels)
                        val_running_loss += val_loss.item() * inputs1.size(0)

                val_epoch_loss = val_running_loss / len(val_loader.dataset)
                val_losses.append((effective_batch_count, val_epoch_loss))
                logging.info(f'Validation Loss at batch {effective_batch_count}: {val_epoch_loss:.4f}')
                if loss_logging:
                    loss_logging.log({"validation_loss": val_epoch_loss, "epoch": epoch, "effective_batch_count": effective_batch_count})
                    
                if print_losses:
                    print(f'Validation Loss: {val_epoch_loss:.4f}')

                early_stopping(val_epoch_loss, model, current_layers)

                if early_stopping.early_stop:
                    print("Early stopping")
                    break

                model.train()

                # Reset running loss for the next validation interval
                running_loss = 0.0

                # Update and save loss plot
                plot_losses(train_losses, val_losses)

                
                if len(val_losses) >= max(2, Rl_start_step):
                    old_loss = val_losses[-2][1]
                    new_loss = val_losses[-1][1]
                    reward = calculate_reward(val_losses)

                    # Credit the PREVIOUS action (if one was taken) for the
                    # observed loss change. After a rollback, action is None
                    # and no Q-update is performed — the reverted
                    # configuration's loss change must not be attributed to
                    # the action that was undone.
                    if action is not None:
                        q_learning_agent.learn(old_loss, action, reward, new_loss)

                    # Check for a huge spike in validation loss
                    if new_loss > old_loss * loss_spike_threshold and q_learning_agent.previous_layers is not None:
                        logging.info("Huge spike in validation loss detected. Reverting to previous layers.")
                        # Revert to the previous layers; no new action this step.
                        current_layers = q_learning_agent.previous_layers.copy()
                        action = None
                        logging.info(f'Reverted to previous layers: {current_layers}')
                    else:
                        # Choose and apply the next action.
                        action = q_learning_agent.choose_action(new_loss)
                        # Store the current layers before making any changes
                        q_learning_agent.store_previous_layers(current_layers)
                        current_layers = translate_action_to_dynamic_layer_counts(action, current_layers, min_depth=min_depth, max_depth=max_depth, window_size=window_size)

                        logging.info(f'action: {action - window_size} -->  {current_layers} ')
                        if loss_logging:
                            loss_logging.log({"Action": action - window_size, "epoch": epoch, "effective_batch_count": effective_batch_count})


        if scheduler is not None and not sched_per_batch:
            scheduler.step()

        if early_stopping.early_stop:
            break

    # Load the last checkpoint with the best model
    model.update_dynamic_layer_counts(early_stopping.best_layers)
    model.load_state_dict(torch.load(checkpoint_path, weights_only=True))

    logging.info(f"Epoch {epoch + 1}/{num_epochs}")
    logging.info(f"Best Validation Loss: {early_stopping.val_loss_min:.4f}")

    if test_loader:
        # Test performance is reported as PLAIN MSE, independent of the
        # training criterion (which may be weighted).
        test_epoch_loss = evaluate_mse(model, test_loader, device)

        logging.info(f'test MSE : {test_epoch_loss:.4f}')
        if loss_logging:
            loss_logging.log({"test_mse": test_epoch_loss})


    if loss_logging:
        loss_logging.log({"Stopping_Epoch": epoch+1 })

    if not print_losses:
        print("Epoch {}/{}".format(epoch + 1, num_epochs))
        print("Best Validation Loss: {:.4f}".format(early_stopping.val_loss_min))
        if test_loader:
            print("Test Loss: {:.4f}".format(test_epoch_loss))


    return model, train_losses, val_losses, early_stopping.val_loss_min, early_stopping.best_layers, epoch+1



def compute_depth_profile(model, data_loader, device):
    """Mean selected depth per time-step position over a loader (eval mode).

    Returns a list of floats of length `window`. With the model in eval mode
    routing is hard (argmax), so this is the actual depth used per position,
    averaged over inputs — directly comparable to the learned-schedule tables
    in the paper, but now input-conditioned.
    """
    model.eval()
    depth_sum = None
    count = 0
    with torch.no_grad():
        for data in data_loader:
            if len(data) == 3:
                inputs1, inputs2, _ = data
                inputs1, inputs2 = inputs1.to(device).float(), inputs2.to(device).float()
                model(inputs1, inputs2)
            else:
                inputs1, _ = data
                inputs1 = inputs1.to(device).float()
                model(inputs1)
            d = model.expected_depth()          # (b, window)
            depth_sum = d.sum(dim=0) if depth_sum is None else depth_sum + d.sum(dim=0)
            count += d.size(0)
    return (depth_sum / max(count, 1)).tolist()


def train_model_routed(
    model,                      # AdDepthRouted
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
    lambda_budget=0.01,         # weight of the compute-budget penalty
    tau_start=2.0,              # initial Gumbel-softmax temperature
    tau_end=0.5,                # final temperature (annealed exponentially)
    entropy_weight=0.0,         # bonus on router entropy (anti-collapse)
    router_warmup_epochs=0,     # epochs of uniform routing before the router engages
    scheduler=None,
    grad_clip=None,             # max grad-norm (OpenSTL-style); None = off
    sched_per_batch=False,      # step scheduler per batch (OneCycleLR) vs per epoch
    print_losses=True,
    loss_logging=None,
    checkpoint_path="checkpoint.pt",
    use_amp=False,
):
    """Joint training of AdDepthRouted: task loss + lambda * depth budget.

    No outer-loop RL: the router is trained end-to-end by backprop. Early
    stopping and checkpoint selection use the TASK validation loss only (the
    budget term is a training regularizer, not part of model quality).
    Validation runs in eval mode, i.e. with hard argmax routing, so the
    early-stopping signal reflects actual inference behavior.

    Returns (model, train_losses, val_losses, best_val_loss, depth_profile,
    last_epoch) where depth_profile is the mean selected depth per time-step
    position on the validation set under the best checkpoint.
    """
    train_losses = []
    val_losses = []
    effective_batch_count = 0

    early_stopping = EarlyStopping(patience=patience, verbose=print_losses, path=checkpoint_path)

    logging.info('----- NEW TRAINING - ROUTED -------')
    logging.info(f"Number of parameters: {model_num_params(model)}")
    logging.info(f"lambda_budget={lambda_budget}, tau: {tau_start} -> {tau_end}")

    for epoch in tqdm(range(num_epochs), desc="Epoch"):
        # Uniform-routing warmup: every depth's layers receive equal gradient
        # before the router starts choosing (prevents rich-get-richer collapse
        # onto whichever depth trains fastest).
        warmup = epoch < router_warmup_epochs
        model.set_router_warmup(warmup)

        # Exponential temperature annealing over the post-warmup epochs.
        anneal_t = max(0, epoch - router_warmup_epochs)
        anneal_T = max(1, num_epochs - router_warmup_epochs - 1)
        tau = tau_start * (tau_end / tau_start) ** (anneal_t / anneal_T)
        model.set_temperature(tau)

        model.train()
        running_loss = 0.0

        for batch_index, data in enumerate(train_loader):
            effective_batch_count = epoch * len(train_loader) + batch_index

            if len(data) == 3:
                inputs1, inputs2, labels = data
                inputs1, inputs2 = inputs1.to(device).float(), inputs2.to(device).float()
            else:
                inputs1, labels = data
                inputs1 = inputs1.to(device).float()

            labels = labels.to(device).float()
            optimizer.zero_grad()
            with _amp_ctx(device, use_amp):
                outputs = model(inputs1, inputs2) if len(data) == 3 else model(inputs1)
                task_loss = criterion(outputs, labels)
            if warmup:
                # Uniform routing: budget/entropy terms carry no router
                # gradient, so train on the task loss alone.
                loss = task_loss
            else:
                # Budget/entropy regularizers stay in fp32 (outside autocast).
                loss = task_loss + lambda_budget * model.depth_budget()
                if entropy_weight > 0:
                    loss = loss - entropy_weight * model.router_entropy()
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            if scheduler is not None and sched_per_batch:
                scheduler.step()

            # Track the TASK loss so train curves are comparable across
            # lambda_budget settings.
            running_loss += task_loss.item() * inputs1.size(0)

            if (effective_batch_count + 1) % validation_interval == 0:
                epoch_loss = running_loss / validation_interval / inputs1.size(0)
                train_losses.append((effective_batch_count, epoch_loss))
                logging.info(f'Train Loss at batch {effective_batch_count}: {epoch_loss:.4f}')
                if loss_logging:
                    loss_logging.log({"train_loss": epoch_loss, "tau": tau, "epoch": epoch,
                                      "effective_batch_count": effective_batch_count})

                if print_losses:
                    print(f'Epoch {epoch+1}/{num_epochs}, Effective Batch {effective_batch_count}: Train Loss: {epoch_loss:.4f}')

                # Validate with hard routing (eval mode) + collect depth usage.
                model.eval()
                val_running_loss = 0.0
                depth_sum = None
                depth_count = 0

                for val_data in val_loader:
                    if len(val_data) == 3:
                        inputs1, inputs2, labels = val_data
                        inputs1, inputs2 = inputs1.to(device).float(), inputs2.to(device).float()
                    else:
                        inputs1, labels = val_data
                        inputs1 = inputs1.to(device).float()

                    labels = labels.to(device).float()
                    with torch.no_grad(), _amp_ctx(device, use_amp):
                        outputs = model(inputs1, inputs2) if len(val_data) == 3 else model(inputs1)
                        val_loss = criterion(outputs, labels)
                        val_running_loss += val_loss.item() * inputs1.size(0)
                        d = model.expected_depth()  # hard routing -> selected depths
                        depth_sum = d.sum(dim=0) if depth_sum is None else depth_sum + d.sum(dim=0)
                        depth_count += d.size(0)

                val_epoch_loss = val_running_loss / len(val_loader.dataset)
                val_losses.append((effective_batch_count, val_epoch_loss))
                depth_profile = (depth_sum / max(depth_count, 1)).tolist()

                logging.info(f'Validation Loss at batch {effective_batch_count}: {val_epoch_loss:.4f}')
                logging.info(f'Mean selected depth per position: {[round(v, 2) for v in depth_profile]}')
                if loss_logging:
                    log_dict = {"validation_loss": val_epoch_loss, "epoch": epoch,
                                "effective_batch_count": effective_batch_count,
                                "mean_depth": sum(depth_profile) / len(depth_profile),
                                "router_entropy": float(model.router_entropy()),
                                "router_warmup": warmup}
                    for pos, dval in enumerate(depth_profile):
                        log_dict[f"depth_pos_{pos}"] = dval
                    loss_logging.log(log_dict)

                if print_losses:
                    print(f'Validation Loss: {val_epoch_loss:.4f} | depths: {[round(v, 1) for v in depth_profile]}')

                early_stopping(val_epoch_loss, model)

                if early_stopping.early_stop:
                    print("Early stopping")
                    break

                model.train()
                running_loss = 0.0
                plot_losses(train_losses, val_losses)

        if scheduler is not None and not sched_per_batch:
            scheduler.step()

        if early_stopping.early_stop:
            break

    # Load the best checkpoint, then report its depth profile and test MSE.
    model.load_state_dict(torch.load(checkpoint_path, weights_only=True))

    logging.info(f"Epoch {epoch + 1}/{num_epochs}")
    logging.info(f"Best Validation Loss: {early_stopping.val_loss_min:.4f}")

    depth_profile = compute_depth_profile(model, val_loader, device)
    logging.info(f'Best-model depth profile (val set): {[round(v, 2) for v in depth_profile]}')
    if loss_logging:
        loss_logging.log({"best_depth_profile": depth_profile})

    if test_loader:
        # Test performance is reported as PLAIN MSE, independent of the
        # training criterion (which may be weighted).
        test_epoch_loss = evaluate_mse(model, test_loader, device)
        logging.info(f'test MSE : {test_epoch_loss:.4f}')
        if loss_logging:
            loss_logging.log({"test_mse": test_epoch_loss})

    return model, train_losses, val_losses, early_stopping.val_loss_min, depth_profile, epoch + 1


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=0, path="checkpoint.pt", trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        self.best_layers = None

    def __call__(self, val_loss, model, current_layers = None):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.best_layers = current_layers
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                self.trace_func(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_layers = current_layers
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            self.trace_func(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
            )
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss