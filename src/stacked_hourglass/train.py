import torch
import torch.backends.cudnn
import torch.nn.parallel
from tqdm.auto import tqdm

from stacked_hourglass.loss import joints_mse_loss, kldiv_distill_loss
from stacked_hourglass.utils.evaluation import accuracy, AverageMeter, final_preds
from stacked_hourglass.utils.transforms import fliplr, flip_back


def do_training_step(model, optimiser, input, target, data_info, target_weight=None):
    assert model.training, 'model must be in training mode.'
    assert len(input) == len(target), 'input and target must contain the same number of examples.'

    with torch.enable_grad():
        # Forward pass and loss calculation.
        output, latent, all_block_outs = model(input,return_all_block_outs=True)
        loss = sum(joints_mse_loss(o, target, target_weight) for o in output)
        # KL-Div loss for self-distillation
        if len(latent) >1:
            loss+=kldiv_distill_loss(latent,all_block_outs)

        # Backward pass and parameter update.
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    return output[-1], loss.item()


def do_training_epoch(train_loader, model, device, data_info, optimiser, quiet=False, acc_joints=None):
    losses = AverageMeter()
    accuracies = AverageMeter()

    # Put the model in training mode.
    model.train()

    iterable = enumerate(train_loader)
    progress = None
    if not quiet:
        progress = tqdm(iterable, desc='Train', total=len(train_loader), ascii=True, leave=False)
        iterable = progress

    for i, (input, target, meta) in iterable:
        input, target = input.to(device), target.to(device, non_blocking=True)
        target_weight = meta['target_weight'].to(device, non_blocking=True)

        output, loss = do_training_step(model, optimiser, input, target, data_info, target_weight)

        acc = accuracy(output, target, acc_joints)

        # measure accuracy and record loss
        losses.update(loss, input.size(0))
        accuracies.update(acc[0], input.size(0))

        # Show accuracy and loss as part of the progress bar.
        if progress is not None:
            progress.set_postfix_str('Loss: {loss:0.4f}, Acc: {acc:6.2f}'.format(
                loss=losses.avg,
                acc=100 * accuracies.avg
            ))

    return losses.avg, accuracies.avg


def do_validation_step(model, input, target, data_info, target_weight=None, flip=False):
    assert not model.training, 'model must be in evaluation mode.'
    assert len(input) == len(target), 'input and target must contain the same number of examples.'

    # Forward pass and loss calculation.
    output, latent = model(input)
    loss = sum(joints_mse_loss(o, target, target_weight) for o in output)


    # Get the heatmaps.
    if flip:
        # If `flip` is true, perform horizontally flipped inference as well. This should
        # result in more robust predictions at the expense of additional compute.
        flip_input = fliplr(input)
        flip_output,_ = model(flip_input)
        flip_output = flip_output[-1].cpu()
        flip_output = flip_back(flip_output.detach(), data_info.hflip_indices)
        heatmaps = (output[-1].cpu() + flip_output) / 2
    else:
        heatmaps = output[-1].cpu()  # changed this to evaluate first hour glass

    return heatmaps, loss.item()


def do_validation_epoch(val_loader, model, device, data_info, flip=False, quiet=False, acc_joints=None):
    losses = AverageMeter()
    accuracies = AverageMeter()
    predictions = [None] * len(val_loader.dataset)

    # Put the model in evaluation mode.
    model.eval()

    iterable = enumerate(val_loader)
    progress = None
    if not quiet:
        progress = tqdm(iterable, desc='Valid', total=len(val_loader), ascii=True, leave=False)
        iterable = progress

    for i, (input, target, meta) in iterable:
        # Copy data to the training device (eg GPU).
        input = input.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        target_weight = meta['target_weight'].to(device, non_blocking=True)

        heatmaps, loss = do_validation_step(model, input, target, data_info, target_weight, flip)

        # Calculate PCK from the predicted heatmaps.
        acc = accuracy(heatmaps, target.cpu(), acc_joints)

        # Calculate locations in original image space from the predicted heatmaps.
        preds = final_preds(heatmaps, meta['center'], meta['scale'], [64, 64])
        for example_index, pose in zip(meta['index'], preds):
            predictions[example_index] = pose

        # Record accuracy and loss for this batch.
        losses.update(loss, input.size(0))
        accuracies.update(acc[0].item(), input.size(0))

        # Show accuracy and loss as part of the progress bar.
        if progress is not None:
            progress.set_postfix_str('Loss: {loss:0.4f}, Acc: {acc:6.2f}'.format(
                loss=losses.avg,
                acc=100 * accuracies.avg
            ))

    predictions = torch.stack(predictions, dim=0)

    return losses.avg, accuracies.avg, predictions
