import time
import torch
from biotorch.training.metrics import accuracy, ProgressMeter, AverageMeter


def train(model,
          mode,
          loss_function,
          optimizer,
          train_dataloader,
          device,
          epoch,
          multi_gpu,
          top_k=5,
          display_iterations=500):

    # Create Metrics
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    topk = AverageMeter('Acc@' + str(top_k), ':6.2f')
    progress = ProgressMeter(
        len(train_dataloader),
        [batch_time, data_time, losses, top1, topk],
        prefix="Epoch: [{}]".format(epoch))

    # Switch mode
    model.train()
    end = time.time()
    use_amp = True
    if use_amp: assert not model.layer_config['options']['gradient_clip'], 'Layer-wise gradient clipping not supported with AMP'
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    for idx_batch, (inputs, targets) in enumerate(train_dataloader):
        # Measure data loading time
        data_time.update(time.time() - end)
        with torch.autocast(device_type=device[:4], dtype=torch.float16, enabled=use_amp): #device[:4] is 'cuda'
            # Send inputs to device
            inputs, targets = inputs.to(device), targets.to(device)
            # change target to long if is int
            if targets.dtype == torch.int64 or targets.dtype == torch.int32:
                targets = targets.long()
            # Get outputs from the model
            if mode == 'dfa':
                outputs = model(inputs, targets, loss_function)
            else:
                outputs = model(inputs)

            # Calculate loss
            outputs = torch.squeeze(outputs)

            loss = loss_function(outputs, targets)

        # Zero gradients
        optimizer.zero_grad()
        # Backward Pass
        scaler.scale(loss).backward() #loss.backward()
        # Unscales the gradients of optimizer's assigned parameters in-place
        scaler.unscale_(optimizer)
        # Since the gradients of optimizer's assigned parameters are now unscaled, clips as usual.
        ## torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=0.1)
        # Update weights
        scaler.step(optimizer) #optimizer.step()
        # Updates the scale for next iteration.
        scaler.update()

        # Measure accuracy and record loss
        acc1, acck = accuracy(outputs, targets, topk=(1, top_k))
        losses.update(loss.item(), inputs.size(0))
        top1.update(acc1[0], inputs.size(0))
        topk.update(acck[0], inputs.size(0))

        if mode == 'weight_mirroring':
            raise ValueError('checking AMP')
            if multi_gpu:
                model.module.mirror_weights(torch.randn(inputs.size()).to(device),
                                            growth_control=True)
            else:
                model.mirror_weights(torch.randn(inputs.size()).to(device),
                                     growth_control=True)

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if idx_batch % display_iterations == 0:
            progress.display(idx_batch)

    return top1.avg, losses.avg


def test(model,
         loss_function,
         test_dataloader,
         device,
         top_k=5,
         ):

    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    topk = AverageMeter('Acc@' + str(top_k), ':6.2f')

    # Switch to evaluate mode
    model.eval()
    use_amp = True
    # Deactivate the autograd engine in test
    with torch.no_grad():
        end = time.time()
        for idx_batch, (data, target) in enumerate(test_dataloader):
            with torch.autocast(device_type=device[:4], dtype=torch.float16, enabled=use_amp): #device[:4] is 'cuda'
                inputs, targets = data.to(device), target.to(device)
                # change target to long if is int
                if targets.dtype == torch.int64 or targets.dtype == torch.int32:
                    targets = targets.long()
                outputs = model(inputs)
                outputs = torch.squeeze(outputs)
                # Measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()
                # Compute loss function
                loss = loss_function(outputs, targets)
            # Measure accuracy and record loss
            acc1, acc5 = accuracy(outputs, targets, topk=(1, top_k))
            losses.update(loss.item(), inputs.size(0))
            top1.update(acc1[0], inputs.size(0))
            topk.update(acc5[0], inputs.size(0))

        print(' * Acc@1 {top1.avg:.3f} Acc@{top_k} {topk.avg:.3f}'.format(top1=top1, top_k=top_k, topk=topk))
    return top1.avg, losses.avg


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
