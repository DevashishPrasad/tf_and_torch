def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def train_model(model, dataloaders, criterion, optimizer, num_epochs=25):
    since = time.time()

    val_acc_history = []
    train_acc_history = []

    # best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('=' * 150)
        ep_time = time.time()

        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for ien, (inputs, labels) in enumerate(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / len(dataloaders[phase].dataset)
                epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

                if(ien%5==0):
                  lr = get_lr(optimizer)
                  print('Phase[{}] Epoch {}/{} Iteration {}/{} :\t Epoch Loss: {:.10f} Accuracy: {:.10f} Learning Rate : {:.10f}'.format(phase, epoch, num_epochs - 1, ien, int(len(dataloaders[phase].dataset)/b_size),epoch_loss,epoch_acc,lr))
                if(ien%10==0 and phase == 'train'):
                  my_lr_scheduler.step()

            print('\n\n*********** Phase[{}] Epoch: {}/{} \t Epoch Acc: {:.10f} \t Epoch Loss: {:.10f} ***********\n\n'.format(phase, epoch, num_epochs - 1, epoch_acc, epoch_loss))

            # epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            # print(' {} Loss: {:.4f} '.format(phase, epoch_loss))

            # deep copy the model
            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                # best_model_wts = copy.deepcopy(model.state_dict())
                # best_model_optm = copy.deepcopy(optimizer.state_dict())
                print("SAVING THE MODEL")
                # Save the best model
                torch.save(model, "/content/drive/My Drive/Orbit Shifters/trunk data/better_trunk_classifier.pth")
            if phase == 'test':
                val_acc_history.append(epoch_acc)
            if phase == 'train':
                train_acc_history.append(epoch_acc)

        est_time = ((time.time() - ep_time) / 60) * (num_epochs - epoch)
        print(" Estimated time remaining : {:.2f}m".format(est_time))

    time_elapsed = (time.time() - since)
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best test acc: {:4f}'.format(best_acc))

    # load best model weights
    # model.load_state_dict(best_model_wts)
    # optimizer.load_state_dict(best_model_optm)

    # Save the best model
    # torch.save(model, "/content/drive/My Drive/Orbit Shifters/trunk data/trunk_classifier2.pth")

    return model, val_acc_history, train_acc_history
