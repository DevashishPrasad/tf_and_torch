from torchsummary import summary

torch.cuda.empty_cache()

model = Res_Type().to(device)
# model = VGG_Type().to(device)

summary(model, (3, 200, 200))
# print(model)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
                             
# Train the model
total_step = len(train_loader)
loss_list = []
acc_list = []
test_loss_list = []
tacc_list = []
ttotal = 1
tcorrect = 0

no_epochs = 80

for epoch in range(no_epochs):
    total = 1
    correct = 0
    for i, (images, labels) in enumerate(train_loader):
        # Put data on GPU
        images = images.to(device)
        labels = labels.to(device)
        
        # Run the forward pass on train 
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss_list.append(loss.item())

        # Backprop and perform Adam optimisation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Run the forward pass on test 
        if (i + 1) % 30 == 0:
          ttotal = 0
          tcorrect = 0
          with torch.no_grad():
            for t, (timages, tlabels) in enumerate(test_loader):
              timages = timages.to(device)
              tlabels = tlabels.to(device)
              test_outputs = model(timages)
              test_loss = criterion(test_outputs, tlabels)
              test_loss_list.append(test_loss.item())
              # Track test the accuracy
              ttotal += tlabels.size(0)
              _, tpredicted = torch.max(test_outputs.data, 1)
              tcorrect += (tpredicted == tlabels).sum().item()
            tacc_list.append(tcorrect / ttotal)

        # Track train the accuracy
        total += labels.size(0)
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum().item()
        acc_list.append(correct / total)

        # Update console
        if (i + 1) % 15 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Train Acc: {:.2f}%, Val Acc : {:.2f}%'
                  .format(epoch + 1, no_epochs, i + 1, total_step, loss.item(),
                          (correct / total) * 100,(tcorrect / ttotal) * 100))
            # Update Log file
            log_file = open("/content/drive/My Drive/Deva/Res_Mini_100e/train_log.txt","a+")
            log_file.write('Epoch[{}/{}],Step[{}/{}],Loss:{:.4f},TrainAcc:{:.2f},ValAcc:{:.2f}\n'.format(epoch + 1, 10, i + 1, total_step, loss.item(),
                          (correct / total) * 100,(tcorrect / ttotal) * 100))
    # Save Model                          
    if((epoch+1)%2==0):
        PATH = '/content/drive/My Drive/Deva/Res_Mini_100e/'+str(epoch+1)+'.pth'
        torch.save(model.state_dict(), PATH)
print("Finished Training")            
log_file.close()
