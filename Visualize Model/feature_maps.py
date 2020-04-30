import numpy as np
import matplotlib.pyplot as plt

# Visualize feature maps
activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

model = VGG_Type().to(device)
model.load_state_dict(torch.load("/content/drive/My Drive/Deva/VGG_Type_28e/20.pth"))
model.layer1[0].register_forward_hook(get_activation('ext_conv1'))

dataiter = iter(train_loader)
images, labels = dataiter.next()

cvimage = np.array(images[1].view([200, 200, 3]))
image = images[1].view([1,3, 200, 200])

image = image.to(device)
output = model(image)

act = activation['ext_conv1'].squeeze()
act = act.cpu()

fig=plt.figure(figsize=(12, 12))
columns = 5
rows = 5
for i in range(1, columns*rows +1):
    img = act[i]
    fig.add_subplot(rows, columns, i)
    plt.imshow(img)
plt.show()
