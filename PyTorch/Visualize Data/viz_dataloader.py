invTrans = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
                                                     std = [ 1/0.21104454, 1/0.21104453, 1/0.21104452 ]),
                                transforms.Normalize(mean = [ -0.37189576, -0.37189585, -0.37189594 ],
                                                     std = [ 1., 1., 1. ]),
                               ])

# Test data processing
def show_img(img):
    plt.figure(figsize=(18,15))
    # unnormalize
    img = invTrans(img)
    npimg = img.numpy()
    npimg = np.clip(npimg, 0., 1.)
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# show images
for idx,(x_batch,y_batch) in enumerate(dataloaders['train']):
    show_img(torchvision.utils.make_grid(x_batch))
    if(idx>=5):
        break
