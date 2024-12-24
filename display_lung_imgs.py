import matplotlib.pyplot as plt
import numpy as np

def load_img(i, filepath):
    data = np.load(filepath)
    train_inp = data['train_images']
    train_sol =data['train_labels']
    classifications ={0:"normal", 1:"pneumonia"}
    img = train_inp[i]

    plt.title(f'28x28 lung #{i}, classification: {train_sol[i]}')
    if('64' in filepath):
        plt.title(f'64x64 lung #{i}, classification: {classifications[train_sol[i][0]]}')

    plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    plt.axis('off')  
    plt.show()

load_img(4,'pneumoniamnist_64.npz')
