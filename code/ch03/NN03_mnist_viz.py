# MNIST Sample Visualization
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os, struct, gzip, urllib.request

# Function to load MNIST manually if torchvision not available
def load_mnist():
    """Load MNIST from local cache or download"""
    base = './data'
    os.makedirs(base, exist_ok=True)
    
    files = {
        'train_images': 'train-images-idx3-ubyte.gz',
        'train_labels': 'train-labels-idx1-ubyte.gz',
    }
    
    local_paths = {}
    for key, fname in files.items():
        path = os.path.join(base, fname)
        if not os.path.exists(path):
            url = f'https://ossci-datasets.s3.amazonaws.com/mnist/{fname}'
            print(f'Downloading {fname}...')
            urllib.request.urlretrieve(url, path)
        local_paths[key] = path
    
    with gzip.open(local_paths['train_images'], 'rb') as f:
        f.read(16)  # header
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(-1, 28, 28)
    
    with gzip.open(local_paths['train_labels'], 'rb') as f:
        f.read(8)  # header
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    
    return images, labels

try:
    images, labels = load_mnist()
    
    fig, axes = plt.subplots(4, 6, figsize=(9, 6))
    for i, ax in enumerate(axes.flat):
        idx = np.where(labels == (i % 10))[0][i // 10] if i // 10 < 11 else i
        ax.imshow(images[idx], cmap='gray')
        ax.set_title(f'Label: {labels[idx]}', fontsize=9)
        ax.axis('off')
    
    plt.suptitle('MNIST Handwritten Digit Samples', fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig('images/ch03/NN03_mnist_samples.png', dpi=150, bbox_inches='tight')
    print("✅ MNIST samples visualization saved")
    
except Exception as e:
    print(f"⚠️ Could not download MNIST: {e}")
    # Fallback: create synthetic digit-like images
    print("Creating synthetic samples instead...")
    fig, axes = plt.subplots(4, 6, figsize=(9, 6))
    for i, ax in enumerate(axes.flat):
        # Create a simple digit-like image
        img = np.random.rand(28, 28) * 0.3
        digit = i % 10
        # Draw a simple pattern
        if digit == 0:
            img[4:24, 4:24] = 0.9
            img[8:20, 8:20] = 0.3
        elif digit == 1:
            img[4:24, 12:16] = 0.9
        else:
            img[4:24, 8:20] = 0.9
        ax.imshow(img, cmap='gray')
        ax.set_title(f'Synthetic: {digit}', fontsize=9)
        ax.axis('off')
    plt.suptitle('MNIST-like Synthetic Samples', fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig('images/ch03/NN03_mnist_samples.png', dpi=150, bbox_inches='tight')
    print("✅ MNIST synthetic samples saved")
