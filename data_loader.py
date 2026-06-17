import os
import warnings

# Reduce TensorFlow logging noise and disable oneDNN optimizations if desired.
# Set these before importing any tensorflow submodules.
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')

# Suppress the protobuf runtime_version UserWarning about gencode mismatch
warnings.filterwarnings(
    'ignore',
    message=r'Protobuf gencode version .* is exactly one major version older',
    category=UserWarning,
    module=r'google.protobuf.runtime_version'
)
import hashlib
import shutil
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input  # Changed to ResNet50 preprocessing
from sklearn.utils.class_weight import compute_class_weight
import json
from collections import Counter

#removes duplicate images 
def remove_duplicates_from_folder(folder_path):
    hashes = set()
    for root, _, files in os.walk(folder_path):
        for fname in files:
            fpath = os.path.join(root, fname)
            with open(fpath, 'rb') as f:
                filehash = hashlib.md5(f.read()).hexdigest()
            if filehash in hashes:
                os.remove(fpath)
                print(f"Removed duplicate: {fpath}")
            else:
                hashes.add(filehash)

remove_duplicates_from_folder('ASL_Alphabet_Dataset/asl_alphabet_train')
remove_duplicates_from_folder('ASL_Alphabet_Dataset/asl_alphabet_test')

for unwanted in ['del', 'space', 'nothing']:
    train_path = os.path.join('ASL_Alphabet_Dataset/asl_alphabet_train', unwanted)
    test_path = os.path.join('ASL_Alphabet_Dataset/asl_alphabet_test', unwanted)
    if os.path.exists(train_path):
        shutil.rmtree(train_path)
        print(f"Removed class folder: {train_path}")
    if os.path.exists(test_path):
        shutil.rmtree(test_path)
        print(f"Removed class folder: {test_path}")

train_dir = 'ASL_Alphabet_Dataset/asl_alphabet_train'
test_dir = 'ASL_Alphabet_Dataset/asl_alphabet_test'

IMG_SIZE = (224, 224)  # ResNet50 default input size
BATCH_SIZE = 16  # Reduced from 32 to prevent memory issues
VAL_SPLIT = 0.2 
SEED = 42
NUM_WORKERS = 2  # Reduced for your RAM constraints

# loads and Data augmentation for ResNet50
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,  # ResNet50 preprocessing
    rotation_range=15,  # Slightly reduced augmentation
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    validation_split=VAL_SPLIT,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=VAL_SPLIT
)

test_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True,
    seed=SEED
)

val_generator = val_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False,
    seed=SEED
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# Compute class weights using generator classes
classes = np.unique(train_generator.classes)
weights = compute_class_weight('balanced', classes=classes, y=train_generator.classes)
class_weight_dict = {int(cls): float(w) for cls, w in zip(classes, weights)}

# Save class indices mapping for inference/serving
class_map_path = os.path.join(os.getcwd(), 'class_indices.json')
with open(class_map_path, 'w') as f:
    json.dump(train_generator.class_indices, f, indent=2)

def get_generators():
    return train_generator, val_generator, test_generator, class_weight_dict

# Class distribution analysis (using generator counts)
inv_map = {v: k for k, v in train_generator.class_indices.items()}
counter = Counter(train_generator.classes)
sorted_indices = sorted(counter.keys())
class_names = [inv_map[i] for i in sorted_indices]
class_counts = [counter[i] for i in sorted_indices]

plt.figure(figsize=(15, 6))
bars = plt.bar(class_names, class_counts)
plt.title('Class Distribution in Training Data', pad=20)
plt.xlabel('ASL Classes')
plt.ylabel('Number of Images')
plt.xticks(rotation=90)
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height, f'{int(height)}', ha='center', va='bottom')
plt.tight_layout()
plt.savefig('class_distribution.png')  # Save instead of show to avoid blocking
plt.close()

# Helper to reverse ResNet50 preprocessing
def deprocess_resnet(img):
    img = img.copy()
    # ResNet50 preprocessing: [0, 255] -> zero-center by mean ImageNet pixel
    # To reverse: add back ImageNet mean values
    mean = [103.939, 116.779, 123.68]
    img[..., 0] += mean[0]
    img[..., 1] += mean[1]
    img[..., 2] += mean[2]
    img = img[..., ::-1]  # BGR to RGB
    return np.clip(img, 0, 255).astype('uint8')

# Sample images visualization
sample_images, sample_labels = next(train_generator)
plt.figure(figsize=(12, 12))
for i in range(min(9, sample_images.shape[0])):
    plt.subplot(3, 3, i+1)
    plt.imshow(deprocess_resnet(sample_images[i]))
    class_idx = int(np.argmax(sample_labels[i]))
    plt.title(f"{class_names[class_idx]} ({class_idx})")
    plt.axis('off')
plt.suptitle('Augmented Training Samples (ResNet50 preprocessed)', y=1.02)
plt.tight_layout()
plt.savefig('training_samples.png')  # Save instead of show
plt.close()

print(f"\n🔍 Dataset Summary:")
print(f"Training samples: {train_generator.samples}")
print(f"Validation samples: {val_generator.samples}")
print(f"Test samples: {test_generator.samples}")
print(f"Number of classes: {len(class_names)}")
print("Class weights for imbalance:", class_weight_dict)
print(f"\n⚙️ ResNet50 Configuration:")
print(f"Batch size: {BATCH_SIZE} (reduced for memory constraints)")
print(f"Image size: {IMG_SIZE}")
print(f"Preprocessing: ResNet50 standard")