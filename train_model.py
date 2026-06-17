import os #for creating folders, saving files, and handling paths
import json  #use to save the model training history in a readable format
from datetime import datetime #to create a unique folder name using current date and time.
import numpy as np  
import warnings  #suppress TensorFlow logs for cleaner output 
import tensorflow as tf  #model creation, training, and tuning 
import matplotlib.pyplot as plt #for plotting accuracy/loss graphs 
from sklearn.metrics import classification_report #to generate precision, recall, and F1-score for each class
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint

# Suppress TensorFlow and protobuf warnings
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')

#ignores a specific protobuf warning about mismatched runtime versions.
warnings.filterwarnings(
    'ignore',
    message=r'Protobuf gencode version .* is exactly one major version older',
    category=UserWarning,
    module=r'google.protobuf.runtime_version'
)

# Import data and model builder
from data_loader import train_generator, val_generator, class_weight_dict
from model_builder import build_best_model, build_lightweight_model  #FIXED import

# Use manual best parameters
best_hps = {
    "dense_units": 448,
    "dropout_rate1": 0.4,
    "dropout_rate2": 0.3,
    "learning_rate": 0.0005
}

#Step 2: Prepare save directory
save_dir = os.path.join('saved_models', datetime.now().strftime('%Y%m%d_%H%M%S'))
os.makedirs(save_dir, exist_ok=True)

print("Using optimized hyperparameters:")
for k, v in best_hps.items():
    print(f"   - {k}: {v}")

#Step 3: Select model based on system resources
print("\n Checking hardware resources...")
try:
    model = build_best_model()  # Use the function directly
    print("Using optimized model")
except Exception as e:
    print(f"Error loading optimized model: {e}")
    print("Switching to lightweight model")
    model = build_lightweight_model()

print("\n Model Summary:")
model.summary()

#Define callbacks for stability
callbacks = [
    #Prevents model from continuing to learn useless patterns once it starts overfitting.
    EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),

    #Slows down learning when loss plateaus
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=4,
        min_lr=1e-7,
        verbose=1
    ),

    #automatically save the model with the highest validation accuracy during training, ensuring that the best-performing version
    ModelCheckpoint(
        os.path.join(save_dir, 'best_epoch.h5'),
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

# Step 5: Train the model
print("\n Starting training with optimized hyperparameters...")

# trains model with callbacks 
history = model.fit(
    train_generator,
    steps_per_epoch=min(100, len(train_generator)),
    validation_data=val_generator,
    validation_steps=min(50, len(val_generator)),
    epochs=15,
    class_weight=class_weight_dict,
    callbacks=callbacks,
    verbose=1
)

# Step 6: Save final model
final_model_path = os.path.join(save_dir, 'final_model.h5')
model.save(final_model_path)
print(f"\n Training complete. Model saved to: {final_model_path}")

#Step 7: Plot training metrics
plt.figure(figsize=(15, 5))

# Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy', alpha=0.7)
plt.plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
plt.title('Model Accuracy (ResNet50)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)

# Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss', alpha=0.7)
plt.plot(history.history['val_loss'], label='Val Loss', linewidth=2)
plt.title('Model Loss (ResNet50)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plot_path = os.path.join(save_dir, 'training_metrics.png')
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
plt.show()

print(f" Training metrics saved to: {plot_path}")

# Step 8: Evaluate model
print("\n Evaluating model on validation set...")
try:
    y_true = val_generator.classes
    y_pred = model.predict(val_generator, batch_size=16, verbose=1)
    y_pred_classes = np.argmax(y_pred, axis=1)

    np.save(os.path.join(save_dir, 'predictions.npy'), y_pred)
    np.save(os.path.join(save_dir, 'true_labels.npy'), y_true)

    print("\n Classification Report:")
    print(classification_report(
        y_true,
        y_pred_classes,
        target_names=list(val_generator.class_indices.keys()),
        digits=4
    ))

    final_acc = np.mean(y_true == y_pred_classes)
    print(f" Final Validation Accuracy: {final_acc:.4f}")

except Exception as e:
    print(f" Evaluation error: {e}")
    print(" Try smaller batch size if you get OOM errors.")

#Step 9: Save training history
with open(os.path.join(save_dir, 'training_history.json'), 'w') as f:
    json.dump({k: [float(x) for x in v] for k, v in history.history.items()}, f, indent=2)

print("\n All training artifacts saved successfully!")
print(" Model training and evaluation complete.")