import os
import json
from datetime import datetime  #data handling, saving results
import warnings
from tensorflow import keras
import keras_tuner as kt #model creation, training, and tuning 
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')

# Option 1: Simple version with fixed best parameters (after tuning)
def build_best_model():
    base_model = ResNet50(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.4)(x)  # Best dropout_rate
    x = Dense(448, activation='relu')(x)  # Best dense_units
    x = Dropout(0.3)(x)  # Best dropout2
    outputs = Dense(26, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=outputs)
    model.compile(
        optimizer=Adam(learning_rate=0.0005),  
        loss='categorical_crossentropy',
        metrics=['accuracy', 'top_k_categorical_accuracy']
    )
    return model

# Option 2:Automates tuning 
class HyperparameterTuner(kt.HyperModel):
    def __init__(self, input_shape=(224, 224, 3), num_classes=26):
        self.input_shape = input_shape
        self.num_classes = num_classes
    
    def build(self, hp):
        # Tune these parameters
        learning_rate = hp.Float('learning_rate', min_value=1e-5, max_value=1e-2, sampling='log')
        dropout_rate1 = hp.Float('dropout_rate1', min_value=0.2, max_value=0.7, step=0.1)
        dropout_rate2 = hp.Float('dropout_rate2', min_value=0.1, max_value=0.6, step=0.1)
        dense_units = hp.Choice('dense_units', values=[64, 128, 256, 512, 1024])
        
        base_model = ResNet50(
            input_shape=self.input_shape,
            include_top=False,
            weights='imagenet'
        )
        base_model.trainable = False

        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dropout(dropout_rate1)(x)
        x = Dense(dense_units, activation='relu')(x)
        x = Dropout(dropout_rate2)(x)
        outputs = Dense(self.num_classes, activation='softmax')(x)

        model = Model(inputs=base_model.input, outputs=outputs)
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'top_k_categorical_accuracy']
        )
        return model

# Function to run tuning RandomSearch
def run_tuning(train_generator, val_generator, max_epochs=5, trials=10):
    tuner = kt.RandomSearch(
        HyperparameterTuner(),
        objective='val_accuracy',
        max_trials=trials,
        directory='tuning_results',
        project_name='asl_tuning'
    )
    
    tuner.search(
        train_generator,
        steps_per_epoch=min(30, len(train_generator)),
        validation_data=val_generator,
        validation_steps=min(15, len(val_generator)),
        epochs=max_epochs,
        verbose=1
    )
    
    # Get best parameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    
    print(" BEST PARAMETERS FOUND:")
    print(f"Learning Rate: {best_hps.get('learning_rate')}")
    print(f"Dropout Rate 1: {best_hps.get('dropout_rate1')}")
    print(f"Dropout Rate 2: {best_hps.get('dropout_rate2')}")
    print(f"Dense Units: {best_hps.get('dense_units')}")
    
    return best_hps