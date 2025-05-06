import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.api.optimizers import Adam
from keras.api.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from keras.api.losses import MeanSquaredError
from keras.api.metrics import MeanAbsoluteError

# Add the parent directory to the path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
# print(sys.path[-1])
from utils.epoch_data_generator import EpochDataGenerator, OutputType
from models.ESTformer import ESTFormer, reconstruction_loss

# Define available channels
lr_channel_names = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']
hr_channel_names = ['Fp1', 'Fz', 'F3', 'F7', 'FT9', 'FC5', 'FC1', 'C3', 'T7', 'TP9', 'CP5', 'CP1', 'Pz', 'P3', 'P7', 'O1', 'Oz', 'O2', 'P4', 'P8', 'TP10', 'CP6', 'CP2', 'C4', 'T8', 'FT10', 'FC6', 'FC2', 'F4', 'F8', 'Fp2', 'AF7', 'AF3', 'AFz', 'F1', 'F5', 'FT7', 'FC3', 'C1', 'C5', 'TP7', 'CP3', 'P1', 'P5', 'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'P6', 'P2', 'CPz', 'CP4', 'TP8', 'C6', 'C2', 'FC4', 'FT8', 'F6', 'AF8', 'AF4', 'F2', 'FCz']

# Create a custom model class to incorporate the specialized loss function
class ESTFormerTrainer(tf.keras.Model):
    def __init__(self, estformer_model):
        super(ESTFormerTrainer, self).__init__()
        self.estformer = estformer_model
        # Initialize learnable parameters for the loss function
        self.sigma1 = tf.Variable(1.0, dtype=tf.float32, trainable=True, name="sigma1")
        self.sigma2 = tf.Variable(1.0, dtype=tf.float32, trainable=True, name="sigma2")
        
    def call(self, inputs, training=None):
        return self.estformer(inputs)
    
    def train_step(self, data):
        x, y = data
        
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = reconstruction_loss(y, y_pred, self.sigma1, self.sigma2)
            
        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        
        # Compute metrics
        mse = tf.reduce_mean(tf.square(y - y_pred))
        mae = tf.reduce_mean(tf.abs(y - y_pred))
        
        # Return metrics
        return {
            "loss": loss, 
            "mse": mse, 
            "mae": mae, 
            "sigma1": self.sigma1, 
            "sigma2": self.sigma2
        }
    
    def test_step(self, data):
        x, y = data
        y_pred = self(x, training=False)
        loss = reconstruction_loss(y, y_pred, self.sigma1, self.sigma2)
        
        # Compute metrics
        mse = tf.reduce_mean(tf.square(y - y_pred))
        mae = tf.reduce_mean(tf.abs(y - y_pred))
        
        # Return metrics
        return {
            "loss": loss, 
            "mse": mse, 
            "mae": mae, 
            "sigma1": self.sigma1, 
            "sigma2": self.sigma2
        }

# Set up data generators
def setup_data_generators(batch_size=32):
    """Set up training, validation, and test data generators"""
    participants_file = os.path.join(EpochDataGenerator.data_dir(), "participants.tsv")
    participants_df = pd.read_csv(participants_file, sep="\t")
    subjects = participants_df.loc[participants_df["exclude"] == 0, "participant_id"].to_list()
    
    # Create subject-epoch coordinates
    subject_epoch_coordinates = [[subject, coordinate] for subject in subjects for coordinate in range(24648) if subject not in ["sub-49", "sub-50"]]
    
    # Split into train, validation, and test sets
    train_coordinates, test_coordinates = train_test_split(subject_epoch_coordinates, test_size=0.3, random_state=42)
    test_coordinates, validation_coordinates = train_test_split(test_coordinates, test_size=0.66, random_state=42)
    
    # Create data generators with super resolution output type
    train_generator = EpochDataGenerator(
        train_coordinates, 
        lr_channel_names=lr_channel_names, 
        hr_channel_names=hr_channel_names,
        output_type=OutputType.SUPER_RESOLUTION
    )
    
    validation_generator = EpochDataGenerator(
        validation_coordinates, 
        lr_channel_names=lr_channel_names, 
        hr_channel_names=hr_channel_names,
        output_type=OutputType.SUPER_RESOLUTION
    )
    
    test_generator = EpochDataGenerator(
        test_coordinates, 
        lr_channel_names=lr_channel_names, 
        hr_channel_names=hr_channel_names,
        output_type=OutputType.SUPER_RESOLUTION
    )
    
    # Set up tf.data.Dataset with batching for efficient training
    def generator_to_dataset(generator, batch_size):
        def gen():
            for i in range(len(generator)):
                yield generator[i]
        
        output_signature = (
            tf.TensorSpec(shape=(len(lr_channel_names), 53), dtype=np.float32),
            tf.TensorSpec(shape=(len(hr_channel_names), 53), dtype=np.float32)
        )
        
        dataset = tf.data.Dataset.from_generator(gen, output_signature=output_signature)
        dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return dataset
    
    train_dataset = generator_to_dataset(train_generator, batch_size)
    val_dataset = generator_to_dataset(validation_generator, batch_size)
    test_dataset = generator_to_dataset(test_generator, batch_size)
    
    return train_dataset, val_dataset, test_dataset, train_generator, validation_generator, test_generator

# Set up the model with optimal parameters
def create_model():
    """Create the ESTFormer model"""
    # Model hyperparameters
    time_steps = 53
    d_model = 60
    num_heads = 4
    mlp_dim = 128
    dropout_rate = 0.1
    Ls = 1  # Number of spatial layers
    Lt = 1  # Number of temporal layers
    builtin_montage = 'standard_1020'
    
    # Create the base model
    base_model = ESTFormer(
        lr_channel_names=lr_channel_names,
        hr_channel_names=hr_channel_names,
        builtin_montage=builtin_montage,
        time_steps=time_steps,
        d_model=d_model,
        num_heads=num_heads,
        mlp_dim=mlp_dim,
        dropout_rate=dropout_rate,
        Ls=Ls,
        Lt=Lt
    )
    
    # Create the trainer model with custom loss
    model = ESTFormerTrainer(base_model)
    
    # Compile the model
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer)
    
    return model

def train_model(model, train_dataset, val_dataset, epochs=100, model_dir='models'):
    """Train the model and save checkpoints"""
    # Create model directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    
    # Set up callbacks
    checkpoint_path = os.path.join(model_dir, 'estformer_best_model.h5')
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=checkpoint_path,
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        TensorBoard(
            log_dir=os.path.join(model_dir, 'logs'),
            histogram_freq=1
        )
    ]
    
    # Train the model
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    return history, model

def evaluate_model(model, test_dataset):
    """Evaluate the trained model on the test dataset"""
    results = model.evaluate(test_dataset, verbose=1)
    print(f"Test results: {results}")
    return results

def save_model(model, model_path):
    """Save the trained model to disk"""
    # Save the base ESTFormer model (more useful than the trainer)
    model.estformer.save(model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    # Set batch size
    BATCH_SIZE = 32
    EPOCHS = 100
    MODEL_DIR = "models"
    
    # Set up data generators and datasets
    train_dataset, val_dataset, test_dataset, train_gen, val_gen, test_gen = setup_data_generators(BATCH_SIZE)
    
    # Print shape information
    print(f"Training dataset input shape: {train_gen.get_input_shape()}")
    print(f"Training dataset output shape: {train_gen.get_output_shape()}")
    
    # Create and summarize model
    model = create_model()
    model.build(input_shape=(None, len(lr_channel_names), 53))
    model.summary()
    
    # Train model
    history, trained_model = train_model(model, train_dataset, val_dataset, epochs=EPOCHS, model_dir=MODEL_DIR)
    
    # Evaluate model
    eval_results = evaluate_model(trained_model, test_dataset)
    
    # Save the trained model
    save_model(trained_model, os.path.join(MODEL_DIR, "estformer_final_model.h5"))
    
    print("Training completed successfully!")