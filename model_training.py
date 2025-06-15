import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras.callbacks import (EarlyStopping, 
                                      ReduceLROnPlateau, 
                                      ModelCheckpoint,
                                      CSVLogger)
from sklearn.metrics import classification_report, confusion_matrix
import argparse

def setup_directories():
    """Create required directories if they don't exist"""
    os.makedirs('models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    os.makedirs('plots', exist_ok=True)

def verify_dataset_path(data_dir):
    """Verify dataset directory exists and has proper structure"""
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Dataset directory not found at: {os.path.abspath(data_dir)}")
    
    subdirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    if not subdirs:
        raise ValueError(f"No class directories found in {data_dir}")
    
    print(f"Found {len(subdirs)} classes in dataset directory")
    return len(subdirs)

def create_data_generators(data_dir, image_size, batch_size):
    """Create train and validation data generators with augmentation"""
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='reflect',
        validation_split=0.2
    )

    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )

    val_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )

    return train_generator, val_generator

def build_model(input_shape, num_classes):
    """Build EfficientNetB4 based model with transfer learning"""
    base_model = EfficientNetB4(
        input_shape=input_shape,
        weights='imagenet',
        include_top=False
    )
    
    base_model.trainable = False
    
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(1024, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy', 
                tf.keras.metrics.AUC(name='auc'),
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall')]
    )
    
    return model

def fine_tune_model(model):
    """Unfreeze top layers for fine-tuning"""
    model.layers[0].trainable = True
    for layer in model.layers[0].layers[:-20]:
        layer.trainable = False
    
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy', 
                tf.keras.metrics.AUC(name='auc'),
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall')]
    )
    return model

def train_model(model, train_gen, val_gen, epochs, callbacks):
    """Train the model with callbacks"""
    history = model.fit(
        train_gen,
        steps_per_epoch=train_gen.samples // train_gen.batch_size,
        validation_data=val_gen,
        validation_steps=val_gen.samples // val_gen.batch_size,
        epochs=epochs,
        callbacks=callbacks
    )
    return history

def evaluate_model(model, val_gen, class_names):
    """Evaluate model and generate reports"""
    # Evaluate metrics
    results = model.evaluate(val_gen)
    print("\nEvaluation Metrics:")
    for name, value in zip(model.metrics_names, results):
        print(f"{name}: {value:.4f}")

    # Classification report
    y_pred = model.predict(val_gen)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = val_gen.classes
    
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred_classes, target_names=class_names))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred_classes)
    plt.figure(figsize=(15, 15))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=90)
    plt.yticks(tick_marks, class_names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('plots/confusion_matrix.png')
    plt.close()

def plot_training_history(history):
    """Plot training history metrics"""
    plt.figure(figsize=(12, 6))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.savefig('plots/training_history.png')
    plt.close()

def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/JSIEC_fundus',
                       help='Path to dataset directory')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for training')
    args = parser.parse_args()

    # Constants
    IMAGE_SIZE = (380, 380)
    
    # Setup
    setup_directories()
    num_classes = verify_dataset_path(args.data_dir)

    # Data generators
    train_gen, val_gen = create_data_generators(
        args.data_dir, IMAGE_SIZE, args.batch_size
    )
    class_names = list(train_gen.class_indices.keys())

    # Callbacks
    callbacks = [
        EarlyStopping(patience=15, restore_best_weights=True, monitor='val_auc', mode='max'),
        ReduceLROnPlateau(factor=0.1, patience=5, monitor='val_loss'),
        ModelCheckpoint('models/best_model.h5', save_best_only=True, monitor='val_auc', mode='max'),
        CSVLogger('logs/training_log.csv')
    ]

    # Build and train initial model
    model = build_model((*IMAGE_SIZE, 3), num_classes)
    print("\nInitial Training Phase:")
    history = train_model(model, train_gen, val_gen, args.epochs, callbacks)
    plot_training_history(history)

    # Fine-tuning
    print("\nFine-tuning Phase:")
    model = fine_tune_model(model)
    history_fine = train_model(model, train_gen, val_gen, 20, callbacks)
    plot_training_history(history_fine)

    # Save and evaluate final model
    model.save('models/final_model.h5')
    evaluate_model(model, val_gen, class_names)

if __name__ == '__main__':
    main()