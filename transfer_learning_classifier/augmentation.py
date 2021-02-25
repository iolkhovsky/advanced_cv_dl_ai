from tensorflow.keras.preprocessing.image import ImageDataGenerator


def build_train_datagen(train_dir="/tmp/cats-v-dogs/training/"):
    train_datagen = ImageDataGenerator(rescale=1./255,
                                       rotation_range=40,
                                       width_shift_range=0.2,
                                       height_shift_range=0.2,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True,
                                       fill_mode='nearest')
    train_generator = train_datagen.flow_from_directory(train_dir,
                                                        batch_size=100,
                                                        class_mode='binary',
                                                        target_size=(150, 150))
    return train_generator


def build_val_datagen(val_dir="/tmp/cats-v-dogs/testing/"):
    validation_datagen = ImageDataGenerator(rescale=1./255)
    validation_generator = validation_datagen.flow_from_directory(val_dir,
                                                                  batch_size=100,
                                                                  class_mode='binary',
                                                                  target_size=(150, 150))
    return validation_generator
