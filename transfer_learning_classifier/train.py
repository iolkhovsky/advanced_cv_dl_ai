import pickle
from tensorflow.keras.optimizers import RMSprop

from augmentation import build_train_datagen, build_val_datagen
from model import build_classifier, download_weights


model_config = "inception_v3.h5"
download_weights(target=model_config)
model = build_classifier(model_config)

model.compile(optimizer=RMSprop(lr=0.0001),
              loss='binary_crossentropy',
              metrics=['acc'])

train_generator = build_train_datagen()
validation_generator = build_val_datagen()

epochs_cnt = 2
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=epochs_cnt,
    verbose=1)

models_name = f"trained_model"
model.save(models_name)
print(f"{models_name} has been saved")

history_name = f"history.pk"
with open(history_name, "wb") as f:
    pickle.dump(history, f)
print(f"{history_name} has been saved")
