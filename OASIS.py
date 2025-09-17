import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

def ReLU(x):
    return max(0, x)

x = []
input_names = []
num_neurons = []

def decision():
    global error2
    print("This is the OASIS decision helper! This will answer any yes or no question you have.")

    # Settings
    num_input = int(input("Input the number of input neurons you would like (number of parameters): "))
    num_neurons.append(num_input)
    for i in range(num_input):
        input_names.append(str(input(f"Input the name of input parameter {i + 1}: ")))
    num_layers = int(input("Input the number of hidden layers you would like (must be greater than 0): "))
    num_layers += 1
    for i in range(num_layers - 1):
        num_neurons.append(int(input(f"Input the number of neurons in hidden layer {i + 1}: ")))

    # Input layer
    print("All input from 1 to 0.")
    for i in input_names:
        print(f"Rate {i}: ")
        x.append(float(input()))

    target = float(input("Enter correct answer (1 = go, 0 = don't go): "))

    weights = []
    biases = []
    inputs = []
    sum = [[]]
    num = 0.0
    z = 0.0
    z_output = 0.0

    def Guess(num, z_output, z):
        for i in range(num_layers):
            if i == 0:
                for k in range(num_neurons[1]):
                    for j in range(num_input):
                        val = x[j]
                        inputs.append(val)
                        weights.append(random.uniform(-1, 1))
                        val *= weights[-1]
                        num += val
                    biases.append(random.uniform(-1, 1))
                    num += biases[-1]
                    z = ReLU(num)
                    sum[0].append(z)
                    num = 0.0
            elif i == num_layers - 1:
                for j in range(num_neurons[i]):
                    val = sum[i - 1][j]
                    inputs.append(val)
                    weights.append(random.uniform(-1, 1))
                    val *= weights[-1]
                    z_output += val
                biases.append(random.uniform(-1, 1))
                z_output += biases[-1]
            else:
                sum.append([])
                for k in range(num_neurons[i + 1]):
                    for j in range(num_neurons[i]):
                        val = sum[i - 1][j]
                        inputs.append(val)
                        weights.append(random.uniform(-1, 1))
                        val *= weights[-1]
                        num += val
                    biases.append(random.uniform(-1, 1))
                    num += biases[-1]
                    z = ReLU(num)
                    sum[i].append(z)
                    num = 0.0
        del sum[-1]
        return z_output

    z_output2 = Guess(num, z_output, z)

    def Error_calc(z_output2):
        error = target - z_output2
        print("Original z_output: ", z_output2)
        print("Original error: ", error)
        return error

    error2 = Error_calc(z_output2)

    def Check():
        sum = [[]]
        num = 0.0
        z_output2 = 0.0
        w = 0
        b = 0
        inputs.clear()

        for i in range(num_layers):
            if i == 0:
                for k in range(num_neurons[1]):
                    for j in range(num_input):
                        val = x[j]
                        inputs.append(val)
                        val *= weights[w]
                        w += 1
                        num += val
                    num += biases[b]
                    b += 1
                    z = ReLU(num)
                    sum[0].append(z)
                    num = 0.0
            elif i == num_layers - 1:
                for j in range(num_neurons[i]):
                    val = sum[i - 1][j]
                    inputs.append(val)
                    val *= weights[w]
                    w += 1
                    z_output2 += val
                z_output2 += biases[b]
                b += 1
            else:
                sum.append([])
                for k in range(num_neurons[i + 1]):
                    for j in range(num_neurons[i]):
                        val = sum[i - 1][j]
                        inputs.append(val)
                        val *= weights[w]
                        w += 1
                        num += val
                    num += biases[b]
                    b += 1
                    z = ReLU(num)
                    sum[i].append(z)
                    num = 0.0
        del sum[-1]
        print("check")
        print("z_output2: ", z_output2)
        return Error_calc(z_output2), z_output2

    def Train():
        global error2, z_output2
        print("train")
        while abs(error2) > 0.01:
            for i in range(len(weights)):
                weights[i] += learning_rate * error2 * inputs[i]
            for i in range(len(biases)):
                biases[i] += learning_rate * error2
            error2, z_output2 = Check()

    learning_rate = 0.01
    Train()

    print("Final z_output: ", z_output2)
    print("Final error: ", error2)

    # make decision again based on training
    for i in input_names:
        print(f"Rate {i}: ")
        x.append(float(input()))
    threshold = float(input("Enter threshold (0 to 1): "))
    Check()
    
    if z_output2 >= threshold:
        print("Decision: YES")
    else:
        print("Decision: NO")

    print("Thank you for using the OASIS decision helper!")


def chessboard_detection():


    # First layer of model

    # def train():
    #     from google.colab import drive

    #     drive.mount('/content/drive')

    #     # paths
    #     data_dir = "/content/drive/MyDrive/OASIS"
    #     latest_ckpt = f"{data_dir}/latest_checkpoint.h5"
    #     log_path = f"{data_dir}/training_log.csv"

    #     # get the last completed epoch from the training log
    #     def get_last_epoch(log_path):
    #         if not os.path.exists(log_path):
    #             return 0
    #         with open(log_path, "r") as f:
    #             lines = f.readlines()
    #             return len(lines) - 1  # subtract header

    #     initial_epoch = get_last_epoch(log_path)
    #     print(f"Resuming from epoch {initial_epoch}")

    #     # load datasets
    #     train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    #         f"{data_dir}/train",
    #         image_size=(224, 224),
    #         batch_size=32,
    #         label_mode="int",
    #         shuffle=True
    #     )

    #     val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    #         f"{data_dir}/valid",
    #         image_size=(224, 224),
    #         batch_size=32,
    #         label_mode="int"
    #     )

    #     class_names = train_ds.class_names
    #     print("Class names:", class_names)

    #     # preprocessing
    #     data_augmentation = tf.keras.Sequential([
    #         tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    #         tf.keras.layers.RandomRotation(0.2),
    #         tf.keras.layers.RandomZoom(0.2),
    #         tf.keras.layers.RandomTranslation(0.2, 0.2),
    #         tf.keras.layers.RandomContrast(0.2),
    #     ])

    #     normalization_layer = tf.keras.layers.Rescaling(1 / 255.0)

    #     train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    #     val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))
    #     train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y))

    #     AUTOTUNE = tf.data.AUTOTUNE
    #     train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    #     val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

    #     # load or build model
    #     if os.path.exists(latest_ckpt):
    #         print("Loading model from latest checkpoint...")
    #         model = tf.keras.models.load_model(latest_ckpt)
    #         model.compile(optimizer='adam',
    #                     loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    #                     metrics=['accuracy'])
    #     else:
    #         print("Starting training from scratch...")
    #         model = tf.keras.Sequential([
    #             tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(224, 224, 3)),
    #             tf.keras.layers.MaxPooling2D(),
    #             tf.keras.layers.Conv2D(64, 3, activation='relu'),
    #             tf.keras.layers.MaxPooling2D(),
    #             tf.keras.layers.Conv2D(128, 3, activation='relu'),
    #             tf.keras.layers.MaxPooling2D(),
    #             tf.keras.layers.Dropout(0.35),
    #             tf.keras.layers.Conv2D(256, 3, activation='relu'),
    #             tf.keras.layers.MaxPooling2D(),
    #             tf.keras.layers.Conv2D(612, 3, activation='relu'),
    #             tf.keras.layers.MaxPooling2D(),
    #             tf.keras.layers.GlobalAveragePooling2D(),
    #             tf.keras.layers.Dense(128, activation='relu'),
    #             tf.keras.layers.Dropout(0.5),
    #             tf.keras.layers.Dense(2, activation='softmax')
    #         ])

    #         model.compile(optimizer='adam',
    #                     loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    #                     metrics=['accuracy'])

        # callbacks
        # early_stopping = tf.keras.callbacks.EarlyStopping(
        #     monitor='val_loss',
        #     patience=3,
        #     restore_best_weights=True
        # )

        # best_model_cb = tf.keras.callbacks.ModelCheckpoint(
        #     filepath=f"{data_dir}/best_model.h5",
        #     save_best_only=True,
        #     monitor='val_loss'
        # )

        # latest_checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        #     filepath=latest_ckpt,
        #     save_freq='epoch',
        #     save_weights_only=False
        # )

        # csv_logger = tf.keras.callbacks.CSVLogger(log_path, append=True)

        # # train the model
        # model.fit(train_ds,
        #         validation_data=val_ds,
        #         epochs=30,
        #         initial_epoch=initial_epoch,
        #         callbacks=[early_stopping, best_model_cb, latest_checkpoint_cb, csv_logger])

        # # save final model
        # final_model_path = f"{data_dir}/model.h5"
        # model.save(final_model_path)
        # print(f"Final model saved to: {final_model_path}")
    

    def input_image():
        # Get image
        global img_file_path
        img_file_path = input("Please input the path to the image of the chessboard here: ")

        # First layer of image recognition
        model = load_model("/Users/aryeh/Downloads/OASIS/chessboard_detection.v1i.yolov5-obb/model.h5")

        img = load_img(img_file_path, target_size=(224, 224))
        img_array = img_to_array(img)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)


        prediction = model.predict(img_array)
        global predicted_class
        predicted_class = tf.argmax(prediction[0])

        for value in prediction[0]:
            print(f"{value:.10f}")
    input_image()

    if predicted_class == 0:
        pass
    else:
        input_image()

program = input('Type "1" if you would like the decision helper, and "2" if you would like the chessboard detector: ')

if program == "1":
    decision()
elif program == "2":
    chessboard_detection()