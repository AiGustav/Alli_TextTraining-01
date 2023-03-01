# Alli_TextTraining-01
This is the code for the training of Alli for Text inputs(even large amount of text) I'll upload another example soon


                                        Code Starts here:
code: import numpy as np
import tensorflow as tf

# Define the training data
training_data = [
    "hello Alli",
    "this is a test",
    "another example",
    "Alli is great",
    "some more text",
    "final training data",
    "another training sentence",
    "yet another example",
    "the last sentence",
    "Hey Alli",
    "do you know that around 53 percent of website traffic comes from organic search which makes search engine optimization a top priority for businesses online the more you rank high in search engines the more you have the possibility to gain traffic to your website which",
    "indirectly results in conversions thereby boosting your business moreover the market is flooded with SEO jobs a few of them includes SEO Associates SEO analysts SEO Executives SEO managers SEO experts with salaries going as high as hundred thousand  ",
    "dollars which makes it a lucrative career so here we present you our SEO full course that will cover every aspects needed to get a job in this field before we move ahead consider subscribing to our Channel and hit the Bell icon to never miss any updates from Simply learn in this SEO full course we will be recovering all the major and minor concepts related to SEO we have smartly divided this course from beginners to advanced level so that anyone can become an SEO expert with this all in one course we will start off",
]

# Define the vocabulary
vocab = sorted(set("".join(training_data)))
char_to_idx = {char: idx for idx, char in enumerate(vocab)}
idx_to_char = np.array(vocab)

# Convert the training data to sequences of indices
sequences = []
for seq in training_data:
    sequences.append([char_to_idx[char] for char in seq])

# Define the inputs and targets
inputs = []
targets = []
for seq in sequences:
    inputs.append(seq[:-1])
    targets.append(seq[1:])

# Pad the sequences to a maximum length
max_len = max(len(seq) for seq in sequences)
inputs = tf.keras.preprocessing.sequence.pad_sequences(inputs, maxlen=max_len, padding="pre")
targets = tf.keras.preprocessing.sequence.pad_sequences(targets, maxlen=max_len, padding="pre")

# Define the model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=len(vocab), output_dim=64),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
    tf.keras.layers.Dense(len(vocab), activation="softmax")
])

# Compile the model
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Train the model
batch_size = 32
num_epochs = 1000
model.fit(inputs, tf.keras.utils.to_categorical(targets, num_classes=len(vocab)), batch_size=batch_size, epochs=num_epochs)
