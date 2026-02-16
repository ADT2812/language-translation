import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

st.title("Neural Machine Translation (Seq2Seq LSTM)")

# -----------------------------
# Load Dataset
# -----------------------------
data = pd.read_csv("data.csv")

input_texts = data["source"].astype(str).tolist()
target_texts = ["<start> " + text + " <end>" for text in data["target"].astype(str).tolist()]

# -----------------------------
# Tokenization
# -----------------------------
input_tokenizer = Tokenizer(filters="")
target_tokenizer = Tokenizer(filters="")

input_tokenizer.fit_on_texts(input_texts)
target_tokenizer.fit_on_texts(target_texts)

input_sequences = input_tokenizer.texts_to_sequences(input_texts)
target_sequences = target_tokenizer.texts_to_sequences(target_texts)

max_input_len = max(len(seq) for seq in input_sequences)
max_target_len = max(len(seq) for seq in target_sequences)

encoder_input_data = pad_sequences(input_sequences, maxlen=max_input_len, padding="post")
decoder_input_data = pad_sequences(target_sequences, maxlen=max_target_len, padding="post")

# Proper shape for sparse categorical crossentropy
decoder_target_data = np.zeros(
    (len(target_sequences), max_target_len, 1),
    dtype="int32"
)

for i in range(len(target_sequences)):
    for t in range(1, len(target_sequences[i])):
        decoder_target_data[i, t - 1, 0] = target_sequences[i][t]

# Vocabulary sizes
num_encoder_tokens = len(input_tokenizer.word_index) + 1
num_decoder_tokens = len(target_tokenizer.word_index) + 1

latent_dim = 128


# -----------------------------
# Build Model (Cached)
# -----------------------------
@st.cache_resource
def build_models():

    # Encoder
    encoder_inputs = Input(shape=(None,), name="encoder_input")
    encoder_embedding = Embedding(num_encoder_tokens, latent_dim, name="encoder_embedding")
    enc_emb = encoder_embedding(encoder_inputs)

    encoder_lstm = LSTM(latent_dim, return_state=True, name="encoder_lstm")
    _, state_h, state_c = encoder_lstm(enc_emb)
    encoder_states = [state_h, state_c]

    # Decoder
    decoder_inputs = Input(shape=(None,), name="decoder_input")
    decoder_embedding = Embedding(num_decoder_tokens, latent_dim, name="decoder_embedding")
    dec_emb = decoder_embedding(decoder_inputs)

    decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True, name="decoder_lstm")
    decoder_outputs, _, _ = decoder_lstm(dec_emb, initial_state=encoder_states)

    decoder_dense = Dense(num_decoder_tokens, activation="softmax", name="decoder_dense")
    decoder_outputs = decoder_dense(decoder_outputs)

    # Training model
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    # Inference Encoder
    encoder_model = Model(encoder_inputs, encoder_states)

    # Inference Decoder
    decoder_state_input_h = Input(shape=(latent_dim,), name="decoder_state_h")
    decoder_state_input_c = Input(shape=(latent_dim,), name="decoder_state_c")
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

    dec_emb2 = decoder_embedding(decoder_inputs)

    decoder_outputs2, state_h2, state_c2 = decoder_lstm(
        dec_emb2, initial_state=decoder_states_inputs
    )

    decoder_states2 = [state_h2, state_c2]
    decoder_outputs2 = decoder_dense(decoder_outputs2)

    decoder_model = Model(
        [decoder_inputs] + decoder_states_inputs,
        [decoder_outputs2] + decoder_states2
    )

    return model, encoder_model, decoder_model


model, encoder_model, decoder_model = build_models()

reverse_target_word_index = {
    i: word for word, i in target_tokenizer.word_index.items()
}


# -----------------------------
# Training
# -----------------------------
if st.button("Train Model"):
    st.write("Training started...")
    model.fit(
        [encoder_input_data, decoder_input_data],
        decoder_target_data,
        batch_size=16,
        epochs=30,
        validation_split=0.2,
        verbose=1
    )
    st.success("Training Complete!")


# -----------------------------
# Translation Function
# -----------------------------
def decode_sequence(input_seq):

    states_value = encoder_model.predict(input_seq, verbose=0)

    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = target_tokenizer.word_index["<start>"]

    decoded_sentence = ""

    while True:

        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value,
            verbose=0
        )

        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_word = reverse_target_word_index.get(sampled_token_index, "")

        if sampled_word == "<end>" or len(decoded_sentence.split()) > max_target_len:
            break

        decoded_sentence += " " + sampled_word

        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index

        states_value = [h, c]

    return decoded_sentence.strip()


# -----------------------------
# User Translation UI
# -----------------------------
st.subheader("Translate Sentence")

user_input = st.text_area("Enter Source Sentence")

if st.button("Translate"):
    if user_input.strip() == "":
        st.warning("Please enter a sentence.")
    else:
        seq = input_tokenizer.texts_to_sequences([user_input])
        seq = pad_sequences(seq, maxlen=max_input_len, padding="post")
        translation = decode_sequence(seq)

        st.success("Translation:")
        st.write(translation)
