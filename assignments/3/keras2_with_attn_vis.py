import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
import argparse
import seaborn as sns
from tensorflow.keras.layers import LSTM, Input, Dense, Embedding, GRU, SimpleRNN, Dropout, Concatenate
from tensorflow.keras import Model
from tensorflow.python.keras.layers import TimeDistributed
import matplotlib.pyplot as plt

import wandb
from keras.optimizers import Adam
from attention import AttentionLayer


# In[7]:
def get_mid_k_items(k, list):
    # computing strt, and end index
    strt_idx = (len(list) // 2) - (k // 2)
    end_idx = (len(list) // 2) + (k // 2)

    return list[strt_idx: end_idx + 1]


def get_model(encoder_layers, decoder_layers, encoder_inputs, decoder_inputs, cell_type, dropout):
    encoder_latent_dims = np.full(encoder_layers, hidden_size)
    decoder_latent_dims = np.full(decoder_layers, hidden_size)

    outputs = Embedding(num_encoder_tokens, embedding_size)(encoder_inputs)
    encoder_states = []
    cell = SimpleRNN
    if cell_type == 'lstm':
        cell = LSTM
    elif cell_type == 'gru':
        cell = GRU
    for j in range(len(encoder_latent_dims))[::-1]:
        if cell_type == 'lstm':
            outputs, h, c = cell(hidden_size, return_state=True, return_sequences=bool(j), recurrent_dropout=dropout)(
                outputs)
            encoder_states += [(h, c)]
        else:
            outputs, hidden = cell(hidden_size, return_state=True, return_sequences=bool(j), recurrent_dropout=dropout)(
                outputs)
            encoder_states += [hidden]
            # Set up the decoder, setting the initial state of each layer to the state of the layer in the encoder
    # which is it's mirror (so for encoder: a->b->c, you'd have decoder initial states: c->b->a).
    manipulated_encoder_states = manipulate_layered(decoder_layers, encoder_layers, encoder_states)
    outputs = decoder_inputs
    output_layers = []
    for j in range(len(decoder_latent_dims)):
        output_layers.append(
            cell(hidden_size, return_sequences=True, return_state=True, recurrent_dropout=dropout)
        )
        if cell_type == 'lstm':
            outputs, dh, dc = output_layers[-1](outputs, initial_state=manipulated_encoder_states[j])
        else:
            outputs, hidden = output_layers[-1](outputs, initial_state=manipulated_encoder_states[j])
        # outputs, dh, dc = output_layers[-1](outputs, initial_state=encoder_states[j])

    dropout_layer = Dropout(dropout)
    decoder_dense = Dense(num_decoder_tokens, activation='softmax')
    decoder_outputs = decoder_dense(dropout_layer(outputs))
    # Define the model that will turn
    # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    return decoder_dense, decoder_outputs, output_layers, encoder_states, model


def get_attention_model(encoder_layers, decoder_layers, encoder_inputs, decoder_inputs, cell_type, dropout):
    encoder_latent_dims = np.full(encoder_layers, hidden_size)
    decoder_latent_dims = np.full(decoder_layers, hidden_size)

    embedding_layer = Embedding(num_encoder_tokens, embedding_size)
    outputs = embedding_layer(encoder_inputs)
    encoder_states = []
    cell = SimpleRNN
    if cell_type == 'lstm':
        cell = LSTM
    elif cell_type == 'gru':
        cell = GRU
    input_layers = []
    for j in range(len(encoder_latent_dims))[::-1]:
        if cell_type == 'lstm':
            cell_ = cell(hidden_size, return_state=True, return_sequences=True, recurrent_dropout=dropout)
            input_layers.append(cell_)
            outputs, h, c = cell_(outputs)
            encoder_states += [(h, c)]
        else:
            cell_ = cell(hidden_size, return_state=True, return_sequences=True, recurrent_dropout=dropout)
            input_layers.append(cell_)
            outputs, hidden = cell_(outputs)
            encoder_states += [hidden]
            # Set up the decoder, setting the initial state of each layer to the state of the layer in the encoder
    # which is it's mirror (so for encoder: a->b->c, you'd have decoder initial states: c->b->a).
    encoder_out = outputs
    manipulated_encoder_states = manipulate_layered(decoder_layers, encoder_layers, encoder_states)
    outputs = decoder_inputs
    output_layers = []
    for j in range(len(decoder_latent_dims)):
        output_layers.append(
            cell(hidden_size, return_sequences=True, return_state=True, recurrent_dropout=dropout)
        )
        if cell_type == 'lstm':
            outputs, dh, dc = output_layers[-1](outputs, initial_state=manipulated_encoder_states[j])
        else:
            outputs, hidden = output_layers[-1](outputs, initial_state=manipulated_encoder_states[j])
        # outputs, dh, dc = output_layers[-1](outputs, initial_state=encoder_states[j])

    attn_layer = AttentionLayer(name='attention_layer')
    print(encoder_out.shape)
    print(outputs.shape)
    attn_out, attn_states = attn_layer([encoder_out, outputs])

    # Concat attention input and decoder GRU output
    decoder_concat_input = Concatenate(axis=-1, name='concat_layer')([outputs, attn_out])

    dropout_layer = Dropout(dropout)
    decoder_dense = Dense(num_decoder_tokens, activation='softmax')
    decoder_outputs = decoder_dense(dropout_layer(decoder_concat_input))
    # Define the model that will turn
    # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`

    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model.summary()
    """ Encoder (Inference) model """
    encoder_inf_inputs = Input(batch_shape=(None, None), name='encoder_inf_inputs')
    output = embedding_layer(encoder_inf_inputs)
    encoder_states_inf = []

    for j in range(len(encoder_latent_dims))[::-1]:
        temp = input_layers[len(encoder_latent_dims) - j - 1](output)

        output, cur_states = temp[0], temp[1:]

        encoder_states_inf += cur_states

    encoder_model = Model(inputs=encoder_inf_inputs, outputs=[output, encoder_states_inf])

    """ Decoder (Inference) model """
    decoder_inf_inputs = Input(batch_shape=(None, 1, num_decoder_tokens), name='decoder_word_inputs')
    encoder_inf_out = Input(batch_shape=(None, None, hidden_size), name='encoder_inf_states')
    # decoder_init_state = Input(batch_shape=(batch_size, hidden_size), name='decoder_init')

    decoder_states_inputs = []
    decoder_states = []
    d_outputs = decoder_inf_inputs
    state_count = 2 if cell_type == 'lstm' else 1
    encoder_inf_states = []
    for j in range(len(decoder_latent_dims))[::-1]:
        initial_state = [Input(shape=(decoder_latent_dims[j],)) for _ in range(state_count)]
        temp = output_layers[len(decoder_latent_dims) - j - 1](d_outputs, initial_state=initial_state)

        d_outputs, cur_states = temp[0], temp[1:]
        encoder_inf_states += initial_state
        decoder_states += cur_states
        # decoder_states_inputs += current_state_inputs

    attn_inf_out, attn_inf_states = attn_layer([encoder_inf_out, d_outputs])
    decoder_inf_concat = Concatenate(axis=-1, name='concat')([d_outputs, attn_inf_out])
    decoder_inf_pred = TimeDistributed(decoder_dense)(decoder_inf_concat)
    decoder_model = Model(inputs=[encoder_inf_out, encoder_inf_states, decoder_inf_inputs],
                          outputs=[decoder_inf_pred, attn_inf_states, decoder_states])

    return decoder_dense, decoder_outputs, output_layers, encoder_states, model, attn_layer, encoder_out, encoder_model, decoder_model


def manipulate_layered(decoder_layers, encoder_layers, encoder_states):
    manipulated_encoder_states = encoder_states
    if decoder_layers < encoder_layers:
        manipulated_encoder_states = encoder_states[-1 * decoder_layers:]
    elif encoder_layers < decoder_layers:
        manipulated_encoder_states = []
        manipulated_encoder_states += encoder_states[:decoder_layers // 2]
        remaining_layers = decoder_layers - 2 * (decoder_layers // 2)
        manipulated_encoder_states += get_mid_k_items(remaining_layers, encoder_states)
        manipulated_encoder_states += encoder_states[-1 * (decoder_layers // 2):]
    return manipulated_encoder_states


def prepare_data(input_texts, target_texts, num_encoder_tokens, num_decoder_tokens):
    max_encoder_seq_length = max([len(txt) for txt in input_texts])
    max_decoder_seq_length = max([len(txt) for txt in target_texts])
    print("Number of samples:", len(input_texts))
    print("Number of unique input tokens:", num_encoder_tokens)
    print("Number of unique output tokens:", num_decoder_tokens)
    print("Max sequence length for inputs:", max_encoder_seq_length)
    print("Max sequence length for outputs:", max_decoder_seq_length)
    # encoder_input_data = np.zeros(
    #     (len(input_texts), max_encoder_seq_length, num_encoder_tokens), dtype="float32"
    # )
    encoder_input_data = []
    decoder_input_data = np.zeros(
        (len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype="float32"
    )
    decoder_target_data = np.zeros(
        (len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype="float32"
    )
    for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
        input_arr = []
        for t, char in enumerate(input_text):
            input_arr.append(input_token_index[char])
        encoder_input_data.append(input_arr)
        # encoder_input_data[i, t + 1 :, input_token_index[" "]] = 1.0
        for t, char in enumerate(target_text):
            # decoder_target_data is ahead of decoder_input_data by one timestep
            decoder_input_data[i, t, target_token_index[char]] = 1.0
            if t > 0:
                # decoder_target_data will be ahead by one timestep
                # and will not include the start character.
                decoder_target_data[i, t - 1, target_token_index[char]] = 1.0
        # decoder_input_data[i, t + 1 :, target_token_index[" "]] = 1.0
        # decoder_target_data[i, t:, target_token_index[" "]] = 1.0
    encoder_input_data = pad_sequences(encoder_input_data, maxlen=max_encoder_seq_length, padding='post')

    return encoder_input_data, decoder_input_data, decoder_target_data


def get_data(data_path, input_characters, target_characters):
    input_texts = []
    target_texts = []
    with open(data_path, "r", encoding="utf-8") as f:
        lines = f.read().split("\n")
    for line in lines:
        try:
            input_text, target_text, _ = line.split("\t")
        except:
            print(line)
        # We use "tab" as the "start sequence" character
        # for the targets, and "\n" as "end sequence" character.
        target_text = "\t" + target_text + "\n"
        input_texts.append(input_text)
        target_texts.append(target_text)
        for char in input_text:
            if char not in input_characters:
                input_characters.add(char)
        for char in target_text:
            if char not in target_characters:
                target_characters.add(char)

    return input_texts, target_texts


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        usage="%(prog)s [OPTION] [FILE]...",
        description="Print or check SHA1 (160-bit) checksums."
    )
    parser.add_argument("--embedding_size")
    parser.add_argument('--encoder_layers', )
    parser.add_argument('--decoder_layers', )
    parser.add_argument('--hidden_size', )
    parser.add_argument('--cell_type', )
    parser.add_argument('--dropout')
    parser.add_argument('--beam_search')
    # parser.add_argument('--learning_rate')

    args = parser.parse_args()
    hidden_size = int(args.hidden_size)
    dropout = int(args.dropout) / 100
    embedding_size = int(args.embedding_size)
    cell_type = args.cell_type
    encoder_layers = int(args.encoder_layers)
    decoder_layers = int(args.decoder_layers)
    # learning_rate = float(args.learning_rate)
    # beam_search = args.beam_search
    # hidden_size = embedding_size
    # beam_search = True
    # learning_rate = 0.001
    name = "hs:" + str(hidden_size) + "_dropout:" + str(dropout) + "_embedding_size:" + str(
        embedding_size) + "_cell_type:" + str(cell_type) + "_encoder_layers:" + str(
        encoder_layers) + "_decoder_layers:" + str(decoder_layers)
    wandb.init(project="assignment-3", name=name)

    wandb.config.hidden_size = hidden_size
    wandb.config.dropout = dropout
    wandb.config.embedding_size = embedding_size
    wandb.config.cell_type = cell_type
    wandb.config.encoder_layers = encoder_layers
    wandb.config.decoder_layers = decoder_layers
    # wandb.config.beam_search = beam_search
    # wandb.config.learning_rate = learning_rate

    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
    from wandb.keras import WandbCallback

    batch_size = 1024  # Batch size for training.
    epochs = 1  # Number of epochs to train for.

    data_path = "/data/dakshina/dakshina_dataset_v1.0/ta/lexicons/ta.translit.sampled.train.tsv"
    val_data_path = "/data/dakshina/dakshina_dataset_v1.0/ta/lexicons/ta.translit.sampled.dev.tsv"
    input_characters = set()
    target_characters = set()
    input_texts, target_texts = get_data(data_path, input_characters, target_characters)
    val_input_texts, val_target_texts = get_data(val_data_path, input_characters, target_characters)

    input_characters = sorted(list(input_characters))
    target_characters = sorted(list(target_characters))

    input_token_index = dict([(char, i) for i, char in enumerate(input_characters)])
    target_token_index = dict([(char, i) for i, char in enumerate(target_characters)])
    num_encoder_tokens = len(input_characters)
    num_decoder_tokens = len(target_characters)

    encoder_input_data, decoder_input_data, decoder_target_data = prepare_data(input_texts, target_texts,
                                                                               num_encoder_tokens, num_decoder_tokens)

    val_encoder_input_data, val_decoder_input_data, val_decoder_target_data = prepare_data(val_input_texts,
                                                                                           val_target_texts,
                                                                                           num_encoder_tokens,
                                                                                           num_decoder_tokens)

    encoder_inputs = Input(shape=(None,))
    decoder_inputs = Input(shape=(None, num_decoder_tokens))

    decoder_dense, decoder_outputs, output_layers, encoder_states, model, attn_layer, encoder_out, encoder_model, decoder_model = get_attention_model(
        encoder_layers,
        decoder_layers,
        encoder_inputs,
        decoder_inputs,
        cell_type, dropout)

    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    history = model.fit(
        [encoder_input_data, decoder_input_data],
        decoder_target_data,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=([val_encoder_input_data, val_decoder_input_data], val_decoder_target_data),
        callbacks=[callback, WandbCallback()]
    )
    # Save model
    model.save("s2s")

    # Define sampling models (modified for n-layer deep network)
    # encoder_model = Model(encoder_inputs, [encoder_states, encoder_out])
    #
    # d_outputs = decoder_inputs
    # decoder_states_inputs = []
    # decoder_states = []
    # decoder_latent_dims = np.full(decoder_layers, hidden_size)
    # for j in range(len(decoder_latent_dims))[::-1]:
    #     state_count = 2 if cell_type == 'lstm' else 1
    #     current_state_inputs = [Input(shape=(decoder_latent_dims[j],)) for _ in range(state_count)]
    #
    #     temp = output_layers[len(decoder_latent_dims) - j - 1](d_outputs, initial_state=current_state_inputs)
    #
    #     d_outputs, cur_states = temp[0], temp[1:]
    #
    #     decoder_states += cur_states
    #     decoder_states_inputs += current_state_inputs
    #
    # print(encoder_out.shape)
    # print(d_outputs.shape)
    # attn_out, attn_states = attn_layer([encoder_out, d_outputs])
    #
    # # Concat attention input and decoder GRU output
    # decoder_concat_input = Concatenate(axis=-1, name='concat_layer')([d_outputs, attn_out])
    #
    # decoder_outputs = decoder_dense(decoder_concat_input)
    # decoder_model = Model(
    #     [decoder_inputs, decoder_states_inputs],
    #     [decoder_outputs] + decoder_states)

    reverse_input_char_index = dict((i, char) for char, i in input_token_index.items())
    reverse_target_char_index = dict((i, char) for char, i in target_token_index.items())
    max_decoder_seq_length = max([len(txt) for txt in target_texts])


    def plot_attention_weights(encoder_inputs, attention_weights, en_id2word, fr_id2word):
        mats = []
        dec_inputs = []
        for dec_ind, attn in attention_weights:
            mats.append(attn.reshape(-1))
            dec_inputs.append(dec_ind)
        attention_mat = np.transpose(np.array(mats))

        fig, ax = plt.subplots(figsize=(32, 32))
        # ax.imshow(attention_mat)
        #
        # ax.set_xticks(np.arange(attention_mat.shape[1]))
        # ax.set_yticks(np.arange(attention_mat.shape[0]))
        #
        output_chars = [fr_id2word[inp] if inp != 0 else "<Res>" for inp in dec_inputs]
        # ax.set_xticklabels(output_chars)
        input_chars = [en_id2word[inp] if inp != 0 else "<Res>" for inp in encoder_inputs]
        # ax.set_yticklabels(input_chars)
        #
        # ax.tick_params(labelsize=32)
        # ax.tick_params(axis='x', labelrotation=90)
        #
        # plt.savefig('attention.png')
        
        heatmap = sns.heatmap(attention_mat,xticklabels=output_chars,yticklabels=input_chars)

        # plt.show()
        heatmap.get_figure().savefig('attention.png')
    # In[12]:

    def decode_sequence(input_seq, encoder_model, decoder_model):
        # Encode the input as state vectors.
        states_value_and_out = encoder_model.predict(input_seq)

        encoder_outs = states_value_and_out[0]
        states_value = states_value_and_out[1]
        # Generate empty target sequence of length 1.
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        # Populate the first character of target sequence with the start character.
        target_seq[0, 0, target_token_index['\t']] = 1.
        attn_weights = []
        # Sampling loop for a batch of sequences
        # (to simplify, here we assume a batch of size 1).
        stop_condition = False
        decoded_sentence = []  # Creating a list then using "".join() is usually much faster for string creation
        states_value = manipulate_layered(decoder_layers, encoder_layers, states_value)
        while not stop_condition:
            # print(target_seq.shape)
            # print(states_value.shape)
            # print(encoder_outs.shape)
            to_split = decoder_model.predict([encoder_outs, states_value, target_seq])
            output_tokens, attn_out, states_value = to_split
            target_seq = output_tokens
            # Sample a token
            sampled_token_index = np.argmax(output_tokens[0, 0])
            attn_weights.append((sampled_token_index, attn_out))
            sampled_char = reverse_target_char_index[sampled_token_index]
            decoded_sentence.append(sampled_char)

            # Exit condition: either hit max length
            # or find stop character.
            if sampled_char == '\n' or len(decoded_sentence) > max_decoder_seq_length:
                stop_condition = True

            # Update the target sequence (of length 1).
            target_seq = np.zeros((1, 1, num_decoder_tokens))
            target_seq[0, 0, sampled_token_index] = 1.

        return "".join(decoded_sentence), attn_weights


    import random

    attn_tuple = []
    for seq_index in range(2):
        seq_index = random.randint(0, len(val_input_texts))
        # Take one sequence (part of the training set)
        # for trying out decoding.
        input_seq = encoder_input_data[seq_index: seq_index + 1]
        decoded_sentence, attn_weights = decode_sequence(input_seq, encoder_model, decoder_model)
        encoder_input = encoder_input_data[seq_index: seq_index + 1]
        encoder_input_cleaned = []
        for input in encoder_input[0]:
            if input != 0:
                encoder_input_cleaned.append(input)
        plot_attention_weights(encoder_input_cleaned, attn_weights, reverse_input_char_index,
                               reverse_target_char_index)
        print("-")
        print("Input sentence:", input_texts[seq_index])
        print("Decoded sentence:", decoded_sentence)
        print("target sentence:", target_texts[seq_index])

import matplotlib.pyplot as plt


