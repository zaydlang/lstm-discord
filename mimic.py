import discord
import keras.utils as ku
import numpy as np
import requests
from discord.ext import commands
from keras.callbacks import EarlyStopping
from keras.layers import LSTM, Dense, Embedding
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
import json

# built off of https://medium.com/@shivambansal36/language-modelling-text-generation-using-lstms-deep-learning-for-nlp-ed36b224b275

bot = commands.Bot(command_prefix='$')
bot.remove_command("help") # discord has its own stupid default help command

############### CLASSES ###############
# A Mimic state with it's model and everything
class Mimic:
    def __init__(self, data):
        self.model            = None
        self.epochs_completed = 0
        self.predictors       = []
        self.labels           = []
        self.max_sequence_len = 0
        self.total_words      = 0
        self.tokenizer        = Tokenizer()

        self.dataset_preparation(data)
        self.create_model()

    def dataset_preparation(self, data):
        lines = data # to clarify, you pass in data, which is a list of lines.

        # converts data to tokens using built-in keras tokenizer
        self.tokenizer.fit_on_texts(lines)
        self.total_words = len(self.tokenizer.word_index) + 1

        input_sequences = []
        for line in lines:
            token_list = self.tokenizer.texts_to_sequences([line])[0]
            for i in range(1, len(token_list)):
                n_gram_sequence = token_list[:i+1]
                input_sequences.append(n_gram_sequence)
        
        self.max_sequence_len = max([len(x) for x in input_sequences])
        input_sequences = np.array(pad_sequences(input_sequences,   
                            maxlen=self.max_sequence_len, padding='pre'))

        # each sentence is a predictor of the next sentence
        self.predictors, self.labels = input_sequences[:,:-1],input_sequences[:,-1]
        self.labels = ku.to_categorical(self.labels, num_classes=self.total_words)

    def create_model(self):
        input_len = self.max_sequence_len - 1
        self.model = Sequential()

        self.model.add(Embedding(self.total_words, 10, input_length=input_len))
        self.model.add(LSTM(150))
        # model.add(Dropout(0.1)) todo: why??
        self.model.add(Dense(self.total_words, activation='softmax'))

        self.model.compile(loss='categorical_crossentropy', optimizer='adam')

    def train(self, num_epochs):
        self.model.fit(self.predictors, self.labels, epochs=num_epochs, verbose=1)
        self.epochs_completed += num_epochs
 
    def predict(self, seed_text):
        next_words = 10

        for j in range(next_words):
            token_list = self.tokenizer.texts_to_sequences([seed_text])[0]
            token_list = pad_sequences([token_list], maxlen= 
                                self.max_sequence_len-1, padding='pre')
            predicted = self.model.predict_classes(token_list, verbose=0)
    
            output_word = ""
            for word, index in self.tokenizer.word_index.items():
                if index == predicted:
                    output_word = word
                    break
            seed_text += " " + output_word
        
        return seed_text

############### DATASET PREPARATION ###############
mimics        = []
current_mimic = 0

# data = open('data2.txt', 'r').read().split("\n")  
# mimics.append(Mimic(data))

############### DISCORD BOT STUFF ###############
@bot.event
async def on_ready():
    print("Online.")

@bot.command()
async def train(ctx, *args):
    await ctx.send("starting yay")
    num_epochs = int(args[0])
    mimics[current_mimic].train(num_epochs)
    await ctx.send("k done")

@bot.command()
async def predict(ctx, *args):
    await ctx.send(mimics[current_mimic].predict(' '.join(args)))

@bot.command()
# harvest a conversation
async def harvest(ctx, *args):
    global mimics

    num_messages = int(args[0])
    guild        = ctx.guild
    data         = []

    # gather messages for each person
    for channel in guild.channels:
        # we only care about text channels
        if str(channel.type) == 'text':
            conversation = []   # list of messages
            current_user         = None # current person in conversation
            current_conversation = ""
            async for message in channel.history(limit=num_messages):
                # if we're still at the same user...
                if message.author == current_user:
                    # we're reading backwards, so insert at the beginning.
                    current_conversation = message.content + " " + current_conversation
                # no? then set up for next user
                else:
                    # we're reading backwards, so insert at the beginning.
                    conversation.insert(0, current_conversation) # this inserts a false ending. Removed later.
                    current_user         = message.author
                    current_conversation = message.content # start with this message

            conversation = conversation[:-1] # aforementioned false ending removed
            
            formatted_conversation = []
            for message1, message2 in zip(conversation[1:], conversation[:-1]):
                formatted_conversation.append(message1 + " " + message2)
            
            data += formatted_conversation
            print("done with " + channel.name)
    
    await ctx.send("k got all the data yee")
    mimics = [Mimic(data)]

@bot.command()
async def save(ctx, *args):
    mimics[current_mimic].model.save_weights('marx.h5')

@bot.command()
async def load(ctx, *args):
    file_name = args[0]
    mimics[current_mimic].model.save_weights(file_name)
    await ctx.send(file_name + " successfully loaded.")

bot_file = open('bot.json')
bot_json = json.load(bot_file)
bot.run(bot_json['token'])
