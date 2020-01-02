
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

# built off of https://medium.com/@shivambansal36/language-modelling-text-generation-using-lstms-deep-learning-for-nlp-ed36b224b275

############## INIT ###############
tokenizer = Tokenizer()
data      = open('data2.txt', 'r').read()

bot = commands.Bot(command_prefix='$')
bot.remove_command("help") # discord has its own stupid default help command

headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}

epochs_completed = 0
predictors       = []
label            = []
max_sequence_len = 0
total_words      = 0

############ FUNCTIONS ############
def dataset_preparation(data):
    # converts data to tokens using built-in keras tokenizer
    corpus = data.lower().split("\n")    
    tokenizer.fit_on_texts(corpus)
    total_words = len(tokenizer.word_index) + 1

    input_sequences = []
    for line in corpus:
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i+1]
            input_sequences.append(n_gram_sequence)
    
    max_sequence_len = max([len(x) for x in input_sequences])
    input_sequences = np.array(pad_sequences(input_sequences,   
                          maxlen=max_sequence_len, padding='pre'))

    # each sentence is a predictor of the next sentence
    predictors = []
    label      = []
    
    predictors, label = input_sequences[:,:-1],input_sequences[:,-1]
    label = ku.to_categorical(label, num_classes=total_words)

    return predictors, label, max_sequence_len, total_words

def create_model(max_len):
    global model

    input_len = max_len - 1
    model = Sequential()

    model.add(Embedding(total_words, 10, input_length=input_len))
    model.add(LSTM(150))
    # model.add(Dropout(0.1)) todo: why??
    model.add(Dense(total_words, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model

@bot.command()
async def train(ctx, *args):
    global epochs_completed
    global total_words
    global model
    num_epochs       = int(args[0])

    await ctx.send("training uwu")
    model.fit(predictors, label, epochs=num_epochs, verbose=1)

    epochs_completed += num_epochs
    await ctx.send("completed, total epochs completed: " + str(num_epochs))

@bot.command()
async def predict(ctx):
    global max_sequence_len
    seed_text        = ctx.message.content[11:]
    next_words       = 10

    for j in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen= 
                             max_sequence_len-1, padding='pre')
        predicted = model.predict_classes(token_list, verbose=0)
  
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text += " " + output_word
    
    await ctx.send(seed_text)


def tokenize(messages):
    sentences           = messages
    tokens              = {}  # token dictionary
    highest_token       = 0   # highest token made so far
    tokenized_sentences = []  # tokenized sentences

    for sentence in sentences:
        words              = sentence.split(" ")
        tokenized_sentence = []

        for word in words:
            # if it hasn't been tokenized, then tokenize it
            if word not in tokens.keys():
                highest_token += 1
                tokens[word] = highest_token
            
            tokenized_sentence.append(tokens[word])
        tokenized_sentences.append(tokenized_sentence)
    
    print(tokenized_sentences)
    return tokens, tokenized_sentences
            
@bot.command()
# harvest a conversation
async def harvest(ctx, *args):
    global predictors
    global label
    global max_sequence_len
    global total_words

    channel_name = args[0]
    num_messages = int(args[1])
    guild        = ctx.guild
    conversation = []   # list of messages

    # gather messages for each person
    for channel in guild.channels:
        # we only care about text channels
        if channel.name == channel_name and str(channel.type) == 'text':
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

    # tokenize and pad
    tokens, tokenized_conversation = tokenize(conversation)
    total_words = len(tokens) + 1
    max_sequence_len = max([len(x) for x in tokenized_conversation])
    input_sequences = np.array(pad_sequences(tokenized_conversation,   
                          maxlen=max_sequence_len, padding='pre'))

    predictors, label = input_sequences[:,:-1],input_sequences[:,-1]
    label = ku.to_categorical(label, num_classes=total_words)

    # create model
    create_model(max_sequence_len)
    await ctx.send("done.")

@bot.command()
# berry's fun command
async def berry(ctx, *args):
    global predictors
    global label
    global max_sequence_len
    global total_words

    num_examples = int(args[0])
    examples     = []
    tokenizer    = Tokenizer()
    corpus       = []

    await ctx.send("collecting examples")
    # collect showerthoughts
    for i in range(num_examples):
        url = 'https://www.reddit.com/r/showerthoughts/random/.json'
        data = requests.get(headers={'user-agent':'scraper by /u/ciwi'}, url=url).json()

        children = data[0]['data']['children'][0]['data']
        title = children['title']

        corpus.append(title.lower())
       
    tokenizer.fit_on_texts(corpus) 

    for message in corpus:
        words = tokenizer.texts_to_sequences([message])[0]
        for i in range(1, len(words)):
            examples.append(words[:i+1])
            print(words[:i+1])
    
    await ctx.send("tokens yee")
    # tokenize
    
    max_sequence_len = max([len(x) for x in examples])
    input_sequences = np.array(pad_sequences(examples,   
                          maxlen=max_sequence_len, padding='pre'))

    # each sentence is a predictor of the next sentence
    predictors = []
    label      = []
    
    predictors, label = input_sequences[:,:-1],input_sequences[:,-1]
    label = ku.to_categorical(label, num_classes=total_words)

    await ctx.send("making model")
    create_model(max_sequence_len)
    await ctx.send("done.")

@bot.command()
# harvest a conversation
async def harvest(ctx, *args):
    global predictors
    global label
    global max_sequence_len
    global total_words

    channel_name = args[0]
    num_messages = int(args[1])
    guild        = ctx.guild
    conversation = []   # list of messages

    # gather messages for each person
    for channel in guild.channels:
        # we only care about text channels
        if channel.name == channel_name and str(channel.type) == 'text':
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

    await ctx.send(conversation)

@bot.command()
async def save(ctx, *args):
    model.save_weights('marx.h5')

@bot.command()
async def load(ctx, *args):
    model.save_weights('marx.h5')

X, Y, max_len, total_words = dataset_preparation(data)
model = create_model(max_len)
######## DISCORD BOT STUFF ########

@bot.event
async def on_ready():
    print("Online.")

bot.run("NjU3MDg0MDI5ODk5NjM2NzU3.XfsDPw.pXu5nv6ETfdxosvhw7WhnetIzjw")
