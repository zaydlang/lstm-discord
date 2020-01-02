import pickle

import discord
from discord.ext import commands
from numpy.random import choice

bot = commands.Bot(command_prefix='$')
bot.remove_command("help") # discord has its own stupid default help command

SIMON      = {}      # a map from Discordpy User to Markov Chain
ORDER      = 2       # ORDER of chain
WORD_LIMIT = 20      # maximum number of words in a sentence
ENDING_TAG = r'\end' # ending tag for processing

########## CLASSES ##########

# The variable 'chain' always refers to a Markov Chain
class MarkovChain:
    def __init__(self):
        # List of links
        self.chain = []
    
    # Adds an element to the chain
    def add_link(self, link):
        if link in self.chain:
            self.chain[self.chain.index(link)].strengthen()
        else:
            self.chain.append(link)

    # Adds sentence to given chain
    # More specifically, it firt converts the sentence to links.
    def process_sentence(self, sentence):
        history = []

        for word in sentence.split(" "):
            # Add entry to chain
            self.add_link(Link(history.copy(), word))
            history = self.update_history(history, word)

    def mimic(self):
        history   = [] # prompt for next word, let's start it off like this for now.
        num_words = 0  # number of words in current sentence
        sentence  = '' # current sentence
        
        while (len(history) == 0 or history[-1] != ENDING_TAG) and num_words <= WORD_LIMIT:
            # list of possible next links
            links = list(link for link in self.chain if link.source == history)
            if len(links) == 0:
                break

            # list of probatilities
            total_power = sum(link.power for link in links) # total power
            powers = list(link.power / total_power for link in links)

            next_word = choice(links, 1, p=powers)[0]
            if next_word.destination == ENDING_TAG:
                break

            sentence += " " + next_word.destination
            history = self.update_history(history, next_word.destination)
        
        return sentence

    def update_history(self, history, word):
        # If the history isn't full already, append the word
        if len(history) < ORDER:
            history.append(word)
        # else, just replace the last word.
        else:
            history.pop(0)
            history.append(word)
        
        return history

    def __str__(self):
        return '\n'.join(str(link) for link in self.chain)

# A link in the Markov Chain
class Link:
    def __init__(self, source, destination):
        self.source      = source
        self.destination = destination
        self.power       = 1
    
    def strengthen(self):
        self.power += 1
    
    def __eq__(self, other):
        return self.source == other.source and self.destination == other.destination
        
    def __str__(self):
        return "[" + ','.join(self.source) + "] -> " + self.destination + " Power: " + str(self.power)






############### METHODS ###############

@bot.command()
# Does a preliminary scan when bot is in a guild that has no associated Markov Chains.
async def scan(ctx, *args):
    num_messages = int(args[0])       # message limit
    counter      = 0                  # messages scanned
    guild        = ctx.guild          # server

    await ctx.send("im scanning yee")

    # gather messages for each person
    for channel in guild.channels:
        # we only care about text channels
        if str(channel.type) == 'text':
            async for message in channel.history(limit=num_messages):
                process(message)

    await ctx.send("doen scanning yay")

# message is discordpy Message
def process(message):
    user    = message.author
    mimicee = None

    # see if user is a mimicee.
    if user in SIMON.keys(): # ahaha very funny joke
        # good, they are.
        mimicee = SIMON[user]
    else:
        # okay lets make them one
        SIMON[user] = MarkovChain()
        mimicee     = SIMON[user]
    
    # append an ending tag and process
    mimicee.process_sentence(message.content + " " + ENDING_TAG)

@bot.command()
async def mimic(ctx, *args):
    mimicee  = ctx.message.mentions[0] # person to mimic
    sentence = SIMON[mimicee].mimic()
    await ctx.send(sentence)

'''
@bot.command()
# save simon
async def save(ctx, *args):
    # format data
    data = {}
    for mimicee in SIMON.keys(): # ahahhaha funny joke
        data[mimicee.author.name]

    with open('simon.pkl', 'wb') as handle:
        pickle.dump(SIMON, handle, protocol=pickle.HIGHEST_PROTOCOL)
    await ctx.send("saved.")

@bot.command()
# load simon
async def load(ctx, *args):
    with open('simon.pkl', 'wb') as handle:
        SIMON = pickle.load(handle) # oop i editted a constant oh no
    await ctx.send("loaded.")
'''



######## DISCORD BOT STUFF ########

@bot.event
async def on_ready():
    print("Online.")

bot.run("NjU3MDg0MDI5ODk5NjM2NzU3.XfsDPw.pXu5nv6ETfdxosvhw7WhnetIzjw")
