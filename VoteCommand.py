from discord.ext import commands
import discord

class VoteCommand():
    def __init__(self, name, v_start=None, v_success=None, v_fail=None):
        self.name      = name
        self.cmd       = cmd
        self.v_start   = v_start
        self.v_success = v_success
        self.v_fail    = v_fail
    
    def start(self, ctx, args):
        if self.v_start == None:
            await ctx.send("Vote started for " + name "!")
        else:
            self.v_start(ctx, args)
    
    def cmd(self, args):
        if self.v_success == None:
            await ctx.send("Vote for " + name + " succeeded!")
        else:
            self.v_success(ctx, args)
    
    def fail(self, args):
        if self.v_fail == None:
            await ctx.send("Vote for " + name + " failed!")
        else:
            self.v_fail(ctx, args)
