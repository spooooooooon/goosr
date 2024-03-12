#!/usr/bin/env python3
import json
import logging
import os
import re
import traceback
import types
import random
import argparse
from asyncio import sleep
from random import choice, randint
from time import time
from dotenv import load_dotenv

load_dotenv()

import openai
import pydle  # https://github.com/Shizmob/pydle
import requests
import yaml
from jellyfish import jaro_winkler_similarity

#
# Uses openai's api to respond to prompts and chat dialogs.
#
# Keeps a buffer of the last n lines in each room, and uses
# that buffer as the prompt to generate a response with a given personality.
#

configPath = "goosr.yaml"

logLevel = "INFO"  # DEBUG INFO WARN

# set api key for openai from environment variable
openai.api_key = os.getenv("oaik")

# accept prompts and commands from these nicks
with open("admin_users") as file:
    adminList = file.read().splitlines()

# ignore all messages from these nicks
with open("ignore_users") as file:
    ignoreList = file.read().splitlines()

# ignore messages containing these words
with open("ignore_words") as file:
    ignoredWords = file.read().splitlines()

with open("moods") as file:
    moods = file.read().splitlines()

def openaiRequest(prompt, stop="", max_tokens=35):
    """ wrapper function for openai completion that just returns the best answer as a string """

    # The new gpt-3.5-turbo
    model = ai.model

    # Varying the base prompt with existing chats might work better, although this can be optional
    # we need to check first if we have the correct values. Goosr has this but goobr does not.
    if '### EXAMPLE CHATS' in basePrompt and exampleChats:
        random_lines = random.sample(exampleChats, 300)
        random_lines_str = '\n'.join(random_lines)
        system_base_prompt = basePrompt.replace('### EXAMPLE CHATS', random_lines_str)
    else:
        system_base_prompt = basePrompt

    response = openai.ChatCompletion.create(
        model=model,
        temperature=ai.temperature,
        messages=[{"role": "system", "content": system_base_prompt}, {"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop=stop,
    )

    # print prompt to console for debugging
    globalLog.debug(prompt)
    # print full response to console for debugging
    globalLog.debug(response)
    finish_reason = response.choices[0].finish_reason

    # select the top answer and replace blank lines with a newline
    answer = response['choices'][0]['message']['content'].strip().replace("\n\n", "\n")

    if finish_reason != "stop":
        if any(punct in answer for punct in ["! ", "? ", ". "]):
            globalLog.warning(
                f"Did not hit a stop token, trimming this answer. Finish reason: {finish_reason}. Answer was: {answer}"
            )
            answer = re.split("\? |\. |\! ", answer)[:-1]
            answer = ". ".join(answer)

    return answer


def randomWord(lang="en", length=8):
    """ Requests a random word from herokuapp API. """
    url = f"https://random-word-api.herokuapp.com/word?length={length}&lang={lang}&number=1"
    words = json.loads(requests.get(url).text)
    return words[0]


# main class


class goosr(pydle.Client):
    """goosr"""

    async def on_connect(self):
        """ Callback called when the client connects. """
        await super().on_connect()
        # join channels on connect
        channels = self.joinChannels
        self.log.info(f"Connected to server. Joining {channels}")
        for channel in channels:
            await self.join(channel)
            await sleep(1)
        # self.joinChannels = []

        # if globalCfg.randomPersonality:
        #     personality, accent = await self.randomizePersonality()
        #     prompt = f"Generate a {personality} response {accent}:"
        #     self.log.info(f"Randomized personlity. New prompt: {prompt}")

    async def on_join(self, channel, user):
        """ Callback called when a user, possibly the client, joined a channel. """
        await super().on_join(channel, user)
        # when we join a channel, initialize the chat buffer
        if user == self.nickname:
            if channel in self.channels.keys():
                self.log.info(f"Joined {channel}")
            if channel in self.rejoining:
                self.rejoining.remove(channel)

    async def on_kick(self, channel, target, by, reason=None):
        """ Callback called when a user, possibly the client, was kicked from a channel. """
        if target == self.nickname:
            self.log.info(f"Kicked from {channel} by {by}")
            if (globalCfg.autoRejoin):
                self.log.debug(
                    f"Waiting {globalCfg.autoRejoinDelay}s before rejoining {channel}"
                )
                await sleep(globalCfg.autoRejoinDelay)
                await self.join(channel)

    async def on_ctcp_version(self, by, target, contents):
        """ Callback called when a user, possibly the client, received a CTCP VERSION request. """
        if target == self.nickname:
            self.log.warning(f"Received VERSION from {by}")
            if globalCfg.ctcp_version:
                await self.ctcp_reply(by, "VERSION", globalCfg.ctcp_version)

    async def on_raw_338(self, message):
        """ignore this message"""
        ...

    async def on_raw_474(self, message):
        """cannot join channel; attempt to rejoin if enabled"""
        if len(message.params) == 3:
            target, channel, message = message.params
            while (globalCfg.autoRejoin):
                if channel in self.rejoining:
                    return
                else:
                    self.rejoining.append(channel)
                    while ((channel not in self.channels.keys()) and (channel in self.rejoining)):
                        self.log.debug(
                            f"Waiting {globalCfg.autoRejoinDelay}s before rejoining {channel}"
                        )
                        await sleep(globalCfg.autoRejoinDelay)
                        await self.join(channel)

    async def on_private_message(self, target, source, message):
        """ Callback called when a user, possibly the client, received a private message. """
        await super().on_private_message(target, source, message)

        global adminList
        global ignoredWords
        global ignoreList

        max_tokens = ai.max_tokens
        commandChar = globalCfg.commandChar
        message = message.strip()

        if (source in adminList) and (target == self.nickname):
            # admin is talking to me

            # check for commands
            # command example:
            # ~command parameters

            if message[:1] == commandChar:

                try:
                    command, *parameters = message[1:].split(" ", 1)
                except ValueError:
                    await self.message(source, "Not enough parameters.")
                    return
                except:
                    return
                else:
                    if command:
                        if command == "forget":
                            # clear the buffer
                            # ~forget #channel1 #channel2 #channel3
                            if parameters:
                                channels = parameters[0].split(" ")
                            else:
                                channels = list(self.channels.keys())
                            self.log.info(f"Clearing buffer for {channels}.")
                            for channel in channels:
                                self.channels[channel]["chatBuffer"] = []
                            await self.message(
                                source, f"I know nothing about {channels}"
                            )
                        elif command == "join":
                            # join channels
                            # ~join #channel1 #channel2 #channel3
                            if parameters:
                                channels = parameters[0].split(" ")
                                self.log.info(f"Joining {channels}")
                                for channel in channels:
                                    await self.join(channel)
                        elif command == "part":
                            # part channels
                            # ~part #channel1 #channel2 #channel3
                            if parameters:
                                channels = parameters[0].split(" ")
                            else:
                                channels = list(self.channels.keys())
                            self.log.info(f"Parting {channels}")
                            for channel in channels:
                                await self.part(channel)
                        elif command == "nick":
                            # change nick
                            # ~nick newNick
                            newNick = parameters[0]
                            self.log.info(f"Changing nick to {newNick}")
                            await self.set_nickname(newNick)
                        elif command == "say":
                            # send a message to a channel
                            # ~say #channel i am goosr!
                            channel, *message = parameters[0].split(" ", 1)
                            if channel and message:
                                await self.message(channel, message[0])
                            else:
                                await self.message(source, "Not enough parameters.")
                        elif command == "kill":
                            # kill goosr :(
                            # ~kill
                            self.log.info(f"Received kill command.")
                            os._exit(0)
                        # elif command == "randomize":
                        #     # randomize personality and accent
                        #     # python is for straight white males
                        #     # ~randomize
                        #     personality, accent = await self.randomizePersonality()
                        #     prompt = f"Generate a {personality} response {accent}:"
                        #     await self.message(source, f"New prompt: {prompt}")
                        #
                        #     # display a sample
                        #     channel = choice(list(self.channels))
                        #     chatBuffer = self.channels[channel]["chatBuffer"]
                        #     prompt = await self.create_prompt(
                        #         chatBuffer,
                        #         personality=self.moods,
                        #         accent=self.accent,
                        #     )
                        #     answer = openaiRequest(prompt, " \n", max_tokens)
                        #     answer = answer.lower().replace("\n", " ")
                        #     answer = await self.strip_answer(answer)
                        #     if await self.filter_answer(answer):
                        #         self.log.warning(
                        #             f"Answer was rejected: {answer}")
                        #         return
                        #     if answer:
                        #         await self.message(
                        #             source, f"Sample response for {channel}: {answer}"
                        #         )
                        #         return
                        #     else:
                        #         await self.message(
                        #             source,
                        #             f"No good response returned response for {channel}",
                        #         )
                        elif command == "reload":
                            # reload global config
                            # ~reload
                            if await self.reloadGlobals():
                                await self.message(source, "Reloaded configuration.")
                        elif command == "status":
                            # check status
                            # ~status
                            status = f"P/A: '{self.moods}' Channels: {list(self.channels.keys())} Rejoining: {self.rejoining}"
                            await self.message(source, status)

        if await self.filter_message(message, target, source):
            return

        # check if message should be filtered
        # check if we have a pm buffer started for this user
        # if not, create a channel buffer placeholder and a pm buffer
        # add the line to the user's pm buffer
        # save the user's pm buffer
        # generate a reponse and return

    async def on_channel_message(self, target, source, message):
        await super().on_channel_message(target, source, message)

        global ignoreList
        global ignoredWords
        global adminList

        message = message.strip()

        if target in self.channels.keys():

            try:
                userBuffer = self.userBuffers[source]
            except KeyError:
                userBuffer = {source: {target: [], "privateBuffer": []}}

            if target not in userBuffer[source].keys():
                userBuffer[source][target] = []

            # check if this channel has a buffer, if not we can initialize it
            try:
                chatBuffer = self.channels[target]["chatBuffer"]
            except KeyError:
                chatBuffer = []

            if await self.filter_message(message, target, source, userBuffer[source][target]):
                self.log.debug(f"Message filtered: {source}: {message}")
                return

            # take the last n lines from the buffers and append the newest line

            userBuffer[source][target].append(
                f"{source}: {message}".strip().replace(
                    f"{self.nickname}: ", "")
            )
            userBuffer[source][target] = userBuffer[source][target][-1 * (globalCfg.chanBufferLength):]
            self.log.debug(
                f"Buffer for {source} in {target}:\n {userBuffer[source][target]}"
            )
            chatBuffer.append(f"{source}: {message}".strip())
            chatBuffer = chatBuffer[-1 * (globalCfg.chanBufferLength):]
            self.log.debug(f"Buffer for {target}:\n {chatBuffer}")

            # save the modified buffers back to the client
            self.userBuffers[source] = userBuffer
            self.channels[target]["chatBuffer"] = chatBuffer

            # check channel modes and make sure we can speak
            chModes = self.channels[target]['modes']
            if set(["m", "v"]).issubset(chModes.keys()):
                try:
                    if chModes["m"] and self.nickname not in chModes["v"]:
                        return
                except KeyError:
                    self.log.warning(
                        f"Got that key error again. chModes: {chModes}")
                    return

            # roll dice and check if we should reply
            diceRoll = randint(1, 100)
            self.log.debug(f"Dice Roll: {diceRoll}")

            chance = globalCfg.replyChance

            if self.nickname.lower() in message.lower():
                if (source in adminList):
                    chance = 100
                else:
                    chance = globalCfg.directReplyChance
                # chatBuffer = userBuffer[source][target]
                # calculate delay timer based on wpm setting to 'read' the line twice as fast as we type
                delayTimer = len(message.split(" ")) / (globalCfg.wpm / 30)
                await sleep(5.55 + delayTimer)

            if diceRoll <= chance:

                # create a prompt
                if self.nickname.lower() in message.lower():
                    prompt = await self.create_prompt(
                        userBuffer[source][target] + chatBuffer, mood=self.moods
                    )
                else:
                    prompt = await self.create_prompt(
                        chatBuffer, mood=self.moods
                    )

                # use newline as the stop token
                answer = openaiRequest(prompt, " \n", ai.max_tokens)
                # answer = answer.lower().split("\n")[0]
                answer = answer.lower().replace("\n", ". ")
                answer = await self.strip_answer(answer)
                if answer:

                    for line in re.split("\. |\? |\! ", answer):
                        if await self.check_similarity(line, chatBuffer):
                            self.log.warning(
                                f"Answer rejected for similarity in {target}: {line}"
                            )
                            continue
                        if await self.filter_answer(line):
                            self.log.warning(f"Answer was rejected: {line}")
                            continue

                        # add my own lines to the channel buffer
                        chatBuffer.append(f"{self.nickname}: {line}".strip())
                        chatBuffer = chatBuffer[-1 * (globalCfg.chanBufferLength):]
                        self.log.debug(f"Buffer for {target}:\n {chatBuffer}")

                        # save the modified buffer
                        self.channels[target]["chatBuffer"] = chatBuffer

                        # log training data to file if enabled
                        if globalCfg.logTrainingData:
                            trainingData = {}
                            trainingPrompt = f"{prompt}\n\n###\n\n"
                            trainingData["prompt"] = trainingPrompt
                            # space in front of completion according to docs
                            trainingData["completion"] = f" {line}\n"
                            # dump dict to json
                            trainingData = json.dumps(trainingData)
                            self.trainingLog.info(trainingData)

                        # calculate delay timer based on wpm setting
                        delayTimer = 5.5555 + len(line.split(" ")) / (globalCfg.wpm / 60)

                        # wait until we are done typing other messages
                        while self.typing:
                            await sleep(1)
                        self.typing = True
                        self.log.info(
                            f"[{target}] Replying to {source} (after sleeping for {int(delayTimer)}s): {line}"
                        )
                        await sleep(delayTimer)
                        await self.message(target, line)
                        self.typing = False
                return

    async def filter_message(self, message, target, source, buffer=[]):
        """Filters the incoming message based on several criteria. Returns False if we don't want to filter."""
        global ignoreList
        global ignoredWords

        commandChar = globalCfg.commandChar
        message = message.strip()

        # ignore blank messages
        if not message:
            return True

        # ignore our own messages
        if source == self.nickname:
            return True

        # check ignore list
        if source in ignoreList:
            return True

        # check ignored words list
        if any(word for word in ignoredWords if (word in message)):
            return True

        # filter out commands sent to other bots
        if (message[:1] in ["!", "@", "?", "`", ",", "."]) and (message[:1] not in commandChar):
            return True

        # filter out messages that contain non-ascii characters or ansi control sequences
        if self.re_ansi.search(message) or self.re_non_ascii.search(message):
            return True

        # filter messages that dont have at least one english letter
        if not self.re_alpha.search(message):
            return True

        # filter messages with more than one colon, this seemed to be causing issues with the prompt
        if len(message.split(":")) > 2:
            return True

        if buffer:
            if await self.check_similarity(message, buffer):
                return True

        # Return false if we don't want to filter
        return False

    async def filter_answer(self, answer):
        """Filters the answer based on several criteria. Returns False if we don't want to filter."""
        # filter answers with more than one colon. probably a bad answer
        if len(answer.split(":")) > 2:
            return True

        # filter answers that dont have at least one english letter
        if not self.re_alpha.search(answer):
            return True

        # return false if we don't want to filter this answer
        return False

    async def strip_answer(self, answer):
        """Cleans up the answer to make it more presentable."""
        answer = answer.strip()

        # remove quotes from the answer
        for char in ['"', "'"]:
            answer = answer.replace(char, "")

        # if the answer had colons just grab everything to the right of it
        answer = answer.split(":")[-1:][0].strip()

        # strip our own nick if it was in there
        # answer = answer.replace(self.nickname, "")

        return answer

    async def create_prompt(self, buffer, mood):
        """Combine the buffer with the personality and mood to create a prompt for the API."""
        header = f"Your name is {self.nickname}. The following is a chat log between you and other chatters:"
        chatlog = "\n".join(buffer)
        if self.nickname in buffer[-1]:
            disposition = f"Responding to {buffer[-1].split(': ', 1)[0]}, create a {mood} response:"
        else:
            disposition = f"Without mentioning your own name, create a {mood} response:"

        prompt = "\n\n".join([header, chatlog, disposition])
        return prompt

    async def randomizePersonality(self):
        """Set a random personality and mood from the lists."""
        personalities = os.listdir("personalities")
        personality = choice(personalities)

        return None

    async def check_similarity(self, answer, buffer):
        """Uses the Jaro-Winkler function to determine the similarity between two strings.
            Returns true if the similarity is over the threshold from settings.
            A setting of 1.0 matches identical strings only."""
        for line in buffer:
            if ": " in line:
                line = line.split(": ", 1)[1]
            jaro_winkler = jaro_winkler_similarity(
                line.lower(), answer.lower())
            if jaro_winkler > globalCfg.reject_similarity:
                return True
        return False

    async def reloadGlobals(self):
        """Reload global configuration and AI settings."""
        global adminList, ignoreList, ignoredWords, moods, globalCfg, ai, basePrompt, exampleChats

        with open(configPath, "r") as file:
            config = yaml.safe_load(file)

        for name, value in config["global"].items():
            globalCfg.__setattr__(name, value)

        for name, value in config["global"]["ai"].items():
            ai.__setattr__(name, value)

        with open("admin_users") as file:
            adminList = file.read().splitlines()

        with open("ignore_users") as file:
            ignoreList = file.read().splitlines()

        with open("ignore_words") as file:
            ignoredWords = file.read().splitlines()

        with open("moods") as file:
            moods = file.read().splitlines()

        # This file must exist as it's used as the system prompt for all API requests
        file_path = "personalities/" + globalCfg.personality + "/system_prompt.txt"
        if os.path.isfile(file_path):
            with open(file_path) as file:
                basePrompt = file.read()
        else:
            raise FileNotFoundError(f"No such file or directory: '{file_path}'")

        # Optional example chats, the  above base prompt personality must have ### EXAMPLE CHATS in it
        # When each request is made to OpenAI, we will vary the example chats.
        file_path = "personalities/" + globalCfg.personality + "/example_chats.txt"
        if os.path.isfile(file_path):
            with open(file_path) as file:
                exampleChats = file.read().splitlines()
        else:
            exampleChats = None

        return True


if __name__ == "__main__":
    global basePrompt, exampleChats

    if not os.path.isfile(".env"):
        print("Copy .env.example to .env and put your openai or compatible key to continue")
        # noinspection PyUnresolvedReferences
        os._exit(0)

    simple = types.SimpleNamespace()
    pond = pydle.ClientPool()
    # set up logging
    globalLog = logging.getLogger("goosr")
    globalLog.setLevel(logLevel)
    ch = logging.StreamHandler()
    ch.setLevel(logLevel)
    formatter = logging.Formatter("%(name)s.%(levelname)s >> %(message)s")
    ch.setFormatter(formatter)
    globalLog.addHandler(ch)

    # load the config
    with open(configPath, "r") as file:
        config = yaml.safe_load(file)

    # create a simple namespace and apply the settings as attributes.
    globalCfg = simple
    for name, value in config["global"].items():
        globalCfg.__setattr__(name, value)

    ai = simple
    for name, value in config["global"]["ai"].items():
        ai.__setattr__(name, value)

    # Check to see if the personality has been specified on the command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--personality', type=str, help='Overwrite personality')
    args = parser.parse_args()

    if args.personality:
        globalCfg.personality = args.personality

    # Load the personality from file
    # This file must exist as it's used as the system prompt for all API requests
    file_path = "personalities/" + globalCfg.personality + "/system_prompt.txt"
    if os.path.isfile(file_path):
        with open(file_path) as file:
            basePrompt = file.read()
    else:
        raise FileNotFoundError(f"No such file or directory: '{file_path}'")

    # Optional example chats, the  above base prompt personality must have ### EXAMPLE CHATS in it
    # When each request is made to OpenAI, we will vary the example chats.
    file_path = "personalities/" + globalCfg.personality + "/example_chats.txt"
    if os.path.isfile(file_path):
        with open(file_path) as file:
            exampleChats = file.read().splitlines()
    else:
        exampleChats = None

    for client in config["clients"].items():
        if client[1]["enabled"]:
            # If the values are blank in the config, we will default to the personality
            if client[1]["ident"]["nick"] == '':
                nick = globalCfg.personality
            else:
                nick = client[1]["ident"]["nick"]

            if client[1]["ident"]["user"] == '':
                user = globalCfg.personality
            else:
                user = client[1]["ident"]["user"]

            if client[1]["ident"]["real"] == '':
                real = globalCfg.personality
            else:
                real = client[1]["ident"]["real"]

            # randomize ident
            if globalCfg.randomIdent["enabled"]:
                globalLog.info("Getting random words for ident...")
                minLength = globalCfg.randomIdent["minLength"]
                maxLength = globalCfg.randomIdent["maxLength"]
                lang = globalCfg.randomIdent["lang"]
                nick = randomWord(
                    lang=lang, length=randint(minLength, maxLength))
                if globalCfg.randomIdent["sameAsUserAndReal"]:
                    user = real = nick
                else:
                    user = randomWord(
                        lang=lang, length=randint(minLength, maxLength))
                    real = randomWord(
                        lang=lang, length=randint(minLength, maxLength))

            bot = goosr(nickname=nick, username=user, realname=real)

            # set up the client's logger
            bot.name = client[0]
            bot.log = logging.getLogger(f"goosr.{bot.name}.{nick}")
            bot.log.setLevel(logLevel)
            bot.ch = logging.StreamHandler()
            bot.ch.setLevel(logLevel)
            bot.ch.setFormatter(formatter)
            bot.log.addHandler(bot.ch)
            bot.log.propagate = False
            if globalCfg.logTrainingData:
                # set up the training data logger
                bot.trainingLog = logging.getLogger(
                    f"trainingLog.{bot.name}.{nick}")
                bot.trainingLog.setLevel("INFO")
                plainFormat = logging.Formatter("%(message)s")
                fh = logging.FileHandler(
                    f"training_data/{bot.trainingLog.name}.{time()}.json")
                fh.setFormatter(plainFormat)
                fh.setLevel("INFO")
                bot.trainingLog.addHandler(fh)

            for category, settings in client[1].items():
                bot.__setattr__(category, settings)

            # connection settings
            server = bot.server["host"]
            port = bot.server["port"]
            tls = bot.server["tls"]
            source_hostname = bot.server["source_hostname"]

            bot.users = {
                "nickname": nick,
                "username": user,
                "realname": real,
                "hostname": source_hostname,
            }

            # initialize some variables and create regex engines
            bot.userBuffers = {}
            bot.rejoining = []
            bot.typing = False
            bot.moods = moods[0]
            bot.re_alpha = re.compile(r"[a-zA-Z]")

            # re to match all non-ascii
            bot.re_non_ascii = re.compile("[^\x00-\x7F]+")

            # re to match ansi control sequences
            bot.re_ansi = re.compile(
                r"(\x9B|\x1B\[)[0-?]*[ -/]*[@-~]", flags=re.IGNORECASE
            )

            # let's get going
            globalLog.info(
                f"Connecting client ({bot.name}.{nick}) to {server}:{port}...")
            if source_hostname:
                globalLog.info(f"Using host {source_hostname}...")
                pond.connect(
                    bot, server, port=port, tls=tls, source_address=[
                        source_hostname, 0]
                )
            else:
                pond.connect(bot, server, port=port, tls=tls)
    try:
        pond.handle_forever()
    except Exception as x:
        globalLog.error(f"Uncaught exception: " + ''.join(traceback.format_exception(None, x, x.__traceback__)))

# todo:

# tokenize prompts with transformers library and count the tokens, use number of tokens to control buffer length precisely.


# flood throttle

# track pm convos and respond
# stats logger
