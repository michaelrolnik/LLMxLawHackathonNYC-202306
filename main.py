from agent_sim.player import Player
from agent_sim.prompts_library import PRIMARY_INCEPTION_PROMPT, HUMAN_INCEPTION_PROMPT
from langchain.chat_models import ChatAnthropic
from langchain.chat_models import ChatOpenAI
from langchain.llms import Baseten, Modal
from pprint import pprint
import os
import json
import re


def simulation_CoT(player1: Player, player2: Player, initial_message: str, num_turns: int = 10,
                   max_words_allowed=10000):
    """
    A function that simulates a conversation between two players.

    Parameters:
        player1 (Player): The first player that initiates the conversation
        player2 (Player): The second player
        initial_message (str): The initial message to start the conversation.
        num_turns (int): The number of turns in the conversation. Default is 10.
        max_tokens_allowed (int): the maximum number of tokens allowed for a simulation

    Returns:
        List[str]: The conversation history.
    """
    # Start with an empty conversation history
    conversation_history = []
    conversation_length = 0

    # Set initial roles (who is the responder and who is the initiator)
    initiator, responder = player1, player2

    # Begin conversation with the provided initial message
    message = initial_message
    conversation_history.append(f"{initiator.role_name}: {message}")

    for i in range(num_turns):
        if conversation_length > max_words_allowed:
            return conversation_history
        # The responder generates a response and does NOT add to its memory
        raw_response = responder.respond(initiator.role_name, message, remember=False)
        raw_response = ".".join(raw_response.split("\n"))

        # Extract the message and thoughts from the response
        match = re.match(r'.*__Thought__:?(?P<thought>.*)__Message__:?(?P<message>.*)', raw_response, re.IGNORECASE)
        if match is None:
            match = re.match(r'.*_Thought_:?(?P<thought>.*)_Message_:?(?P<message>.*)', raw_response, re.IGNORECASE)
        if match is None:
            match = re.match(r'.*Thought:?(?P<thought>.*)Message:?(?P<message>.*)', raw_response, re.IGNORECASE)
        if match is None:
            match = re.match(r'(?P<thought>.*)Message:?(?P<message>.*)', raw_response, re.IGNORECASE)

        if match is not None:
            response_message = match.group("message")
            response_thought = match.group("thought")
        else:
            response_message = raw_response
            response_thought = "..."

        print("\n[{:>3}] {}".format(i, "-" * 100))
        print(f"{responder.role_name}'s thoughts: {response_thought}")
        print(f"{responder.role_name}'s message: {response_message}")

        conversation_history.append(f"{responder.role_name}'s thoughts: {response_thought} \n\n")
        conversation_history.append(f"{responder.role_name}: {response_message}")
        conversation_length += len(response_message.split(" "))

        # Manually add the response_message to responder's memory
        responder.add_to_memory(initiator.role_name, message)
        responder.add_to_memory(responder.role_name, response_message)

        # Swap roles for the next turn
        initiator, responder = responder, initiator
        # The new message for the next turn is the response from this turn
        message = response_message

    return conversation_history

def noChat():
    chat = Modal(
        endpoint_url="https://nomosartificial--falcon-40b-instruct-get.modal.run",
        model_kwargs={
            "temperature": 0.9,
            "max_new_tokens": 128})
    return chat

def main():
    chat = lambda: noChat()

    if os.getenv("ANTHROPIC_API_KEY") is not None:
        chat = lambda: ChatAnthropic()

    if os.getenv("OPENAI_API_KEY") is not None:
        chat = lambda: ChatOpenAI(model_name='gpt-4')

    with open("scenarios.json") as file:
        scenarios = json.load(file)

    inception_format = """
    return the result in a format like below:
    __Thought__: (this is your thoughts)
    __Message__: (this is what you would actually say)
    """

    for scenario in scenarios["scenarios"]:
        print("scenario - [{}]".format(scenario["name"]))

        inception = '\n'.join(scenario["primary"] + [inception_format])
        agentPrimary = Player(chat(), chat(), inception, "Primary")

        inception = '\n'.join(scenario["human"] + [inception_format])
        agentHuman = Player(chat(), chat(), inception, "Human")

        # NOTE:
        # NOTE: Should we feed scenario["background"] into the primary agent
        # NOTE: I guess we should split it into data known by human and primary
        # NOTE: agents and add to the "human"/"primary" sections of the scenarios.json file
        # NOTE:

        simulation_CoT(agentHuman, agentPrimary, scenario["initial"], 2)


if __name__ == "__main__":
    main()
