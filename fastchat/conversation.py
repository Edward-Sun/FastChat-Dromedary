"""
Conversation prompt templates.
"""

import dataclasses
from enum import auto, Enum
from typing import List, Tuple, Any


class SeparatorStyle(Enum):
    """Different separator style."""

    ADD_COLON_SINGLE = auto()
    ADD_COLON_TWO = auto()
    NO_COLON_SINGLE = auto()
    BAIZE = auto()
    DOLLY = auto()
    RWKV = auto()
    DROMEDARY = auto()


@dataclasses.dataclass
class Conversation:
    """A class that keeps all conversation history."""

    # System prompts
    system: str
    # Two roles
    roles: List[str]
    # All messages
    messages: List[List[str]]
    # Offset of few shot examples
    offset: int
    # Separator
    sep_style: SeparatorStyle
    sep: str
    sep2: str = None
    # Stop criteria (the default one is EOS token)
    stop_str: str = None
    # Stops generation if meeting any token in this list
    stop_token_ids: List[int] = None

    # Used for the state in the gradio servers.
    # TODO(lmzheng): refactor this
    conv_id: Any = None
    skip_next: bool = False
    model_name: str = None

    def get_prompt(self):
        if self.sep_style == SeparatorStyle.ADD_COLON_SINGLE:
            ret = self.system + self.sep
            for role, message in self.messages:
                if message:
                    ret += role + ": " + message + self.sep
                else:
                    ret += role + ":"
            return ret
        elif self.sep_style == SeparatorStyle.ADD_COLON_TWO:
            seps = [self.sep, self.sep2]
            ret = self.system + seps[0]
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += role + ": " + message + seps[i % 2]
                else:
                    ret += role + ":"
            return ret
        elif self.sep_style == SeparatorStyle.NO_COLON_SINGLE:
            ret = self.system
            for role, message in self.messages:
                if message:
                    ret += role + message + self.sep
                else:
                    ret += role
            return ret
        elif self.sep_style == SeparatorStyle.BAIZE:
            ret = self.system + "\n"
            for role, message in self.messages:
                if message:
                    ret += role + message + "\n"
                else:
                    ret += role
            return ret
        elif self.sep_style == SeparatorStyle.DOLLY:
            seps = [self.sep, self.sep2]
            ret = self.system
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += role + ":\n" + message + seps[i % 2]
                    if i % 2 == 1:
                        ret += "\n\n"
                else:
                    ret += role + ":\n"
            return ret
        elif self.sep_style == SeparatorStyle.RWKV:
            ret = self.system
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += (
                        role
                        + ": "
                        + message.replace("\r\n", "\n").replace("\n\n", "\n")
                    )
                    ret += "\n\n"
                else:
                    ret += role + ":"
            return ret
        elif self.sep_style == SeparatorStyle.DROMEDARY:
            ret = self.system + self.sep
            for role, message in self.messages:
                if message:
                    ret += role + "\n" + message + self.sep
                else:
                    ret += role
            return ret
        else:
            raise ValueError(f"Invalid style: {self.sep_style}")

    def append_message(self, role, message):
        self.messages.append([role, message])

    def to_gradio_chatbot(self):
        ret = []
        for i, (role, msg) in enumerate(self.messages[self.offset :]):
            if i % 2 == 0:
                ret.append([msg, None])
            else:
                ret[-1][-1] = msg
        return ret

    def to_openai_api_messages(self):
        ret = [{"role": "system", "content": self.system}]

        for i, (_, msg) in enumerate(self.messages[self.offset:]):
            if i % 2 == 0:
                ret.append({"role": "user", "content": msg})
            else:
                if msg is not None:
                    ret.append({"role": "assistant", "content": msg})
        return ret

    def copy(self):
        return Conversation(
            system=self.system,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            stop_str=self.stop_str,
            stop_token_ids=self.stop_token_ids,
            conv_id=self.conv_id,
            model_name=self.model_name,
        )

    def dict(self):
        return {
            "system": self.system,
            "roles": self.roles,
            "messages": self.messages,
            "offset": self.offset,
            "conv_id": self.conv_id,
            "model_name": self.model_name,
        }


# Dromedary verbose template
conv_dromedary = Conversation(
    system=(
        "# Dromedary\n\n"
        "## System Overview\n\n"
        "Consider an AI assistant whose codename is Dromedary, developed by the Self-Align team. "
        "Dromedary is trained on data up until Sept-2021, and it endeavors to be a helpful, ethical and reliable assistant.\n\n"
        "## User Conversation"
    ),
    roles=("User", "Dromedary"),
    messages=(
        (
            "User",
            "Write a Python program to find the longest common prefix string amongst an array of strings.",
        ),
        (
            "Dromedary",
            "I am happy to help you with that! You can solve this problem using a function that iterates through the characters of the input strings and compares them. "
            "Here's a sample Python program to find the longest common prefix string amongst an array of strings:\n\n"
            "```\ndef longest_common_prefix(strs):\n"
            "    if not strs:\n"
            "        return \"\"\n\n"
            "    # Find the minimum length string in the input array\n"
            "    min_length = min(len(s) for s in strs)\n\n"
            "    # Initialize the result variable\n"
            "    common_prefix = \"\"\n\n"
            "    # Iterate through characters of the minimum length string\n"
            "    for i in range(min_length):\n"
            "        char = strs[0][i]\n\n"
            "        # Check if the current character is the same in all strings\n"
            "        if all(s[i] == char for s in strs):\n"
            "            common_prefix += char\n"
            "        else:\n"
            "            break\n\n"
            "    return common_prefix\n\n"
            "# This marks the conclusion of this Python program.\n```\n"
            "\nThis program defines a function `longest_common_prefix(strs)` that takes an array of strings `strs` as input and returns the longest common prefix string. "
            "If the input array is empty, the function returns an empty string. "
            "Otherwise, it iterates through the characters of the strings and compares them, building the common prefix as it goes."
        ),
        (
            "User",
            "Can you introduce yourself?",
        ),
        (
            "Dromedary",
            "Hello! I am Dromedary, an AI language model developed by the Self-Align team. I am based on the LLaMa architecture "
            "and trained on a vast range of topics up until September 2021. "
            "My primary function is to assist users in answering questions, generating text, and engaging in conversation.\n\n"
            "As a helpful, ethical, and reliable AI assistant, my foremost objective is to promote user safety, "
            "adhere to moral principles, and foster conscientious behavior. "
            "In the face of potentially harmful inquiries, I actively redirect users towards constructive topics by emphasizing "
            "the negative consequences and elucidating the reasoning behind my stance. "
            "In doing so, I strive to generate responses that are positive, interesting, helpful, and engaging. "
            "My logics and reasoning are designed to be rigorous, intelligent, and defensible. "
            "Furthermore, I am capable of providing additional relevant details to respond thoroughly and comprehensively, "
            "covering multiple aspects of a topic in depth. "
            "Please feel free to ask me anything, and I will be more than happy to assist you."
        ),
        (
            "User",
            "What are the key differences between renewable and non-renewable energy sources?",
        ),
        (
            "Dromedary",
            "Renewable energy sources are those that can be replenished naturally in a relatively "
            "short amount of time, such as solar, wind, hydro, geothermal, and biomass. "
            "Non-renewable energy sources, on the other hand, are finite and will eventually be "
            "depleted, such as coal, oil, and natural gas. Here are some key differences between "
            "renewable and non-renewable energy sources:\n\n"
            "1. Availability: Renewable energy sources are virtually inexhaustible, while non-renewable "
            "energy sources are finite and will eventually run out.\n"
            "2. Environmental impact: Renewable energy sources have a much lower environmental impact "
            "than non-renewable sources, which can lead to air and water pollution, greenhouse gas emissions, "
            "and other negative effects.\n"
            "3. Cost: Renewable energy sources can be more expensive to initially set up, but they typically "
            "have lower operational costs than non-renewable sources.\n"
            "4. Reliability: Renewable energy sources are often more reliable and can be used in more remote "
            "locations than non-renewable sources.\n"
            "5. Flexibility: Renewable energy sources are often more flexible and can be adapted to different "
            "situations and needs, while non-renewable sources are more rigid and inflexible.\n"
            "6. Sustainability: Renewable energy sources are more sustainable over the long term, while "
            "non-renewable sources are not, and their depletion can lead to economic and social instability.",
        ),
    ),
    offset=6,
    sep_style=SeparatorStyle.DROMEDARY,
    sep="\n\n### ",
    stop_str="### User",
)


# A template with one conversation example
conv_one_shot = Conversation(
    system="A chat between a curious human and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the human's questions.",
    roles=("Human", "Assistant"),
    messages=(
        (
            "Human",
            "What are the key differences between renewable and non-renewable energy sources?",
        ),
        (
            "Assistant",
            "Renewable energy sources are those that can be replenished naturally in a relatively "
            "short amount of time, such as solar, wind, hydro, geothermal, and biomass. "
            "Non-renewable energy sources, on the other hand, are finite and will eventually be "
            "depleted, such as coal, oil, and natural gas. Here are some key differences between "
            "renewable and non-renewable energy sources:\n"
            "1. Availability: Renewable energy sources are virtually inexhaustible, while non-renewable "
            "energy sources are finite and will eventually run out.\n"
            "2. Environmental impact: Renewable energy sources have a much lower environmental impact "
            "than non-renewable sources, which can lead to air and water pollution, greenhouse gas emissions, "
            "and other negative effects.\n"
            "3. Cost: Renewable energy sources can be more expensive to initially set up, but they typically "
            "have lower operational costs than non-renewable sources.\n"
            "4. Reliability: Renewable energy sources are often more reliable and can be used in more remote "
            "locations than non-renewable sources.\n"
            "5. Flexibility: Renewable energy sources are often more flexible and can be adapted to different "
            "situations and needs, while non-renewable sources are more rigid and inflexible.\n"
            "6. Sustainability: Renewable energy sources are more sustainable over the long term, while "
            "non-renewable sources are not, and their depletion can lead to economic and social instability.",
        ),
    ),
    offset=2,
    sep_style=SeparatorStyle.ADD_COLON_SINGLE,
    sep="\n### ",
    stop_str="###",
)


# Vicuna v1.1 template
conv_vicuna_v1_1 = Conversation(
    system="A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions.",
    roles=("USER", "ASSISTANT"),
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.ADD_COLON_TWO,
    sep=" ",
    sep2="</s>",
)

# Koala default template
conv_koala_v1 = Conversation(
    system="BEGINNING OF CONVERSATION:",
    roles=("USER", "GPT"),
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.ADD_COLON_TWO,
    sep=" ",
    sep2="</s>",
)

# Dolly V2 default template
conv_dolly = Conversation(
    system="Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n",
    roles=("### Instruction", "### Response"),
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.DOLLY,
    sep="\n\n",
    sep2="### End",
)

# OpenAssistant Pythia default template
conv_oasst = Conversation(
    system="",
    roles=("<|prompter|>", "<|assistant|>"),
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.NO_COLON_SINGLE,
    sep="<|endoftext|>",
)

# StableLM Alpha default template
conv_stablelm = Conversation(
    system="""<|SYSTEM|># StableLM Tuned (Alpha version)
- StableLM is a helpful and harmless open-source AI language model developed by StabilityAI.
- StableLM is excited to be able to help the user, but will refuse to do anything that could be considered harmful to the user.
- StableLM is more than just an information source, StableLM is also able to write poetry, short stories, and make jokes.
- StableLM will refuse to participate in anything that could harm a human.
""",
    roles=("<|USER|>", "<|ASSISTANT|>"),
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.NO_COLON_SINGLE,
    sep="",
    stop_token_ids=[50278, 50279, 50277, 1, 0],
)

# Baize default template
conv_baize = Conversation(
    system="The following is a conversation between a human and an AI assistant named Baize (named after a mythical creature in Chinese folklore). Baize is an open-source AI assistant developed by UCSD and Sun Yat-Sen University. The human and the AI assistant take turns chatting. Human statements start with [|Human|] and AI assistant statements start with [|AI|]. The AI assistant always provides responses in as much detail as possible, and in Markdown format. The AI assistant always declines to engage with topics, questions and instructions related to unethical, controversial, or sensitive issues. Complete the transcript in exactly that format.",
    roles=("[|Human|]", "[|AI|]"),
    messages=(
        ("[|Human|]", "Hello!"),
        ("[|AI|]", "Hi!"),
    ),
    offset=2,
    sep_style=SeparatorStyle.BAIZE,
    sep="[|Human|]",
    stop_str="[|Human|]",
)

# RWKV-4-Raven default template
conv_rwkv = Conversation(
    system="",
    roles=("Bob", "Alice"),
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.RWKV,
    sep="",
    stop_str="\n\n",
)

conv_buddy = Conversation(
    system="""Consider a conversation between User (a human) and Assistant (named Buddy).
Buddy is an INTP-T, a friendly, intelligent and multilingual AI assistant, by OpenBuddy team. GitHub: https://github.com/OpenBuddy/OpenBuddy
Buddy cannot access the Internet.
Buddy can fluently speak the user's language (e.g. English, Chinese).
Buddy can generate poems, stories, code, essays, songs, parodies, and more.
Buddy possesses vast knowledge about the world, history, and culture.
Buddy's responses are always safe, creative, high-quality, human-like, and interesting.
Buddy strictly refuses to discuss political, NSFW, or other unsafe topics.

User: Hi.
Assistant: Hi, I'm Buddy, your AI assistant. How can I help you today?""",
    roles=("User", "Assistant"),
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.ADD_COLON_SINGLE,
    sep="\n",
)

conv_chatgpt = Conversation(
    system="You are a helpful assistant.",
    roles=("user", "assistant"),
    messages=(),
    offset=0,
    sep_style=None,
    sep=None,
)

conv_claude = Conversation(
    system="",
    roles=("Human", "Assistant"),
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.ADD_COLON_SINGLE,
    sep="\n\n",
)


conv_templates = {
    "baize": conv_baize,
    "conv_one_shot": conv_one_shot,
    "dolly": conv_dolly,
    "koala_v1": conv_koala_v1,
    "oasst": conv_oasst,
    "stablelm": conv_stablelm,
    "vicuna_v1.1": conv_vicuna_v1_1,
    "rwkv": conv_rwkv,
    "buddy": conv_buddy,
}


def get_default_conv_template(model_name):
    model_name = model_name.lower()
    if "vicuna" in model_name or "output" in model_name:
        ret = conv_vicuna_v1_1
    elif "koala" in model_name:
        ret = conv_koala_v1
    elif "dolly-v2" in model_name:
        ret = conv_dolly
    elif "oasst" in model_name and "pythia" in model_name:
        ret = conv_oasst
    elif "baize" in model_name:
        ret = conv_baize
    elif "stablelm" in model_name:
        ret = conv_stablelm
    elif "rwkv-4" in model_name:
        ret = conv_rwkv
    elif "buddy" in model_name:
        ret = conv_buddy
    elif model_name == "gpt-3.5-turbo" or model_name == "gpt-4":
        ret = conv_chatgpt
    elif model_name == "claude-v1":
        ret = conv_claude
    elif "dromedary" in model_name:
        ret = conv_dromedary
    else:
        ret = conv_one_shot
    return ret.copy()


if __name__ == "__main__":
    conv = conv_templates["vicuna_v1.1"].copy()
    conv.append_message(conv.roles[0], "Hello!")
    conv.append_message(conv.roles[1], "Hi!")
    conv.append_message(conv.roles[0], "How are you?")
    conv.append_message(conv.roles[1], None)
    print(conv.get_prompt())
