def spam_prompt(email: str) -> str:
    return f"""
Classify the following email as SPAM or NOT_SPAM.

Email:
{email}

Answer only SPAM or NOT_SPAM.
"""

def summary_prompt(email: str) -> str:
    return f"""
You are an email summarization system.

Summarize the email in ONE short sentence.
Do NOT repeat the email.
Do NOT add new information.

Email:
{email}
"""


def action_prompt(email: str) -> str:
    return f"""
Based on the email below, suggest one action:
REPLY, ARCHIVE, or FLAG.

Email:
{email}

Answer with only one word.
"""
