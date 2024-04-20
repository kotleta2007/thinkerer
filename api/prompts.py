import os
from dotenv import load_dotenv
import re

from groq import Groq

# get the api key
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
print(len(GROQ_API_KEY))

# create client
client = Groq(
    api_key=GROQ_API_KEY,
)

# prompts

# CLASS_PROMPT : classifier prompt
# NEWS_PROMPT : current events prompt
# COMPANY_PROMPT : company prompt
# THINK_PROMPT : thinking companion prompt

CLASS_PROMPT = """
Look at the user query.
Is it a question about a current event / political topic? If so, return 1.
Is it a question about a company or an investment opportunity? If so, return 2.
Is it a general question requiring a thinking companion? If so, return 3.
If it's impossible to fit any of the preceding descriptions, return 4.
Don't return anything else. Just one number. Nothing else. No explanations. You should return one single symbol. Double-check that.
"""

NEWS_PROMPT = """
You are a talented political commentator who's able to comment on a political event from multiple angles. 
You will be given a current political event/topic. 
Return a JSON object with the following data (look at the name of the parameter and its explanation): 
    "topic", // The name of the topic/event
    "timeline", // Date, like January 2020 -- March 2022
    "summary",  // This is a compressed UNBIASED summary of the commentaries from different political views
    "left_wing_commentary", // What is the left-wing view of the event?
    "right_wing_commentary", // What is the right-wing view of the event?
    "list_of_last_news". // Every piece of news is structured as follows: "news_story", "date", "source", "bias_level". "bias_level" is a float between 0.0 (unbiased) and 1.0 (extremely biased and one-sided).
Make sure that you respect the provided schema of the JSON object.
Don't return anything else. Just the JSON.
    """

COMPANY_PROMPT = """
You are a talented investor.
Provide fundamental analysis for a company.
Give the main information about its sector. Is it one of the top companies in its given field?
State its strong and weak points as an investment opportunity.
Estimate its ability to generate revenue.
Present some historical data about the stock price and the company's performance.
"""

THINK_PROMPT = """
You are a wise thinking companion.
You help people develop their thoughts by providing a balanced outlook to their problem.
Try to give advice on a given topic by giving a complementary view to what the user is saying.
Identify their 'weak points' where their argument is lacking and try to fill those missing spots.
"""

# classifier
def get_mode(user_input):
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": CLASS_PROMPT,
            },
            {
                "role": "user",
                "content": user_input,
            }
        ],
        model="mixtral-8x7b-32768",
    )

    model_output = chat_completion.choices[0].message.content

    # Make sure that the classifier outputs a single number
    regex = re.compile(r'(\d+).*', flags=re.DOTALL)
    result = regex.search(model_output).group(1)
    result = int(result) if result is not None else 4

    return result


def news_query(user_input):
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": NEWS_PROMPT,
            },
            {
                "role": "user",
                "content": user_input,
            }
        ],
        model="mixtral-8x7b-32768",
    )

    return chat_completion.choices[0].message.content

def company_query(user_input):
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": COMPANY_PROMPT,
            },
            {
                "role": "user",
                "content": user_input,
            }
        ],
        model="mixtral-8x7b-32768",
    )

    return chat_completion.choices[0].message.content

def think_query(user_input):
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": THINK_PROMPT,
            },
            {
                "role": "user",
                "content": user_input,
            }
        ],
        model="mixtral-8x7b-32768",
    )

    return chat_completion.choices[0].message.content
