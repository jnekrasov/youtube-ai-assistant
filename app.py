from autogen import UserProxyAgent

# YouTube creator proxy
ycreator_proxy = UserProxyAgent(
    name="ycreator",
    code_execution_config=False,
    human_input_mode='TERMINATE',
    description='Youtube content creator',
    system_message='You are YouTube content creator',
    is_termination_msg=lambda x: x.get("content", "") and x.get("content", "").rstrip().rstrip('.').endswith("TERMINATE")
)

# Code executor agent
code_executor = UserProxyAgent(
    name='code_executor',
    human_input_mode='NEVER',
    code_execution_config={
        'work_dir': 'code-runtime-sandbox',
        'use_docker':False
    }
)

from autogen import AssistantAgent, config_list_from_json
from prompts import yresearch_manager_system_prompt, yresearcher_system_prompt

llm_config = {
    'timeout': 600,
    'cache_seed': 41,
    'config_list': config_list_from_json(env_or_file='OAI_CONFIG_LIST'),
    'temperature': 0
}

# YouTube research manager
yresearch_manager = AssistantAgent(
    name='yresearch_manager',
    system_message=yresearch_manager_system_prompt,
    llm_config=llm_config,
    description="Research manager, it takes creator's question and start guiding the process of the research to produce the fact-based answer."
)

# YouTube researcher
yresearcher = AssistantAgent(
    name='yresearcher',
    system_message=yresearcher_system_prompt,
    llm_config=llm_config,
    description="Researcher, it can research on any given topic."
)

from autogen import GroupChat, GroupChatManager
chat = GroupChat(
    [
        ycreator_proxy,
        code_executor,
        yresearch_manager,
        yresearcher
    ],
    messages=[],
    max_round=25
)

chat_manager = GroupChatManager(
    groupchat=chat,
    llm_config={'config_list':llm_config['config_list']}
)

# Toolset available:

from typing_extensions import Annotated
import os

# 1. google_search
import serpapi
serpapi_client = serpapi.Client(api_key=os.environ['SERPER_API_KEY'])

@code_executor.register_for_execution()
@yresearcher.register_for_llm(name="google_search", description="Performs a Google search for the given query")
def google_search(q:Annotated[str, "The topic or query to perform a Google search on"]):
    params = {
        'engine': 'google',
        'q': q,
        'num': 3
    }

    found = serpapi_client.search(params)
    return found["organic_results"]

# 2. scrape_page_content

import os
from langchain_openai import AzureChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate

llm = AzureChatOpenAI(
    api_key=os.environ['AZURE_OPENAI_API_KEY'],
    azure_endpoint=os.environ['AZURE_OPENAI_ENDPOINT'],
    api_version=os.environ['AZURE_OPENAI_API_VERSION'],
    azure_deployment=os.environ['AZURE_OPENAI_DEPLOYMENT_NAME'],
    model_version=os.environ['AZURE_OPENAI_MODEL_VERSION'])

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 8000,
    chunk_overlap = 500,
    separators=['\n', '\n\n']
)
summarization_prompt_template = PromptTemplate(
    template="""
    Generate detailed summary of the following text tailored to the provided reason. 
    Be structured and do not miss any relevant details.
    Reason: {reason}
    Text:
    ```
    {text}
    ```
    Summary:
    """, input_variables=['reason', 'text'])

def summarize_scraped_content(content:str, reason:str):
    docs = text_splitter.create_documents([content])
   
    summarization_chain = load_summarize_chain(
        llm=llm,
        chain_type='map_reduce',
        map_prompt=summarization_prompt_template,
        combine_prompt=summarization_prompt_template
    )
   
    summarized = summarization_chain.invoke({ 'input_documents' :docs, 'reason': reason })
    return summarized['output_text']


from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
@code_executor.register_for_execution()
@yresearcher.register_for_llm(name="scrape_page_content", description="Extracts and summarizes content from a specified webpage URL based on the scraping objective")
def scrape_page_content(
    url: Annotated[str, 'The URL of the page to scrape'],
    reason: Annotated[str, 'The reason for scraping the page']):
    with sync_playwright() as pl:
        browser = pl.chromium.launch()
        page = browser.new_page()
        page.goto(url=url)
        content = page.content()
        browser.close()
       
        soup = BeautifulSoup(content, 'html.parser')
        for script_or_css in soup(['script', 'style']):
            script_or_css.decompose()
       
        return summarize_scraped_content(soup.get_text(), reason)


# 3. fetch_gtrends_average_interest_over_time
# 4. fetch_gtrends_related_topics
# 5. fetch_gtrends_related_quiries
from typing import List
@code_executor.register_for_execution()
@yresearcher.register_for_llm(name="fetch_gtrends_average_interest_over_time", description="Retrieves the average 12-month trend of earch interest for up to five keywords using Google Trends")
def fetch_gtrends_average_interest_over_time(keywords: Annotated[List[str], "A list of keywords (up to 5) to track trends over the past 12 months"]):
    params = {
        "engine": "google_trends",
        "q": ', '.join(keywords[:5]),
        "data_type": "TIMESERIES",
        "date": "today 12-m"
    }
   
    interest_over_time = serpapi_client.search(params)['interest_over_time']    
    return interest_over_time['averages']

@code_executor.register_for_execution()
@yresearcher.register_for_llm(name="fetch_gtrends_related_topics", description="Retrieves related topics for a keyword, showing rising and top areas of interest over the past 12 months on Google Trends")
def fetch_gtrends_related_topics(keyword: Annotated[str, "The keyword to find related topics for"]):
    params = {
        "engine": "google_trends",
        "q": keyword,
        "data_type": "RELATED_TOPICS",
        "date": "today 12-m"
    }
   
    related_topics = serpapi_client.search(params)['related_topics']
   
    for category in ['rising', 'top']:
        if category in related_topics:
            for topic in related_topics[category]:
                topic.pop('link', None)
                topic.pop('serpapi_link', None)
                if 'topic' in topic:
                    topic['topic'].pop('value', None)
   
    return related_topics

@code_executor.register_for_execution()
@yresearcher.register_for_llm(name="fetch_gtrends_related_quiries", description="Retrieves related search queries for a keyword, categorizing them into rising and top queries over the past 12 months on Google Trends.")
def fetch_gtrends_related_quiries(keyword: Annotated[str, "The keyword to find related searches for"]):
    params = {
        "engine": "google_trends",
        "q": keyword,
        "data_type": "RELATED_QUERIES",
        "date": "today 12-m"
    }
   
    related_quiries = serpapi_client.search(params)['related_queries']
   
    for category in ['rising', 'top']:
        if category in related_quiries:
            for topic in related_quiries[category]:
                topic.pop('link', None)
                topic.pop('serpapi_link', None)
   
    return related_quiries

# 6. fetch_youtube_trends
# 7. fetch_youtube_channel_and_videos_stats
from googleapiclient.discovery import build
youtube = build('youtube', 'v3', developerKey=os.environ['GOOGLE_API_KEY'])

@code_executor.register_for_execution()
@yresearcher.register_for_llm(name="fetch_youtube_trends", description="Retrieves trending YouTube videos for specified keywords")
def fetch_youtube_trends(keywords: Annotated[List[str], "Keywords to search for trending YouTube videos"]):
    # Search for videos by keywords
    search_response = youtube.search().list(
        q=' | '.join(keywords),
        part="id",
        type="video",
        maxResults=50,
        order="viewCount"
    ).execute()

    video_ids = [item['id']['videoId'] for item in search_response.get('items', [])]

    # Retrieve video details
    videos_response = youtube.videos().list(
        part="snippet,statistics",
        id=",".join(video_ids),
    ).execute()

    videos = []
    for video in videos_response.get('items', []):
        videos.append({
            "title": video['snippet']['title'],
            "videoId": video['id'],
            "viewCount": video['statistics'].get('viewCount'),
            "likeCount": video['statistics'].get('likeCount'),
        })

    # Sort Videos by View Count (if further sorting is needed)
    videos_sorted = sorted(videos, key=lambda x: int(x['viewCount']), reverse=True)

    return videos_sorted

@code_executor.register_for_execution()
@yresearcher.register_for_llm(name="fetch_youtube_channel_and_videos_stats", description="Gathers statistics about a YouTube channel and its videos")
def fetch_youtube_channel_and_videos_stats(
    channel_identifier: Annotated[str, "YouTube channel ID"]):

    # Determine if the identifier is a channel ID or a username
    if channel_identifier.startswith('UC'):
        channel_info = youtube.channels().list(
            id=channel_identifier,
            part="snippet,contentDetails,statistics"
        ).execute()
    else:
        channel_info = youtube.channels().list(
            forUsername=channel_identifier,
            part="snippet,contentDetails,statistics"
        ).execute()

    if not channel_info.get('items'):
        return "Channel not found."

    channel = channel_info['items'][0]
    uploads_playlist_id = channel['contentDetails']['relatedPlaylists']['uploads']
    channel_stats = channel['statistics']

    channel_details = {
        "title": channel['snippet']['title'],
        "description": channel['snippet']['description'],
        "viewCount": channel_stats.get('viewCount'),
        "subscriberCount": channel_stats.get('subscriberCount'),
        "videoCount": channel_stats.get('videoCount'),
    }

    # Fetch videos from the channel's uploads playlist
    videos = []
    next_page_token = None
    while True:
        playlist_items_response = youtube.playlistItems().list(
            playlistId=uploads_playlist_id,
            part="snippet",
            maxResults=50,
            pageToken=next_page_token
        ).execute()

        video_ids = [item['snippet']['resourceId']['videoId'] for item in playlist_items_response.get('items', [])]

        # Fetch statistics for each video
        videos_response = youtube.videos().list(
            part="snippet,statistics",
            id=",".join(video_ids),
        ).execute()

        for video in videos_response.get('items', []):
            videos.append({
                "title": video['snippet']['title'],
                "videoId": video['id'],
                "viewCount": video['statistics'].get('viewCount'),
                "likeCount": video['statistics'].get('likeCount'),
                # Additional video fields as needed
            })

        next_page_token = playlist_items_response.get('nextPageToken')
        if not next_page_token:
            break

    channel_details['videos'] = videos[:20]

    return channel_details

print('I am YouTube AI assistant! How can I help you today?')
yquestion = input()
ycreator_proxy.initiate_chat(chat_manager, message=yquestion)



