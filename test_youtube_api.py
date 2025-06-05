from googleapiclient.discovery import build
from dotenv import load_dotenv
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_youtube_api():
    # Load environment variables
    load_dotenv()
    api_key = os.getenv('YOUTUBE_API_KEY')
    
    if not api_key:
        logger.error("No YouTube API key found in .env file")
        return
    
    try:
        # Build the YouTube API client
        youtube = build('youtube', 'v3', developerKey=api_key)
        logger.info("Successfully built YouTube API client")
        
        # Test a simple API call - get video details for a known video
        test_video_id = 'dQw4w9WgXcQ'  # A popular video that's likely to exist
        request = youtube.videos().list(
            part='snippet,statistics',
            id=test_video_id
        )
        response = request.execute()
        
        if response['items']:
            video = response['items'][0]
            logger.info("Successfully retrieved video data:")
            logger.info(f"Title: {video['snippet']['title']}")
            logger.info(f"Views: {video['statistics']['viewCount']}")
            logger.info("API key is working correctly!")
        else:
            logger.error("No video data returned")
            
    except Exception as e:
        logger.error(f"Error testing YouTube API: {str(e)}")

if __name__ == "__main__":
    test_youtube_api() 