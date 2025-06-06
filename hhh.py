"""
YouTubeFetcher - A tool for fetching YouTube videos from specified channels within a time window.

This tool can be used by agents to collect YouTube video information from financial and trading channels.
It processes videos posted after a specified time (default: Friday 00:00 EST) until the current time.
"""

import requests
from bs4 import BeautifulSoup
import json
from datetime import datetime, timedelta
import os
import concurrent.futures
import re
import time
from requests.exceptions import RequestException, SSLError
from urllib3.exceptions import MaxRetryError, TimeoutError
import pytz
from typing import List, Dict, Optional, Tuple
import logging
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound, VideoUnavailable
from tqdm import tqdm
from functools import lru_cache
import threading
from collections import defaultdict
from dotenv import load_dotenv
import yaml
from config.paths import PATHS
from pathlib import Path

# Configure logging
logs_dir = PATHS['YOUTUBE']['LOGS']
logs_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(logs_dir / 'youtube_fetcher.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Add a custom formatter to handle Unicode characters in console output
for handler in logger.handlers:
    if isinstance(handler, logging.StreamHandler):
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        # Set the stream's encoding to utf-8
        if hasattr(handler.stream, 'encoding'):
            handler.stream.reconfigure(encoding='utf-8')

# Disable tqdm's default logging to prevent interference with our logging
tqdm.pandas = lambda *args, **kwargs: None
tqdm.pandas()

def load_config() -> Dict:
    """Load configuration from YAML file"""
    config_path = PATHS['YOUTUBE']['CONFIG'] / 'youtube_config.yaml'
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Error loading config file: {str(e)}")
        raise

class RateLimiter:
    """Rate limiter for API calls"""
    def __init__(self, calls_per_second: float):
        self.calls_per_second = calls_per_second
        self.last_call_time = defaultdict(float)
        self.lock = threading.Lock()

    def wait(self, key: str):
        """Wait if necessary to respect rate limits"""
        with self.lock:
            current_time = time.time()
            time_since_last_call = current_time - self.last_call_time[key]
            if time_since_last_call < 1.0 / self.calls_per_second:
                time.sleep(1.0 / self.calls_per_second - time_since_last_call)
            self.last_call_time[key] = time.time()

class YouTubeFetcher:
    def __init__(self, 
                 channels: List[str] = None,
                 output_dir: str = None,
                 max_urls_per_channel: int = 50,
                 request_delay: float = 1.0,
                 max_retries: int = 3,
                 timeout: int = 30):
        """
        Initialize the YouTubeFetcher with configuration from YAML file.
        
        Args:
            channels (List[str], optional): List of YouTube channel names to fetch from. If None, uses config file.
            output_dir (str, optional): Directory to save output files. If None, uses config file.
            max_urls_per_channel (int, optional): Maximum number of videos to fetch per channel. If None, uses config file.
            request_delay (float, optional): Delay between requests in seconds. If None, uses config file.
            max_retries (int, optional): Maximum number of retries for failed requests. If None, uses config file.
            timeout (int, optional): Request timeout in seconds. If None, uses config file.
        """
        # Load configuration
        config = load_config()
        
        # Load environment variables
        load_dotenv()
        self.api_key = os.getenv('YOUTUBE_API_KEY')
        
        if not self.api_key:
            logger.warning("YOUTUBE_API_KEY not found in .env file. Some statistics will not be available.")
        else:
            logger.info("YouTube API key loaded successfully")
            self.youtube = build('youtube', 'v3', developerKey=self.api_key)
        
        # Store configuration
        self.channels = channels or config['channels']
        
        # Set output_dir from config if not provided
        if output_dir is None:
            self.output_dir = PATHS['YOUTUBE']['OUTPUTS']
        else:
            self.output_dir = Path(output_dir) if os.path.isabs(output_dir) else PATHS['YOUTUBE']['OUTPUTS'] / output_dir
            
        # Set other parameters from config if not provided
        self.max_urls_per_channel = max_urls_per_channel or config['fetcher_settings']['max_urls_per_channel']
        self.request_delay = request_delay or config['fetcher_settings']['request_delay']
        self.max_retries = max_retries or config['fetcher_settings']['max_retries']
        self.timeout = timeout or config['fetcher_settings']['timeout']
        
        # Timezone settings
        self.israel_tz = pytz.timezone('Asia/Jerusalem')
        self.est_tz = pytz.timezone('US/Eastern')
        
        # Output settings
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize rate limiters
        self.api_rate_limiter = RateLimiter(calls_per_second=1.0)  # 1 call per second for API
        self.web_rate_limiter = RateLimiter(calls_per_second=2.0)  # 2 calls per second for web scraping
        
        # Initialize cache directory
        self.cache_dir = PATHS['YOUTUBE']['CACHE']
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize quota tracking
        self.quota_used = 0
        self.quota_costs = {
            'videos.list': 1,
            'channels.list': 1,
            'search.list': 100
        }
        
        # Store time window configuration
        self.time_window_config = config['time_window']

    def get_weekly_time_window(self) -> Tuple[datetime, datetime]:
        """
        Get the time window for video collection based on configuration.
        
        Returns:
            Tuple[datetime, datetime]: Start and end times in UTC
        """
        config = self.time_window_config
        
        # Get current time in UTC
        current_time = datetime.now(pytz.UTC)
        
        # If specific start and end times are provided in config
        if 'start_time' in config and 'end_time' in config:
            start_time = datetime.strptime(config['start_time'], "%Y-%m-%d %H:%M:%S")
            end_time = datetime.strptime(config['end_time'], "%Y-%m-%d %H:%M:%S")
            start_time = pytz.UTC.localize(start_time)
            end_time = pytz.UTC.localize(end_time)
        else:
            # Use hours_to_look_back from config
            hours_to_look_back = config.get('hours_to_look_back', 24)
            end_time = current_time
            start_time = current_time - timedelta(hours=hours_to_look_back)
        
        # Log the time window
        logger.info(f"Collecting videos from {start_time.strftime('%Y-%m-%d %H:%M')} UTC to {end_time.strftime('%Y-%m-%d %H:%M')} UTC")
        return start_time, end_time
    
    def is_video_in_weekly_window(self, video_datetime_utc: datetime) -> bool:
        """
        Check if video falls within the weekly time window.
        
        Args:
            video_datetime_utc (datetime): Video upload datetime in UTC
            
        Returns:
            bool: True if video is within time window, False otherwise
        """
        start_time, end_time = self.get_weekly_time_window()
        return start_time <= video_datetime_utc <= end_time
    
    def update_quota(self, cost: int):
        """Update and log quota usage"""
        self.quota_used += cost
        logger.info(f"Current quota usage: {self.quota_used} units")

    def _get_video_statistics(self, video_id: str) -> Dict:
        """Get video statistics using YouTube Data API with caching and quota tracking"""
        cache_file = self.cache_dir / f"stats_{video_id}.json"
        
        # Check cache first
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Error reading cache for video {video_id}: {str(e)}")
        
        # Respect rate limits
        self.api_rate_limiter.wait('api')
        
        for attempt in range(self.max_retries):
            try:
                self.update_quota(self.quota_costs['videos.list'])
                video_response = self.youtube.videos().list(
                    part='statistics,snippet',
                    id=video_id
                ).execute()

                if not video_response['items']:
                    logger.warning(f"No data found for video {video_id}")
                    return {}

                video_data = video_response['items'][0]
                channel_id = video_data['snippet']['channelId']

                # Respect rate limits
                self.api_rate_limiter.wait('api')
                self.update_quota(self.quota_costs['channels.list'])
                
                channel_response = self.youtube.channels().list(
                    part='statistics',
                    id=channel_id
                ).execute()

                if not channel_response['items']:
                    logger.warning(f"No channel data found for video {video_id}")
                    return {}

                channel_data = channel_response['items'][0]

                stats = {
                    'views': int(video_data['statistics']['viewCount']),
                    'likes': int(video_data['statistics'].get('likeCount', 0)),
                    'subscribers': int(channel_data['statistics']['subscriberCount']),
                    'channel_id': channel_id,
                    'comment_count': int(video_data['statistics'].get('commentCount', 0))
                }
                
                # Cache the results
                try:
                    with open(cache_file, 'w') as f:
                        json.dump(stats, f)
                except Exception as e:
                    logger.warning(f"Error caching stats for video {video_id}: {str(e)}")
                
                return stats

            except HttpError as e:
                if e.resp.status == 403 and "quotaExceeded" in str(e):
                    logger.error("YouTube API quota exceeded")
                    return {}
                elif e.resp.status == 400 and "API key not valid" in str(e):
                    logger.error("Invalid YouTube API key. Please check your configuration.")
                    return {}
                else:
                    logger.error(f"YouTube API error for video {video_id}: {str(e)}")
                    if attempt < self.max_retries - 1:
                        time.sleep(self.request_delay * (attempt + 1))
                        continue
                    return {}
            except Exception as e:
                logger.error(f"Unexpected error for video {video_id}: {str(e)}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.request_delay * (attempt + 1))
                    continue
                return {}
        
        return {}

    @lru_cache(maxsize=1000)
    def _get_video_transcript(self, video_id: str) -> Optional[str]:
        """Get video transcript if available with caching"""
        cache_file = self.cache_dir / f"transcript_{video_id}.txt"
        
        # Check cache first
        if cache_file.exists():
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    return f.read()
            except Exception as e:
                logger.warning(f"Error reading transcript cache for video {video_id}: {str(e)}")
        
        try:
            transcript_data = YouTubeTranscriptApi.get_transcript(video_id)
            transcript_text = ' '.join([entry['text'] for entry in transcript_data])
            
            # Cache the transcript
            try:
                with open(cache_file, 'w', encoding='utf-8') as f:
                    f.write(transcript_text)
            except Exception as e:
                logger.warning(f"Error caching transcript for video {video_id}: {str(e)}")
            
            return transcript_text
        except (TranscriptsDisabled, NoTranscriptFound, VideoUnavailable) as e:
            logger.warning(f"Transcript not available for video {video_id}: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Error getting transcript for video {video_id}: {str(e)}")
            return None

    def get_video_details(self, video_id: str, session: requests.Session) -> Optional[Dict]:
        """
        Fetch details for a specific YouTube video.
        
        Args:
            video_id (str): YouTube video ID
            session (requests.Session): Active requests session
            
        Returns:
            Optional[Dict]: Video details or None if failed to fetch
        """
        cache_file = self.cache_dir / f"details_{video_id}.json"
        
        # Check cache first
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Error reading details cache for video {video_id}: {str(e)}")
        
        url = f"https://www.youtube.com/watch?v={video_id}"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        for attempt in range(self.max_retries):
            try:
                # Respect rate limits
                self.web_rate_limiter.wait('web')
                
                response = session.get(url, headers=headers, timeout=self.timeout)
                response.raise_for_status()
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Extract title
                title = self._extract_title(soup, response.text)
                
                # Extract length
                length, duration_seconds = self._extract_length(soup)
                
                # Extract upload date and time
                upload_date, upload_time = self._extract_upload_datetime(soup, response.text)
                
                video_info = {
                    "url": url,
                    "video_id": video_id,
                    "title": title,
                    "length": length,
                    "duration_seconds": duration_seconds,
                    "upload_date": upload_date,
                    "upload_time": upload_time
                }

                # Add statistics if API key is available
                if self.youtube:
                    try:
                        stats = self._get_video_statistics(video_id)
                        video_info.update(stats)
                    except Exception as e:
                        logger.warning(f"Failed to get video statistics for {video_id}: {str(e)}")
                
                # Cache the results
                try:
                    with open(cache_file, 'w') as f:
                        json.dump(video_info, f)
                except Exception as e:
                    logger.warning(f"Error caching details for video {video_id}: {str(e)}")
                
                return video_info
            except (RequestException, SSLError, MaxRetryError, TimeoutError) as e:
                logger.error(f"Error processing video {video_id} (Attempt {attempt + 1}/{self.max_retries}): {str(e)}")
                if attempt == self.max_retries - 1:
                    logger.error(f"Failed to process video {video_id} after {self.max_retries} attempts.")
                    return None
                time.sleep(self.request_delay * (attempt + 1))
        
        return None
    
    def _extract_title(self, soup: BeautifulSoup, response_text: str) -> str:
        """Extract video title using multiple methods"""
        title = "Unknown Title"
        title_methods = [
            lambda: soup.find('meta', property='og:title')['content'],
            lambda: soup.find('meta', {'name': 'title'})['content'],
            lambda: soup.find('title').text.strip(),
            lambda: re.search(r'"title":"([^"]+)"', response_text).group(1),
        ]
        for method in title_methods:
            try:
                title = method()
                if title and title != "YouTube":
                    break
            except:
                continue
        return title
    
    def _extract_length(self, soup: BeautifulSoup) -> Tuple[str, int]:
        """Extract video length and duration in seconds"""
        length = "00:00"
        duration_seconds = 0
        length_tag = soup.find('meta', itemprop='duration')
        if length_tag:
            length_text = length_tag['content']
            match = re.search(r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?', length_text)
            if match:
                hours = int(match.group(1) or 0)
                minutes = int(match.group(2) or 0)
                seconds = int(match.group(3) or 0)
                duration_seconds = hours * 3600 + minutes * 60 + seconds
                length = f"{hours:02d}:{minutes:02d}:{seconds:02d}" if hours else f"{minutes:02d}:{seconds:02d}"
        return length, duration_seconds
    
    def _extract_upload_datetime(self, soup: BeautifulSoup, response_text: str) -> Tuple[str, str]:
        """Extract upload date and time, converting from UTC to EST"""
        upload_date = "Unknown Date"
        upload_time = "Unknown Time"
        date_methods = [
            lambda: {
                'date': soup.find('meta', itemprop='uploadDate')['content'].split('T')[0],
                'time': soup.find('meta', itemprop='uploadDate')['content'].split('T')[1][:5]
            },
            lambda: {
                'date': re.search(r'"uploadDate":"(\d{4}-\d{2}-\d{2})T([^"]+)"', response_text).group(1),
                'time': re.search(r'"uploadDate":"(\d{4}-\d{2}-\d{2})T([^"]+)"', response_text).group(2)[:5]
            }
        ]
        for method in date_methods:
            try:
                date_info = method()
                # Convert UTC to EST
                utc_dt = datetime.strptime(f"{date_info['date']} {date_info['time']}", "%Y-%m-%d %H:%M")
                utc_dt = pytz.UTC.localize(utc_dt)
                est_dt = utc_dt.astimezone(self.est_tz)
                upload_date = est_dt.strftime("%Y-%m-%d")
                upload_time = est_dt.strftime("%H:%M")
                break
            except:
                continue
        return upload_date, upload_time
    
    def get_channel_video_ids(self, channel_name: str, session: requests.Session) -> List[str]:
        """
        Fetch video IDs from a YouTube channel's video page.
        
        Args:
            channel_name (str): YouTube channel name
            session (requests.Session): Active requests session
            
        Returns:
            List[str]: List of video IDs
        """
        url = f"https://www.youtube.com/@{channel_name}/videos"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        for attempt in range(self.max_retries):
            try:
                time.sleep(self.request_delay)
                response = session.get(url, headers=headers, timeout=self.timeout)
                
                # Check for 404 or invalid channel
                if response.status_code == 404:
                    logger.warning(f"Channel @{channel_name} not found (404)")
                    return []
                    
                response.raise_for_status()
                soup = BeautifulSoup(response.text, 'html.parser')
                
                script_tag = soup.find("script", string=re.compile("var ytInitialData"))
                if script_tag:
                    json_text = re.search(r'var ytInitialData = (.+);</script>', str(script_tag)).group(1)
                    data = json.loads(json_text)
                    
                    video_ids = []
                    tabs = data.get('contents', {}).get('twoColumnBrowseResultsRenderer', {}).get('tabs', [])
                    
                    # Find the tab with 'VIDEOS' title
                    videos_tab = next((tab for tab in tabs if tab.get('tabRenderer', {}).get('title') == 'Videos'), None)
                    
                    if videos_tab:
                        contents = videos_tab.get('tabRenderer', {}).get('content', {}).get('richGridRenderer', {}).get('contents', [])
                        for item in contents:
                            if 'richItemRenderer' in item:
                                video_renderer = item['richItemRenderer']['content'].get('videoRenderer')
                                if video_renderer:
                                    video_id = video_renderer.get('videoId')
                                    if video_id:
                                        video_ids.append(video_id)
                                        if len(video_ids) >= self.max_urls_per_channel:
                                            break
                    
                    return video_ids
                return []
            
            except (RequestException, json.JSONDecodeError, SSLError, MaxRetryError, TimeoutError) as e:
                print(f"Error fetching video IDs for {channel_name} (Attempt {attempt+1}/{self.max_retries}): {str(e)}")
                if attempt < self.max_retries - 1:
                    print(f"Retrying in {self.request_delay * (attempt + 1)} seconds...")
                    time.sleep(self.request_delay * (attempt + 1))
                else:
                    print(f"Failed to fetch video IDs for {channel_name} after {self.max_retries} attempts.")
                    return []
        
        return []
    
    def validate_channel(self, channel_name: str, session: requests.Session) -> bool:
        url = f"https://www.youtube.com/@{channel_name}"
        try:
            response = session.head(url, timeout=10)
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"Error validating channel @{channel_name}: {str(e)}")
            return False
    
    def process_channel(self, channel_name: str) -> Tuple[List[Dict], int, int, int]:
        """
        Process a single YouTube channel to fetch and save video information.
        
        Args:
            channel_name (str): YouTube channel name
            
        Returns:
            Tuple[List[Dict], int, int, int]: (videos, collected_count, discarded_count, saved_count)
        """
        logger.info(f"\n{'='*50}\nFetching videos for channel: {channel_name}\n{'='*50}")
        
        # Create directory structure for current date
        current_date = datetime.now().strftime('%Y-%m-%d')
        date_dir = self.output_dir / current_date
        transcripts_dir = date_dir / 'transcripts'
        stats_dir = date_dir / 'stats'
        
        # Create directories if they don't exist
        for directory in [date_dir, transcripts_dir, stats_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Get list of already processed video IDs
        processed_video_ids = set()
        if transcripts_dir.exists():
            for file in transcripts_dir.glob('*.json'):
                try:
                    with open(file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if 'meta_data' in data and 'video_id' in data['meta_data']:
                            processed_video_ids.add(data['meta_data']['video_id'])
                except Exception as e:
                    logger.warning(f"Error reading transcript file {file}: {str(e)}")
        
        with requests.Session() as session:
            video_ids = self.get_channel_video_ids(channel_name, session)
            
            videos = []
            videos_collected = 0
            videos_discarded = 0

            # Add progress bar for video processing with custom formatting
            with tqdm(total=len(video_ids), 
                     desc=f"Processing {channel_name}", 
                     unit="video",
                     bar_format='{l_bar}{bar:30}{r_bar}{bar:-30b}',
                     position=0,
                     leave=True) as pbar:
                for video_id in video_ids:
                    # Skip if already processed
                    if video_id in processed_video_ids:
                        logger.info(f"Skipping already processed video: {video_id}")
                        pbar.update(1)
                        continue
                    
                    videos_collected += 1
                    video_info = self.get_video_details(video_id, session)
                    
                    if video_info and video_info['upload_date'] != "Unknown Date":
                        try:
                            # Parse video datetime and convert to UTC
                            video_datetime = datetime.strptime(
                                f"{video_info['upload_date']} {video_info['upload_time']}", 
                                "%Y-%m-%d %H:%M"
                            )
                            video_datetime_utc = pytz.UTC.localize(video_datetime)
                            
                            if self.is_video_in_weekly_window(video_datetime_utc):
                                video_info['Channel_Name'] = channel_name
                                
                                # Try to get transcript
                                try:
                                    transcript = self._get_video_transcript(video_id)
                                    if transcript:
                                        video_info['transcript'] = transcript
                                        
                                        # Save transcript immediately
                                        transcript_data = {
                                            'meta_data': {
                                                'subscribers': video_info.get('subscribers', 'Unknown'),
                                                'views': video_info.get('views', 'Unknown'),
                                                'likes': video_info.get('likes', 'Unknown'),
                                                'comments': video_info.get('comment_count', 'Unknown'),
                                                'video_id': video_info['video_id'],
                                                'title': video_info['title'],
                                                'upload_date': video_info['upload_date'],
                                                'upload_time': video_info['upload_time'],
                                                'channel_name': video_info['Channel_Name']
                                            },
                                            'transcript': transcript
                                        }
                                        transcript_filename = transcripts_dir / f"{video_id}.json"
                                        with open(transcript_filename, 'w', encoding='utf-8') as f:
                                            json.dump(transcript_data, f, ensure_ascii=False, indent=2)
                                        
                                        logger.info(f"[+] Video saved: {video_info['title']}")
                                except Exception as e:
                                    logger.warning(f"Failed to get transcript for video {video_id}: {str(e)}")
                                
                                videos.append(video_info)
                                logger.info(f"[+] Video accepted: {video_info['title']}")
                            else:
                                videos_discarded += 1
                                logger.info(f"[-] Video rejected: Outside weekly time window - {video_info['title']}")
                                
                        except ValueError as e:
                            logger.error(f"Invalid date/time format for video: {video_info['title']} - Error: {str(e)}")
                    
                    pbar.update(1)

        logger.info(f"\nNumber of recent videos found for {channel_name}: {len(videos)}")
        return videos, videos_collected, videos_discarded, len(videos)

    def calculate_statistics(self, videos: List[Dict]) -> Dict:
        """Calculate detailed statistics from video data"""
        channels = {}
        for video in videos:
            channel_name = video['Channel_Name']
            if channel_name not in channels:
                channels[channel_name] = {
                    'subscribers': int(video.get('subscribers', 0)),
                    'views': 0,
                    'likes': 0,
                    'comments': 0,
                    'videos': 0,
                    'total_minutes': 0,
                    'total_seconds': 0
                }
            channels[channel_name]['views'] += int(video.get('views', 0))
            channels[channel_name]['likes'] += int(video.get('likes', 0)) if video.get('likes') != 'Unknown' else 0
            channels[channel_name]['comments'] += int(video.get('comment_count', 0)) if video.get('comment_count') != 'Unknown' else 0
            channels[channel_name]['videos'] += 1
            minutes, seconds = map(int, video.get('length', '0:0').split(':'))
            channels[channel_name]['total_minutes'] += minutes
            channels[channel_name]['total_seconds'] += seconds

        total_channels = len(channels)
        total_videos = sum(channel['videos'] for channel in channels.values())
        total_views = sum(channel['views'] for channel in channels.values())
        total_likes = sum(channel['likes'] for channel in channels.values())
        total_comments = sum(channel['comments'] for channel in channels.values())
        total_subscribers = sum(channel['subscribers'] for channel in channels.values())
        
        total_minutes = sum(channel['total_minutes'] for channel in channels.values())
        total_seconds = sum(channel['total_seconds'] for channel in channels.values())
        total_minutes += total_seconds // 60
        total_hours, total_minutes = divmod(total_minutes, 60)
        total_time = f"{total_hours:02d}:{total_minutes:02d}"
        
        view_ratio = (total_views / total_subscribers) * 100 if total_subscribers > 0 else 0
        likes_ratio = (total_likes / total_views) * 100 if total_views > 0 else 0
        
        return {
            "total_channels": total_channels,
            "total_videos": total_videos,
            "total_views": total_views,
            "total_likes": total_likes,
            "total_comments": total_comments,
            "total_subscribers": total_subscribers,
            "total_time (HH:MM)": total_time,
            "view_ratio": f"{view_ratio:.2f}%",
            "likes_ratio": f"{likes_ratio:.2f}%",
            "quota_used": self.quota_used,
            "channels": channels
        }

    def fetch_videos(self) -> str:
        """
        Main method to fetch videos from all configured channels.
        
        Returns:
            str: Path to the saved weekly URLs file
        """
        all_videos = []
        total_channels = len(self.channels)
        total_videos_collected = 0
        total_videos_discarded = 0
        total_videos_saved = 0
    
        logger.info(f"\n{'='*50}\nStarting video collection for {total_channels} channels\n{'='*50}")
    
        # Add progress bar for channel processing with custom formatting
        with tqdm(total=total_channels, 
                 desc="Processing channels", 
                 unit="channel",
                 bar_format='{l_bar}{bar:30}{r_bar}{bar:-30b}',
                 position=0,
                 leave=True) as pbar:
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(self.channels)) as executor:
                results = list(executor.map(self.process_channel, self.channels))
                for result in results:
                    videos, collected, discarded, saved = result
                    all_videos.extend(videos)
                    total_videos_collected += collected
                    total_videos_discarded += discarded
                    total_videos_saved += saved
                    pbar.update(1)
        
        # Create directory structure based on current date
        current_date = datetime.now().strftime('%Y-%m-%d')
        date_dir = self.output_dir / current_date
        stats_dir = date_dir / 'stats'
        
        # Create stats directory if it doesn't exist
        stats_dir.mkdir(parents=True, exist_ok=True)
        
        # Save statistics
        statistics = self.calculate_statistics(all_videos)
        # Update statistics to include channel names
        statistics['channels'] = list(statistics['channels'].keys())
        stats_filename = stats_dir / 'statistics.json'
        with open(stats_filename, 'w', encoding='utf-8') as f:
            json.dump(statistics, f, ensure_ascii=False, indent=2)
        
        # Create a timestamped log file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_filename = logs_dir / f"{timestamp}.log"
        
        # Reconfigure logging to use the new log file
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        file_handler = logging.FileHandler(log_filename, encoding='utf-8')
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)
        logger.addHandler(logging.StreamHandler())

        logger.info(f"All video information has been saved to '{date_dir}'")
        logger.info(f"Total number of videos across all channels: {len(all_videos)}")

        self._display_statistics(statistics)
        
        return str(date_dir)

    def _display_statistics(self, statistics: Dict):
        """Display statistics about the processed videos."""
        logger.info("\n" + "="*50)
        logger.info("=== Statistics ===")
        logger.info("="*50)
        logger.info(f"Total channels processed: {statistics['total_channels']}")
        logger.info(f"Total videos collected: {statistics['total_videos']}")
        logger.info(f"Total views: {statistics['total_views']}")
        logger.info(f"Total likes: {statistics['total_likes']}")
        logger.info(f"Total comments: {statistics['total_comments']}")
        logger.info(f"Total subscribers: {statistics['total_subscribers']}")
        logger.info(f"Total video duration: {statistics['total_time (HH:MM)']}")
        logger.info(f"View ratio: {statistics['view_ratio']}")
        logger.info(f"Likes ratio: {statistics['likes_ratio']}")
        logger.info(f"Quota used: {statistics['quota_used']} units")
        logger.info("="*50 + "\n")

def main():
    """Main function for YouTubeFetcher"""
    try:
        # Initialize fetcher with configuration from YAML
        fetcher = YouTubeFetcher()
        
        # Fetch videos and save results
        output_file = fetcher.fetch_videos()
        logger.info(f"Video collection completed successfully. Results saved to: {output_file}")
    except Exception as e:
        logger.error(f"Video collection failed with error: {str(e)}")

if __name__ == "__main__":
    main()