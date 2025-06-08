import os
import json
import feedparser
import yaml
from datetime import datetime, timedelta
from youtube_transcript_api import YouTubeTranscriptApi
import logging
from pathlib import Path
from tqdm import tqdm
import concurrent.futures
import time
from collections import defaultdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class YouTubeTranscriptFetcher:
    def __init__(self, config_path):
        self.start_time = time.time()
        self.config_path = config_path
        self.config = self._load_config(config_path)
        self.output_dir = Path("outputs/youtube")
        current_date = datetime.now().strftime('%Y-%m-%d')
        date_dir = self.output_dir / current_date
        self.transcripts_dir = date_dir / 'transcripts'
        self.stats_dir = date_dir / 'stats'
        self._setup_dirs()
        self.inactive_channels = set()
        self.channel_names = {}  # Store channel names for YAML comments
        self.stats = {
            'total_videos_found': 0,
            'channel_videos': defaultdict(int),
            'channel_transcripts': defaultdict(int),
            'total_transcript_hours': 0.0,
            'channel_transcript_hours': defaultdict(float)
        }
        
    def _setup_dirs(self):
        """Create output directories if they don't exist."""
        for directory in [self.transcripts_dir, self.stats_dir]:
            directory.mkdir(parents=True, exist_ok=True)

    def _load_config(self, config_path):
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            raise

    def _save_config(self):
        """Save updated configuration back to YAML file with channel names as comments."""
        try:
            # Remove inactive channels from the config
            active_channels = [cid for cid in self.config['channels'] if cid not in self.inactive_channels]
            self.config['channels'] = active_channels
            
            # Create YAML content with comments
            yaml_content = "time_window:\n"
            yaml_content += f"  hours_to_look_back: {self.config['time_window']['hours_to_look_back']}\n\n"
            yaml_content += "channels:\n"
            
            for channel_id in active_channels:
                channel_name = self.channel_names.get(channel_id, "Unknown Channel")
                yaml_content += f"  - {channel_id}  # {channel_name}\n"
            
            # Save the updated config
            with open(self.config_path, 'w') as f:
                f.write(yaml_content)
            
            logger.info(f"Removed {len(self.inactive_channels)} inactive channels from config")
        except Exception as e:
            logger.error(f"Error saving updated config: {e}")

    def _get_recent_videos(self, channel_id):
        """Fetch recent videos from channel RSS feed."""
        feed_url = f"https://www.youtube.com/feeds/videos.xml?channel_id={channel_id}"
        try:
            feed = feedparser.parse(feed_url)
            channel_name = feed.feed.get('title', 'Unknown Channel')
            self.channel_names[channel_id] = channel_name

            if not feed.entries:
                logger.warning(f"No entries found for channel {channel_id} ({channel_name})")
                self.inactive_channels.add(channel_id)
                return []

            threshold = datetime.now() - timedelta(hours=self.config['time_window']['hours_to_look_back'])
            
            recent_videos = []
            for entry in feed.entries:
                published = datetime.strptime(entry.published, "%Y-%m-%dT%H:%M:%S%z")
                if published.replace(tzinfo=None) > threshold:
                    recent_videos.append({
                        'video_id': entry.yt_videoid,
                        'published': published.isoformat(),
                        'title': entry.title,
                        'channel_id': channel_id,
                        'channel_name': channel_name
                    })
            
            # Update stats
            self.stats['total_videos_found'] += len(recent_videos)
            self.stats['channel_videos'][channel_name] = len(recent_videos)
            
            return recent_videos
        except Exception as e:
            logger.error(f"Error fetching feed for channel {channel_id}: {e}")
            self.inactive_channels.add(channel_id)
            return []

    def _get_transcript_for_video(self, video):
        """Fetch transcript for a single video and return transcript data."""
        try:
            transcript_segments = YouTubeTranscriptApi.get_transcript(video['video_id'])
            # Join transcript segments into a single string
            transcript_text = ' '.join([segment['text'] for segment in transcript_segments])
            
            # Calculate total duration in hours
            total_duration = sum(segment['duration'] for segment in transcript_segments)
            duration_hours = total_duration / 3600  # Convert seconds to hours
            
            transcript_data = {
                'channel_id': video['channel_id'],
                'channel_name': video['channel_name'],
                'video_id': video['video_id'],
                'published': video['published'],
                'title': video['title'],
                'transcript': transcript_text,  # Now a single string
                'duration_hours': duration_hours
            }
            
            # Save individual transcript
            transcript_filename = self.transcripts_dir / f"{video['video_id']}.json"
            with open(transcript_filename, 'w', encoding='utf-8') as f:
                json.dump(transcript_data, f, ensure_ascii=False, indent=2)
            
            # Update stats
            self.stats['channel_transcripts'][video['channel_name']] += 1
            self.stats['total_transcript_hours'] += duration_hours
            self.stats['channel_transcript_hours'][video['channel_name']] += duration_hours
            
            logger.info(f"Successfully processed video: {video['title']} from {video['channel_name']}")
            return transcript_data
        except Exception:
            logger.warning(f"No transcript for video: {video['title']} ({video['video_id']})")
            return None

    def _process_channel(self, channel_id):
        """Process a single channel: get videos and fetch transcripts concurrently."""
        try:
            videos = self._get_recent_videos(channel_id)
            if not videos:
                return []
            
            channel_transcripts = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                future_to_video = {executor.submit(self._get_transcript_for_video, video): video for video in videos}
                for future in concurrent.futures.as_completed(future_to_video):
                    result = future.result()
                    if result:
                        channel_transcripts.append(result)
            return channel_transcripts
        except Exception as e:
            logger.error(f"Error processing channel {channel_id}: {e}")
            self.inactive_channels.add(channel_id)
            return []

    def calculate_statistics(self):
        """Calculate final statistics from collected data."""
        stats = {
            "total_channels_processed": len(self.config['channels']),
            "total_channels_with_videos": len(self.stats['channel_videos']),
            "total_videos_found": self.stats['total_videos_found'],
            "total_transcripts_retrieved": sum(self.stats['channel_transcripts'].values()),
            "total_transcript_hours": round(self.stats['total_transcript_hours'], 2),
            "pipeline_execution_time": f"{time.time() - self.start_time:.2f} seconds",
            "inactive_channels_removed": len(self.inactive_channels),
            "channels": {}
        }
        
        # Combine video and transcript counts for each channel
        for channel_name in set(self.stats['channel_videos'].keys()) | set(self.stats['channel_transcripts'].keys()):
            stats['channels'][channel_name] = {
                'videos_found': self.stats['channel_videos'].get(channel_name, 0),
                'transcripts_retrieved': self.stats['channel_transcripts'].get(channel_name, 0),
                'transcript_hours': round(self.stats['channel_transcript_hours'].get(channel_name, 0), 2)
            }
        
        return stats

    def process_channels(self):
        """Process all channels concurrently and save transcripts."""
        all_transcripts = []
        
        with tqdm(total=len(self.config['channels']), desc="Processing channels", unit="channel") as pbar:
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                # Submit all channel processing tasks
                future_to_channel = {
                    executor.submit(self._process_channel, channel_id): channel_id 
                    for channel_id in self.config['channels']
                }
                
                # Process results as they complete
                for future in concurrent.futures.as_completed(future_to_channel):
                    channel_id = future_to_channel[future]
                    try:
                        results = future.result()
                        if results:
                            all_transcripts.extend(results)
                    except Exception as e:
                        logger.error(f"Error processing channel {channel_id}: {e}")
                    finally:
                        pbar.update(1)
        
        if all_transcripts:
            statistics = self.calculate_statistics()
            stats_filename = self.stats_dir / 'statistics.json'
            with open(stats_filename, 'w', encoding='utf-8') as f:
                json.dump(statistics, f, ensure_ascii=False, indent=2)
            
            self._display_statistics(statistics)
            
            # Save updated config with inactive channels removed
            if self.inactive_channels:
                self._save_config()
        else:
            logger.info("No new transcripts were collected in the last 24 hours.")

    def _display_statistics(self, statistics):
        """Display statistics about the processed videos."""
        # Use a list to build the report and log it once to avoid tqdm interference
        report = ["\n" + "="*50, "=== Final Statistics ===", "="*50]
        report.append(f"Total channels processed: {statistics['total_channels_processed']}")
        report.append(f"Channels with recent videos: {statistics['total_channels_with_videos']}")
        report.append(f"Total recent videos found: {statistics['total_videos_found']}")
        report.append(f"Total transcripts retrieved: {statistics['total_transcripts_retrieved']}")
        report.append(f"Total transcript hours: {statistics['total_transcript_hours']}")
        report.append(f"Pipeline execution time: {statistics['pipeline_execution_time']}")
        report.append(f"Inactive channels removed: {statistics['inactive_channels_removed']}")
        report.append("\n--- Per-Channel Breakdown ---")
        
        for channel_name, stats in statistics['channels'].items():
            report.append(f"  - {channel_name}: {stats['transcripts_retrieved']} transcripts ({stats['transcript_hours']} hours) from {stats['videos_found']} videos")
            
        report.append("="*50 + "\n")
        logger.info("\n".join(report))

def main():
    config_path = "config/youtube/youtube_config.yaml"
    fetcher = YouTubeTranscriptFetcher(config_path)
    fetcher.process_channels()

if __name__ == "__main__":
    main() 