import requests
from bs4 import BeautifulSoup
import re
import yaml
from datetime import datetime

def extract_channel_id(url):
    try:
        # Add https:// if not present
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
            
        # Handle different URL formats
        if '/channel/' in url:
            # Direct channel URL format
            channel_id = url.split('/channel/')[1].split('/')[0]
            return channel_id
        elif '/c/' in url or '/user/' in url:
            # Custom URL format
            response = requests.get(url)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                # Look for channel ID in meta tags
                meta_tags = soup.find_all('meta', property='og:url')
                for tag in meta_tags:
                    content = tag.get('content', '')
                    if '/channel/' in content:
                        return content.split('/channel/')[1].split('/')[0]
        elif 'youtube.com/@' in url:
            # Handle channel handles
            response = requests.get(url)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                meta_tags = soup.find_all('meta', property='og:url')
                for tag in meta_tags:
                    content = tag.get('content', '')
                    if '/channel/' in content:
                        return content.split('/channel/')[1].split('/')[0]
        
        return None
    except Exception as e:
        print(f"Error processing {url}: {str(e)}")
        return None

def load_channels_from_config():
    try:
        with open('config/youtube/youtube_config.yaml', 'r') as file:
            config = yaml.safe_load(file)
            return config.get('channels', [])
    except Exception as e:
        print(f"Error loading config file: {str(e)}")
        return []

def save_to_file(results, filename):
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("YouTube Channel IDs\n")
            f.write("=" * 50 + "\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 50 + "\n\n")
            
            for channel_name, data in results.items():
                f.write(f"Channel Name: {channel_name}\n")
                f.write(f"URL: {data['url']}\n")
                if data['channel_id']:
                    f.write(f"Channel ID: {data['channel_id']}\n")
                else:
                    f.write("Channel ID: Not found\n")
                f.write("-" * 50 + "\n")
        
        print(f"\nResults have been saved to: {filename}")
    except Exception as e:
        print(f"Error saving to file: {str(e)}")

def main():
    # Load channels from config file
    channel_names = load_channels_from_config()
    
    print("\nProcessing YouTube channels...")
    print("-" * 50)
    
    results = {}
    for channel_name in channel_names:
        # Construct YouTube URL
        url = f"https://www.youtube.com/@{channel_name}"
        channel_id = extract_channel_id(url)
        
        results[channel_name] = {
            'url': url,
            'channel_id': channel_id
        }
        
        if channel_id:
            print(f"Channel Name: {channel_name}")
            print(f"URL: {url}")
            print(f"Channel ID: {channel_id}")
            print("-" * 50)
        else:
            print(f"Could not extract channel ID for: {channel_name}")
            print("-" * 50)
    
    # Save results to file
    output_filename = f"youtube_channel_ids_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    save_to_file(results, output_filename)

if __name__ == "__main__":
    main() 