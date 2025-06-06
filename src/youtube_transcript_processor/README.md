# YouTube Transcript Fetcher

This script fetches transcripts from YouTube videos using RSS feeds and the youtube-transcript-api. It processes videos from specified channels that were published in the last 24 hours.

## Features

- Fetches videos from YouTube channels using RSS feeds (no API key required)
- Extracts transcripts using youtube-transcript-api
- Saves results in JSON format with date-based filenames
- Includes error handling and logging
- Configurable through YAML configuration file

## Requirements

- Python 3.7+
- Required packages (install using `pip install -r requirements.txt`):
  - feedparser
  - PyYAML
  - youtube-transcript-api

## Configuration

The script uses the configuration from `config/youtube/youtube_config.yaml`. The configuration includes:

- List of YouTube channel IDs to process
- Time window settings (how many hours to look back)
- Output directory settings
- Fetcher settings (request delays, retries, etc.)

## Usage

1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the script:
   ```bash
   python youtube_transcript_fetcher.py
   ```

## Output

The script creates a `transcripts` directory inside the configured output directory. Each day's transcripts are saved in a JSON file named with the current date (e.g., `2024-03-21.json`).

The JSON file contains an array of transcript records, each including:
- channel_id
- video_id
- published date
- video title
- transcript array

## Error Handling

The script includes comprehensive error handling:
- Logs errors for failed channel processing
- Skips videos with disabled or unavailable transcripts
- Creates output directories if they don't exist
- Handles network errors and timeouts

## Logging

The script logs its progress and any errors to the console. Log messages include:
- Channel processing status
- Video processing status
- Error messages
- Summary of collected transcripts 