# YouTube Transcript Processor

A tool for processing YouTube transcripts into semantic chunks with embeddings for trading analysis.

## Features

- Automatically finds and processes the latest YouTube transcripts
- Uses Azure OpenAI for semantic chunking
- Generates embeddings using sentence-transformers
- Saves chunks as individual JSON files with metadata

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd youtube-transcript-processor
```

2. Install dependencies:
```bash
pip install -e .
```

3. Create a `.env` file with your Azure OpenAI credentials:
```
AZURE_OPENAI_API_KEY=your_key
AZURE_OPENAI_API_VERSION=your_version
AZURE_OPENAI_ENDPOINT=your_endpoint
AZURE_OPENAI_DEPLOYMENT_NAME=your_deployment
```

## Usage

Run the chunking pipeline:
```bash
python -m youtube_transcript_processor.chunking_pipeline
```

The pipeline will:
1. Find the latest date folder in `outputs/youtube/`
2. Process all transcripts in that folder
3. Create semantic chunks using Azure OpenAI
4. Generate embeddings for each chunk
5. Save chunks as individual JSON files in the `processed` directory

## Output Format

Each chunk is saved as a separate JSON file named `video_id_chunknumber.json` with the following structure:

```json
{
    "id": "video_id_chunknumber",
    "chunk": "chunk text content",
    "embeddings": [embedding vector],
    "metadata": [
        {
            "video_id": "...",
            "title": "...",
            "upload_date": "...",
            "channel_name": "...",
            "chunk_index": chunk_number
        }
    ]
}
```

## Development

This project uses:
- Python 3.8+
- Hatch for build management
- Ruff for linting and formatting

To set up the development environment:
```bash
pip install hatch
hatch shell
```

## License

[Your chosen license] 