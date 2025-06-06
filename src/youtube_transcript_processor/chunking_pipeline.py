"""Semantic chunking pipeline for YouTube transcripts."""

import os
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from openai import AzureOpenAI
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.INFO,  # Changed to INFO to reduce noise
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('chunking_pipeline.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Add tqdm to your requirements.txt: pip install tqdm
MAX_WORKERS = 10  # To avoid overwhelming the API endpoint. Adjust as needed.

# Log Python version and environment
logger.info("Python version: %s", sys.version)
logger.info("Current working directory: %s", os.getcwd())

# Load environment variables
logger.info("Loading environment variables")
load_dotenv()
logger.debug("Environment variables loaded")

# Initialize Azure OpenAI client
logger.info("Initializing Azure OpenAI client")
try:
    client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
    )
    logger.info("Azure OpenAI client initialized successfully")
except Exception as e:
    logger.error("Failed to initialize Azure OpenAI client: %s", str(e), exc_info=True)
    raise

# Initialize sentence transformer for embeddings
logger.info("Initializing sentence transformer model")
try:
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    logger.info("Sentence transformer model initialized successfully")
except Exception as e:
    logger.error("Failed to initialize sentence transformer model: %s", str(e), exc_info=True)
    raise

def get_latest_date_folder(base_path):
    """Get the most recent date folder in the youtube outputs directory."""
    logger.info("Searching for latest date folder in: %s", base_path)
    youtube_path = Path(base_path) / "outputs" / "youtube"
    
    try:
        date_folders = [d for d in youtube_path.iterdir() if d.is_dir()]
        
        if not date_folders:
            logger.error("No date folders found in %s", youtube_path)
            raise ValueError(f"No date folders found in {youtube_path}")
            
        latest_folder = max(date_folders, key=lambda x: datetime.strptime(x.name, "%Y-%m-%d"))
        logger.info("Latest date folder found: %s", latest_folder)
        return latest_folder
    except Exception as e:
        logger.error("Error finding latest date folder: %s", str(e), exc_info=True)
        raise

def semantic_chunk_transcript(transcript_text):
    """Use Azure OpenAI to create semantic chunks from transcript."""
    logger.info("Making request to Azure OpenAI for semantic chunking")
    prompt = f"""Please analyze this trading transcript and break it into semantic chunks. 
    Each chunk should be self-contained and focus on a specific aspect of the technical analysis.
    
    Guidelines for chunking:
    1. Each chunk should be 2-3 paragraphs long
    2. Preserve technical analysis relationships (e.g., keep related indicators together)
    3. Group related concepts (e.g., all moving average analysis in one chunk)
    4. Maintain the logical flow of the analysis
    5. Include relevant numerical data and technical terms
    
    Return the chunks as a JSON array of strings. Each chunk should be meaningful on its own.
    
    Transcript:
    {transcript_text}
    
    Return format:
    ["chunk1", "chunk2", ...]"""

    try:
        request_data = {
            "model": os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            "messages": [
                {"role": "system", "content": "You are a technical analysis expert that creates semantic chunks from trading transcripts. Always return a valid JSON array of strings. Each chunk should be 2-3 paragraphs long and maintain technical analysis context."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3,
            "max_tokens": 4000
        }
        
        response = client.chat.completions.create(**request_data)
        content = response.choices[0].message.content.strip()
        
        prompt_tokens = response.usage.prompt_tokens
        completion_tokens = response.usage.completion_tokens
        
        # Clean the content by removing markdown code block markers
        if content.startswith("```json"):
            content = content[7:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()
        
        try:
            chunks = json.loads(content)
            if not isinstance(chunks, list):
                logger.error("Response is not a list, but %s", type(chunks))
                return [], 0, 0
            
            logger.info("Successfully parsed %d chunks", len(chunks))
            return chunks, prompt_tokens, completion_tokens
            
        except json.JSONDecodeError as e:
            logger.error("JSON Parse Error: %s. Content starts with: %s...", str(e), content[:100])
            return [], 0, 0
            
    except Exception as e:
        logger.error("Error in semantic chunking: %s", str(e), exc_info=True)
        return [], 0, 0

def create_embeddings(chunks):
    """Create embeddings for the chunks using sentence-transformers."""
    logger.info("Creating embeddings for %d chunks", len(chunks))
    try:
        embeddings = embedding_model.encode(chunks).tolist()
        logger.info("Successfully created embeddings (dimension: %d)", len(embeddings[0]))
        return embeddings
    except Exception as e:
        logger.error("Error creating embeddings: %s", str(e), exc_info=True)
        return []

def process_transcript(transcript_path, processed_dir):
    """Process a single transcript file: chunk, embed, and save."""
    logger.info("Processing: %s", transcript_path.name)
    
    try:
        with open(transcript_path, 'r', encoding='utf-8') as f:
            transcript_data = json.load(f)
        logger.info("[OK] Read transcript file %s", transcript_path.name)
    except Exception as e:
        logger.error("[ERROR] Failed to read transcript file %s: %s", transcript_path.name, str(e))
        return None
    
    # Extract transcript text and metadata - note the structure change
    transcript_text = transcript_data['transcript']
    video_id = transcript_data['video_id']
    title = transcript_data['title']
    channel_name = transcript_data['channel_name']
    published = transcript_data['published']
    
    logger.info("Video: %s (%s)", title, video_id)
    logger.info("Channel: %s", channel_name)
    logger.info("Published: %s", published)
    logger.info("Transcript length: %d characters", len(transcript_text))
    
    # Create semantic chunks
    chunks, prompt_tokens, completion_tokens = semantic_chunk_transcript(transcript_text)
    
    if not chunks:
        logger.warning("[SKIP] Failed to create chunks for %s", video_id)
        return {"video_id": video_id, "status": "chunking_failed"}
    
    # Create embeddings
    logger.info("Creating embeddings for %d chunks...", len(chunks))
    embeddings = create_embeddings(chunks)
    
    if not embeddings:
        logger.warning("[SKIP] Failed to create embeddings for %s", video_id)
        return {"video_id": video_id, "status": "embedding_failed"}
    
    # Prepare all chunks data
    chunks_data = []
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        chunk_data = {
            "id": f"{video_id}_{i}",
            "chunk": chunk,
            "embeddings": embedding,
            "chunk_index": i
        }
        chunks_data.append(chunk_data)
    
    # Prepare final output with all metadata
    output_data = {
        "chunks": chunks_data,
        "meta_data": {
            "video_id": video_id,
            "title": title,
            "published": published,
            "channel_name": channel_name,
            "channel_id": transcript_data.get('channel_id', ''),
            "total_chunks": len(chunks),
            "original_transcript_length": len(transcript_text)
        }
    }
    
    # Save to JSON
    output_file = processed_dir / f"{video_id}.json"
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2)
        logger.info("[OK] Saved %d chunks to %s", len(chunks), output_file)
    except Exception as e:
        logger.error("[ERROR] Failed to save chunks for %s: %s", video_id, str(e))
        return {"video_id": video_id, "status": "save_failed"}

    return {
        "video_id": video_id,
        "status": "success",
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "chunks_created": len(chunks)
    }

def main():
    """Main entry point for the chunking pipeline."""
    logger.info("=== Starting Chunking Pipeline ===")
    start_time = time.time()
    
    try:
        # Get latest date folder
        base_path = Path.cwd()
        latest_date_folder = get_latest_date_folder(base_path)
        logger.info("Processing transcripts from: %s", latest_date_folder)
        
        # Create processed directory
        processed_dir = latest_date_folder / "processed"
        processed_dir.mkdir(exist_ok=True)
        logger.info("Created processed directory: %s", processed_dir)
        
        # Process all transcripts
        transcripts_dir = latest_date_folder / "transcripts"
        transcript_files = list(transcripts_dir.glob("*.json"))
        logger.info("Found %d transcript files to process", len(transcript_files))
        
        if not transcript_files:
            logger.info("No transcripts to process. Exiting.")
            return

        all_stats = []
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(process_transcript, f, processed_dir): f for f in transcript_files}
            
            # Use tqdm for a progress bar
            with tqdm(total=len(transcript_files), desc="Processing Transcripts", unit="file") as pbar:
                for future in as_completed(futures):
                    result = future.result()
                    if result:
                        all_stats.append(result)
                    pbar.update(1)

        logger.info("=== Pipeline Execution Summary ===")
        
        successful_procs = [s for s in all_stats if s.get('status') == 'success']
        failed_procs = [s for s in all_stats if s.get('status') != 'success']
        
        total_api_calls = len(successful_procs)
        total_prompt_tokens = sum(s.get('prompt_tokens', 0) for s in successful_procs)
        total_completion_tokens = sum(s.get('completion_tokens', 0) for s in successful_procs)
        total_tokens = total_prompt_tokens + total_completion_tokens
        
        end_time = time.time()
        pipeline_execution_time = round(end_time - start_time, 2)
        
        final_stats = {
            "pipeline_execution_time_seconds": pipeline_execution_time,
            "total_transcripts_found": len(transcript_files),
            "total_transcripts_processed_successfully": len(successful_procs),
            "total_transcripts_failed": len(failed_procs),
            "api_calls_made": total_api_calls,
            "total_prompt_tokens": total_prompt_tokens,
            "total_completion_tokens": total_completion_tokens,
            "total_tokens_processed": total_tokens,
            "individual_transcript_stats": all_stats
        }
        
        stats_dir = latest_date_folder / "stats"
        stats_dir.mkdir(exist_ok=True)
        stats_file = stats_dir / "stats_chunking.json"
        
        try:
            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump(final_stats, f, indent=4)
            logger.info("Pipeline stats saved to %s", stats_file)
        except Exception as e:
            logger.error("Failed to save stats file: %s", str(e), exc_info=True)

        logger.info("=== Finished Processing All Transcripts ===")
        logger.info("Execution time: %.2f seconds", pipeline_execution_time)
        logger.info("Successfully processed %d/%d transcripts", len(successful_procs), len(transcript_files))
        
    except Exception as e:
        logger.critical("=== Pipeline Failed ===", exc_info=True)
        raise

if __name__ == "__main__":
    main() 