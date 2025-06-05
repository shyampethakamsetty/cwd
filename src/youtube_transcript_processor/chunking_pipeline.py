"""Semantic chunking pipeline for YouTube transcripts."""

import os
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from openai import AzureOpenAI
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG to see all logs
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.FileHandler('chunking_pipeline.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

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
    logger.debug("Azure OpenAI configuration: endpoint=%s, version=%s", 
                os.getenv("AZURE_OPENAI_ENDPOINT"),
                os.getenv("AZURE_OPENAI_API_VERSION"))
except Exception as e:
    logger.error("Failed to initialize Azure OpenAI client: %s", str(e), exc_info=True)
    raise

# Initialize sentence transformer for embeddings
logger.info("Initializing sentence transformer model")
try:
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    logger.info("Sentence transformer model initialized successfully")
    logger.debug("Model name: all-MiniLM-L6-v2")
except Exception as e:
    logger.error("Failed to initialize sentence transformer model: %s", str(e), exc_info=True)
    raise

def get_latest_date_folder(base_path):
    """Get the most recent date folder in the youtube outputs directory."""
    logger.info("Searching for latest date folder in: %s", base_path)
    youtube_path = Path(base_path) / "outputs" / "youtube"
    logger.debug("YouTube path: %s", youtube_path)
    
    try:
        date_folders = [d for d in youtube_path.iterdir() if d.is_dir()]
        logger.debug("Found date folders: %s", [str(d) for d in date_folders])
        
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
    print("\n=== Azure OpenAI Request ===")
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
        # Prepare request
        request_data = {
            "model": os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            "messages": [
                {"role": "system", "content": "You are a technical analysis expert that creates semantic chunks from trading transcripts. Always return a valid JSON array of strings. Each chunk should be 2-3 paragraphs long and maintain technical analysis context."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3,
            "max_tokens": 4000  # Increased to handle longer chunks
        }
        
        # Print request details
        print("\nRequest Details:")
        print(f"Model: {request_data['model']}")
        print(f"Temperature: {request_data['temperature']}")
        print(f"Max Tokens: {request_data['max_tokens']}")
        
        # Make request
        print("\n=== Azure OpenAI Response ===")
        response = client.chat.completions.create(**request_data)
        
        # Get content and clean it
        content = response.choices[0].message.content.strip()
        print(f"\nResponse length: {len(content)} characters")
        
        # Clean the content by removing markdown code block markers
        if content.startswith("```json"):
            content = content[7:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()
        
        # Try to parse JSON
        try:
            chunks = json.loads(content)
            if not isinstance(chunks, list):
                print(f"\nError: Response is not a list")
                return []
            
            print(f"\nSuccessfully parsed {len(chunks)} chunks")
            for i, chunk in enumerate(chunks):
                print(f"Chunk {i} length: {len(chunk)} characters")
                print(f"Preview: {chunk[:200]}...")
            return chunks
            
        except json.JSONDecodeError as e:
            print(f"\nJSON Parse Error: {str(e)}")
            print(f"First 100 chars of content: {content[:100]}...")
            return []
            
    except Exception as e:
        print(f"\nError in semantic chunking: {str(e)}")
        return []

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
    """Process a single transcript file."""
    print(f"\nProcessing: {transcript_path.name}")
    
    # Read transcript
    try:
        with open(transcript_path, 'r', encoding='utf-8') as f:
            transcript_data = json.load(f)
        print("✓ Successfully read transcript file")
    except Exception as e:
        print(f"✗ Error reading transcript file: {str(e)}")
        return
    
    # Extract transcript text and metadata
    transcript_text = transcript_data['transcript']
    metadata = transcript_data['meta_data']
    video_id = metadata['video_id']
    
    print(f"Video: {metadata['title']} ({video_id})")
    print(f"Channel: {metadata['channel_name']}")
    print(f"Upload Date: {metadata['upload_date']}")
    print(f"Transcript length: {len(transcript_text)} characters")
    
    # Create semantic chunks
    chunks = semantic_chunk_transcript(transcript_text)
    
    if not chunks:
        print("✗ Failed to create chunks")
        return
    
    # Create embeddings
    print("\nCreating embeddings...")
    embeddings = create_embeddings(chunks)
    
    if not embeddings:
        print("✗ Failed to create embeddings")
        return
    
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
            "video_id": metadata['video_id'],
            "title": metadata['title'],
            "upload_date": metadata['upload_date'],
            "upload_time": metadata['upload_time'],
            "channel_name": metadata['channel_name'],
            "subscribers": metadata['subscribers'],
            "views": metadata['views'],
            "likes": metadata['likes'],
            "comments": metadata['comments'],
            "total_chunks": len(chunks),
            "original_transcript_length": len(transcript_text)
        }
    }
    
    # Save to JSON
    output_file = processed_dir / f"{video_id}.json"
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2)
        print(f"✓ Saved {len(chunks)} chunks to {output_file}")
    except Exception as e:
        print(f"✗ Error saving chunks: {str(e)}")

def main():
    """Main entry point for the chunking pipeline."""
    print("\n=== Starting Chunking Pipeline ===")
    print(f"Python version: {sys.version}")
    print(f"Current working directory: {os.getcwd()}")
    
    try:
        # Load environment variables
        print("\n=== Environment Variables ===")
        load_dotenv()
        print(f"AZURE_OPENAI_ENDPOINT: {os.getenv('AZURE_OPENAI_ENDPOINT')}")
        print(f"AZURE_OPENAI_API_VERSION: {os.getenv('AZURE_OPENAI_API_VERSION')}")
        print(f"AZURE_OPENAI_DEPLOYMENT_NAME: {os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME')}")
        
        # Initialize Azure OpenAI client
        print("\n=== Initializing Azure OpenAI Client ===")
        client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
        )
        print("✓ Azure OpenAI client initialized")
        
        # Initialize sentence transformer
        print("\n=== Initializing Sentence Transformer ===")
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("✓ Sentence transformer model initialized")
        
        # Get latest date folder
        print("\n=== Finding Latest Date Folder ===")
        base_path = Path.cwd()
        latest_date_folder = get_latest_date_folder(base_path)
        print(f"Processing transcripts from: {latest_date_folder}")
        
        # Create processed directory
        processed_dir = latest_date_folder / "processed"
        processed_dir.mkdir(exist_ok=True)
        print(f"Created processed directory: {processed_dir}")
        
        # Process all transcripts
        transcripts_dir = latest_date_folder / "transcripts"
        transcript_files = list(transcripts_dir.glob("*.json"))
        print(f"\nFound {len(transcript_files)} transcript files to process")
        
        for transcript_file in transcript_files:
            process_transcript(transcript_file, processed_dir)
        
        print("\n=== Finished Processing All Transcripts ===")
        
    except Exception as e:
        print(f"\n=== Pipeline Failed ===")
        print(f"Error: {str(e)}")
        raise

if __name__ == "__main__":
    main() 