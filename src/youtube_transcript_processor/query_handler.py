"""Query handler for processing user queries against YouTube transcript chunks."""

import logging
import argparse
from typing import List, Dict, Any, Optional
from chroma_storage import ChromaStorage
from sentence_transformers import SentenceTransformer
import torch
import json
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.FileHandler('query_handler.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class QueryHandler:
    """Handles query processing and response generation."""
    
    def __init__(
        self,
        chroma_storage: ChromaStorage,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        top_k: int = 5
    ):
        """Initialize the query handler.
        
        Args:
            chroma_storage: Initialized ChromaStorage instance
            embedding_model_name: Name of the sentence transformer model to use
            top_k: Number of top chunks to retrieve
        """
        self.chroma_storage = chroma_storage
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.top_k = top_k
        
    def _embed_query(self, query: str) -> List[float]:
        """Embed the query using the sentence transformer model."""
        with torch.no_grad():
            embedding = self.embedding_model.encode(query)
        return embedding.tolist()
        
    def _format_chunks_for_llm(self, chunks: List[Dict[str, Any]]) -> str:
        """Format retrieved chunks into a prompt for the LLM."""
        formatted_chunks = []
        for chunk in chunks:
            metadata = chunk.get("metadata", {})
            formatted_chunk = (
                f"Video: {metadata.get('title', 'Unknown')}\n"
                f"Channel: {metadata.get('channel_name', 'Unknown')}\n"
                f"Date: {metadata.get('upload_date', 'Unknown')}\n"
                f"Content: {chunk.get('document', '')}\n"
                f"---\n"
            )
            formatted_chunks.append(formatted_chunk)
            
        return "\n".join(formatted_chunks)
        
    def process_query(
        self,
        query: str,
        llm_client: Any,  # Type hint for your preferred LLM client
        system_prompt: Optional[str] = None,
        debug: bool = False
    ) -> str:
        """Process a user query and generate a response.
        
        Args:
            query: User's query string
            llm_client: Initialized LLM client instance
            system_prompt: Optional system prompt for the LLM
            debug: Whether to print retrieved chunks for debugging
            
        Returns:
            Generated response string
        """
        try:
            # Embed the query
            query_embedding = self._embed_query(query)
            
            # Search ChromaDB for relevant chunks
            results = self.chroma_storage.collection.query(
                query_embeddings=[query_embedding],
                n_results=self.top_k
            )
            
            # Format chunks for LLM
            formatted_chunks = self._format_chunks_for_llm([
                {
                    "document": doc,
                    "metadata": meta
                }
                for doc, meta in zip(results["documents"][0], results["metadatas"][0])
            ])
            
            # Debug output if requested
            if debug:
                print("\nRetrieved chunks:")
                print(formatted_chunks)
            
            # Prepare the prompt
            if system_prompt is None:
                system_prompt = (
                    "You are a friendly and knowledgeable financial analysis assistant. Your role is to provide "
                    "clear, concise, and focused answers based on the information available in the video transcripts. "
                    "Follow these guidelines:\n"
                    "1. Focus ONLY on the specific question asked\n"
                    "2. If the question is about specific stocks (like AAPL or TSLA), only include information about those stocks\n"
                    "3. Keep responses concise and to the point\n"
                    "4. Use a conversational tone\n"
                    "5. If you don't have enough information about the specific topic, simply say 'I don't have enough information about that specific topic in my current knowledge base. Would you like to ask about something else?'\n"
                    "6. Never mention 'chunks' or technical details about how you process information\n"
                    "7. Never say 'no specific analysis was found' or similar phrases\n"
                    "8. If you have partial information, focus on what you do know rather than what you don't"
                )
            
            # Generate response using LLM
            response = llm_client.generate(
                system_prompt=system_prompt,
                user_prompt=f"Question: {query}\n\nRelevant chunks:\n{formatted_chunks}"
            )
            
            # If no relevant information was found, provide a more conversational response
            if "cannot find relevant information" in response.lower() or "no specific analysis" in response.lower():
                return "I don't have enough information about that specific topic in my current knowledge base. Would you like to ask about something else?"
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return "I apologize, but I encountered an error while processing your question. Could you please try rephrasing it?"

def list_available_tags():
    """List all available tags in the ChromaDB collection."""
    try:
        # Initialize ChromaStorage
        storage = ChromaStorage()
        
        # Get all unique tags
        all_metadata = storage.collection.get()["metadatas"]
        tags = set()
        for metadata in all_metadata:
            if "tags" in metadata:
                tags.update(metadata["tags"])
        
        print("\nAvailable tags:")
        for tag in sorted(tags):
            print(f"- {tag}")
            
    except Exception as e:
        logger.error(f"Error listing tags: {e}")
        print(f"Error listing tags: {str(e)}")

def search_by_tags(tags: List[str], output_file: Optional[str] = None):
    """Search for videos by tags and optionally save results to a file."""
    try:
        # Initialize ChromaStorage
        storage = ChromaStorage()
        
        # Get all documents and metadata
        results = storage.collection.get()
        
        # Filter by tags
        matching_videos = []
        for doc, metadata in zip(results["documents"], results["metadatas"]):
            if "tags" in metadata and all(tag in metadata["tags"] for tag in tags):
                matching_videos.append({
                    "title": metadata.get("title", "Unknown"),
                    "channel": metadata.get("channel_name", "Unknown"),
                    "date": metadata.get("upload_date", "Unknown"),
                    "content": doc
                })
        
        # Print or save results
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(matching_videos, f, indent=2, ensure_ascii=False)
            print(f"\nResults saved to {output_file}")
        else:
            print(f"\nFound {len(matching_videos)} videos matching tags: {', '.join(tags)}")
            for video in matching_videos:
                print(f"\nTitle: {video['title']}")
                print(f"Channel: {video['channel']}")
                print(f"Date: {video['date']}")
                print("-" * 50)
            
    except Exception as e:
        logger.error(f"Error searching by tags: {e}")
        print(f"Error searching by tags: {str(e)}")

def main():
    parser = argparse.ArgumentParser(
        description="YouTube Transcript Query Handler",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all available tags
  python query_handler.py --list-tags

  # Search for videos with specific tags
  python query_handler.py --tags "technical analysis" "stock market" --output results.json

  # Search for videos with specific tags and display results
  python query_handler.py --tags "technical analysis" "stock market"

  # Ask a question about the videos
  python query_handler.py --question "What are the key technical indicators mentioned in recent videos?"

  # Ask a question with specific tags
  python query_handler.py --question "What are the key technical indicators?" --tags "technical analysis" "stock market"

  # Ask a question with debug output and custom number of chunks
  python query_handler.py --question "What are the key technical indicators?" --debug --top-k 10
        """
    )
    
    parser.add_argument(
        "--list-tags",
        action="store_true",
        help="List all available tags in the database"
    )
    
    parser.add_argument(
        "--tags",
        nargs="+",
        help="Search for videos with specific tags"
    )
    
    parser.add_argument(
        "--output",
        help="Save search results to a JSON file"
    )

    parser.add_argument(
        "--question",
        help="Ask a question about the videos"
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print retrieved chunks for debugging"
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of chunks to retrieve (default: 5)"
    )
    
    args = parser.parse_args()
    
    if args.list_tags:
        list_available_tags()
    elif args.question:
        try:
            # Initialize ChromaStorage
            storage = ChromaStorage()
            
            # Initialize QueryHandler with custom top_k
            query_handler = QueryHandler(storage, top_k=args.top_k)
            
            # Initialize LLM client
            from llm_client import LLMClient
            llm_client = LLMClient()
            
            # Process the question
            response = query_handler.process_query(
                query=args.question,
                llm_client=llm_client,
                debug=args.debug
            )
            
            print("\nQuestion:", args.question)
            print("\nAnswer:", response)
            
        except Exception as e:
            logger.error(f"Error processing question: {e}")
            print(f"Error processing question: {str(e)}")
    elif args.tags:
        search_by_tags(args.tags, args.output)
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 