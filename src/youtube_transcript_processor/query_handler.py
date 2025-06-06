"""Query handler for YouTube transcript chunks in ChromaDB."""

import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from chroma_storage import ChromaStorage
from tabulate import tabulate
from llm_client import LLMClient

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
    """Handles semantic search queries with metadata filtering."""
    
    def __init__(
        self,
        persist_directory: str = "chroma_db",
        collection_name: str = "youtube_transcripts"
    ):
        """Initialize the query handler.
        
        Args:
            persist_directory: Directory where ChromaDB data is persisted
            collection_name: Name of the ChromaDB collection
        """
        self.storage = ChromaStorage(
            persist_directory=persist_directory,
            collection_name=collection_name
        )
        self.llm_client = LLMClient()
        
    def _format_chunks_for_llm(self, chunks: List[Dict[str, Any]]) -> str:
        """Format retrieved chunks into a prompt for the LLM."""
        formatted_chunks = []
        for chunk in chunks:
            metadata = chunk["metadata"]
            formatted_chunk = (
                f"Source: {metadata['video_title']} (by {metadata['channel_name']})\n"
                f"Published: {metadata['published_at']}\n"
                f"Content: {chunk['text']}\n"
                f"---\n"
            )
            formatted_chunks.append(formatted_chunk)
        return "\n".join(formatted_chunks)

    def semantic_search(
        self,
        query: str,
        channel_id: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: int = 5,
        debug: bool = False,
        generate_response: bool = True
    ) -> Dict[str, Any]:
        """Perform semantic search with metadata filtering.
        
        Args:
            query: The search query text
            channel_id: Optional channel ID to filter by
            start_date: Optional start date (Unix timestamp)
            end_date: Optional end date (Unix timestamp)
            limit: Maximum number of results to return
            debug: Whether to print detailed debug information
            generate_response: Whether to generate an LLM response
            
        Returns:
            Dictionary containing search results and optional LLM response
        """
        try:
            # Build where clause for metadata filtering
            where = {}
            if channel_id:
                where["channel_id"] = channel_id
                
            if start_date or end_date:
                where["published_at"] = {}
                if start_date:
                    where["published_at"]["$gte"] = start_date
                if end_date:
                    where["published_at"]["$lte"] = end_date
            
            if debug:
                print("\n=== Search Parameters ===")
                print(f"Query: {query}")
                print(f"Filters: {where if where else 'None'}")
                print(f"Limit: {limit}")
            
            # Perform the query
            results = self.storage.collection.query(
                query_texts=[query],
                n_results=limit,
                where=where if where else None
            )
            
            # Format results
            formatted_results = []
            for i in range(len(results["ids"][0])):
                metadata = results["metadatas"][0][i]
                # Convert Unix timestamp to readable date
                published_at = metadata.get("published_at", "unknown")
                if published_at != "unknown":
                    try:
                        published_at = datetime.fromtimestamp(int(published_at)).strftime('%Y-%m-%d %H:%M:%S')
                    except (ValueError, TypeError):
                        pass
                
                formatted_results.append({
                    "id": results["ids"][0][i],
                    "text": results["documents"][0][i],
                    "metadata": {
                        "video_id": metadata.get("video_id", "unknown"),
                        "channel_id": metadata.get("channel_id", "unknown"),
                        "channel_name": metadata.get("channel_name", "unknown"),
                        "video_title": metadata.get("video_title", "unknown"),
                        "published_at": published_at,
                        "chunk_index": metadata.get("chunk_index", -1)
                    },
                    "similarity": results["distances"][0][i] if "distances" in results else None
                })
            
            if debug:
                print("\n=== Retrieved Chunks ===")
                table_data = []
                for result in formatted_results:
                    table_data.append([
                        result["id"],
                        result["metadata"]["video_title"],
                        result["metadata"]["channel_name"],
                        result["metadata"]["published_at"],
                        result["metadata"]["chunk_index"],
                        f"{result['similarity']:.4f}" if result['similarity'] is not None else "N/A",
                        len(result["text"])
                    ])
                
                print(tabulate(
                    table_data,
                    headers=["ID", "Video Title", "Channel", "Published", "Chunk Index", "Similarity", "Text Length"],
                    tablefmt="grid"
                ))
                
                print("\n=== Full Text of Retrieved Chunks ===")
                for i, result in enumerate(formatted_results, 1):
                    print(f"\nChunk {i}:")
                    print(f"ID: {result['id']}")
                    print(f"Video: {result['metadata']['video_title']}")
                    print(f"Channel: {result['metadata']['channel_name']}")
                    print(f"Published: {result['metadata']['published_at']}")
                    print(f"Chunk Index: {result['metadata']['chunk_index']}")
                    print(f"Similarity: {result['similarity']:.4f}" if result['similarity'] is not None else "N/A")
                    print(f"Text:\n{result['text']}\n")
                    print("-" * 80)
            
            response = None
            if generate_response and formatted_results:
                # Format chunks for LLM
                formatted_chunks = self._format_chunks_for_llm(formatted_results)
                
                # Generate response using LLM
                system_prompt = (
                    "You are a knowledgeable financial analysis assistant. Your role is to provide "
                    "clear, concise, and focused answers based on the information available in the video transcripts. "
                    "Follow these guidelines:\n"
                    "1. Focus ONLY on the specific question asked\n"
                    "2. Keep responses concise and to the point\n"
                    "3. Use a conversational tone\n"
                    "4. If you don't have enough information, say so\n"
                    "5. Never mention 'chunks' or technical details about how you process information\n"
                    "6. If you have partial information, focus on what you do know rather than what you don't"
                )
                
                response = self.llm_client.generate(
                    system_prompt=system_prompt,
                    user_prompt=f"Question: {query}\n\nRelevant information:\n{formatted_chunks}"
                )
            
            return {
                "results": formatted_results,
                "response": response
            }
            
        except Exception as e:
            logger.error(f"Error performing semantic search: {e}")
            return {"results": [], "response": None}
            
    def search_by_channel(
        self,
        channel_id: str,
        query: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: int = 5,
        debug: bool = False
    ) -> List[Dict[str, Any]]:
        """Search for content from a specific channel.
        
        Args:
            channel_id: Channel ID to search in
            query: Optional semantic search query
            start_date: Optional start date (Unix timestamp)
            end_date: Optional end date (Unix timestamp)
            limit: Maximum number of results to return
            debug: Whether to print detailed debug information
            
        Returns:
            List of matching documents
        """
        return self.semantic_search(
            query=query if query else "",
            channel_id=channel_id,
            start_date=start_date,
            end_date=end_date,
            limit=limit,
            debug=debug
        )
        
    def search_by_date_range(
        self,
        start_date: str,
        end_date: str,
        query: Optional[str] = None,
        channel_id: Optional[str] = None,
        limit: int = 5,
        debug: bool = False
    ) -> List[Dict[str, Any]]:
        """Search for content within a date range.
        
        Args:
            start_date: Start date (Unix timestamp)
            end_date: End date (Unix timestamp)
            query: Optional semantic search query
            channel_id: Optional channel ID to filter by
            limit: Maximum number of results to return
            debug: Whether to print detailed debug information
            
        Returns:
            List of matching documents
        """
        return self.semantic_search(
            query=query if query else "",
            channel_id=channel_id,
            start_date=start_date,
            end_date=end_date,
            limit=limit,
            debug=debug
        )

def main():
    """Example usage of the QueryHandler."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Query YouTube transcript chunks')
    parser.add_argument('--query', type=str, help='Search query')
    parser.add_argument('--channel', type=str, help='Channel ID to filter by')
    parser.add_argument('--start-date', type=str, help='Start date (Unix timestamp)')
    parser.add_argument('--end-date', type=str, help='End date (Unix timestamp)')
    parser.add_argument('--limit', type=int, default=5, help='Maximum number of results')
    parser.add_argument('--debug', action='store_true', help='Print detailed debug information')
    parser.add_argument('--no-response', action='store_true', help='Skip generating LLM response')
    
    args = parser.parse_args()
    
    handler = QueryHandler()
    
    if args.query:
        result = handler.semantic_search(
            query=args.query,
            channel_id=args.channel,
            start_date=args.start_date,
            end_date=args.end_date,
            limit=args.limit,
            debug=args.debug,
            generate_response=not args.no_response
        )
        
        if not args.debug:  # Only print summary if not in debug mode
            print(f"\nFound {len(result['results'])} results:")
            for i, res in enumerate(result['results'], 1):
                print(f"\nResult {i}:")
                print(f"Video: {res['metadata']['video_title']}")
                print(f"Channel: {res['metadata']['channel_name']} (ID: {res['metadata']['channel_id']})")
                print(f"Published: {res['metadata']['published_at']}")
                print(f"Text: {res['text'][:200]}...")
                if res['similarity'] is not None:
                    print(f"Similarity: {res['similarity']:.4f}")
        
        if result['response']:
            print("\n=== Generated Response ===")
            print(result['response'])
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 