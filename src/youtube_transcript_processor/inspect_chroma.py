"""Inspect and display data inside ChromaDB."""

from pathlib import Path
from tabulate import tabulate
import argparse
from datetime import datetime
from chroma_storage import ChromaStorage

def inspect_chroma_data(channel_id: str = None, start_date: str = None, end_date: str = None):
    # Initialize ChromaStorage
    storage = ChromaStorage(
        persist_directory="chroma_db",
        collection_name="youtube_transcripts"
    )
    
    # Get documents based on filters
    if channel_id:
        docs = storage.get_channel_chunks(
            channel_id=channel_id,
            start_date=start_date,
            end_date=end_date,
            limit=1000
        )
    else:
        docs = storage.list_all_documents(limit=1000)
    
    if not docs:
        print("No documents found in the collection.")
        return
    
    # Prepare table data
    table_data = []
    for doc in docs:
        metadata = doc["metadata"]
        # Convert Unix timestamp to readable date if available
        published_at = metadata.get("published_at", "unknown")
        if published_at != "unknown":
            try:
                published_at = datetime.fromtimestamp(int(published_at)).strftime('%Y-%m-%d %H:%M:%S')
            except (ValueError, TypeError):
                pass
                
        table_data.append([
            doc["id"],
            metadata.get("video_id", "unknown"),
            metadata.get("channel_id", "unknown"),
            metadata.get("channel_name", "unknown"),
            published_at,
            metadata.get("video_title", "unknown"),
            len(doc["text"])
        ])
    
    # Print table
    headers = ["ID", "Video ID", "Channel ID", "Channel Name", "Published At", "Title", "Text Length"]
    print(tabulate(table_data, headers=headers, tablefmt="grid"))

def clear_chroma_data():
    # Initialize ChromaStorage
    storage = ChromaStorage(
        persist_directory="chroma_db",
        collection_name="youtube_transcripts"
    )
    
    # Clear all documents from the collection
    if storage.clear_collection():
        print("All documents have been cleared from the collection.")
    else:
        print("Failed to clear the collection. Check the logs for details.")

def main():
    parser = argparse.ArgumentParser(description='ChromaDB data inspection and management tool')
    parser.add_argument('--data', action='store_true', help='Display all data in the collection')
    parser.add_argument('--clear', action='store_true', help='Clear all data from the collection')
    parser.add_argument('--channel', type=str, help='Filter by channel ID')
    parser.add_argument('--start-date', type=str, help='Start date (Unix timestamp)')
    parser.add_argument('--end-date', type=str, help='End date (Unix timestamp)')
    
    args = parser.parse_args()
    
    if args.data or args.channel:
        inspect_chroma_data(
            channel_id=args.channel,
            start_date=args.start_date,
            end_date=args.end_date
        )
    elif args.clear:
        clear_chroma_data()
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 