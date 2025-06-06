import chromadb
import json
from datetime import datetime, time
import os

def setup_collection(client: chromadb.Client, collection_name: str) -> chromadb.Collection:
    """Creates or gets a ChromaDB collection."""
    collection = client.get_or_create_collection(name=collection_name)
    return collection

def process_and_add_to_chroma(collection: chromadb.Collection, json_file_path: str):
    """Processes a JSON file and adds its contents to the ChromaDB collection."""
    print(f"Processing {json_file_path}...")
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    meta_data = data['meta_data']
    chunks_data = data['chunks']

    documents = []
    embeddings = []
    metadatas = []
    ids = []

    for chunk in chunks_data:
        documents.append(chunk['chunk'])
        embeddings.append(chunk['embeddings'])
        ids.append(chunk['id'])
        
        # Parse published date to unix timestamp
        published_dt = datetime.fromisoformat(meta_data['published'])
        published_timestamp = int(published_dt.timestamp())

        metadatas.append({
            'video_id': meta_data['video_id'],
            'channel_id': meta_data['channel_id'],
            'published_at': published_timestamp,
            'channel_name': meta_data['channel_name'],
            'video_title': meta_data['title'],
            'chunk_index': chunk['chunk_index']
        })

    if documents:
        collection.add(
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
    print(f"Added {len(documents)} chunks from {json_file_path} to the '{collection.name}' collection.")

def main():
    """Main function to set up ChromaDB and demonstrate queries."""
    # Setup ChromaDB client
    db_path = os.path.join(os.getcwd(), "chroma_db")
    if not os.path.exists(db_path):
        os.makedirs(db_path)
    client = chromadb.PersistentClient(path=db_path)
    
    # Create or get collection
    collection_name = "youtube_transcripts"
    collection = setup_collection(client, collection_name)

    # Process the specified JSON file
    json_file = "outputs/youtube/2025-06-06/processed/__ZdlfL75Lg.json"
    if os.path.exists(json_file):
        process_and_add_to_chroma(collection, json_file)
    else:
        print(f"Error: {json_file} not found.")
        return

    print("\n--- Query Examples ---")

    # 1. Get all chunks for a specific channel_id
    channel_id_to_find = "UCj9dPFcre600TsmHxHC37Iw"
    print(f"\n1. Getting chunks for channel_id: {channel_id_to_find}")
    results = collection.get(where={"channel_id": channel_id_to_find})
    print(f"Found {len(results['ids'])} chunks.")
    if results['documents']:
        print("First chunk:", results['documents'][0][:100] + "...")


    # 2. Get all chunks for a specific date (e.g., 2025-06-05)
    search_date_str = "2025-06-05"
    print(f"\n2. Getting chunks for date: {search_date_str}")
    try:
        search_date = datetime.strptime(search_date_str, "%Y-%m-%d").date()
        start_of_day = datetime.combine(search_date, time.min)
        end_of_day = datetime.combine(search_date, time.max)
        start_timestamp = int(start_of_day.timestamp())
        end_timestamp = int(end_of_day.timestamp())

        date_results = collection.get(
            where={
                "$and": [
                    {"published_at": {"$gte": start_timestamp}},
                    {"published_at": {"$lte": end_timestamp}}
                ]
            }
        )
        print(f"Found {len(date_results['ids'])} chunks.")
        if date_results['documents']:
            print("First chunk:", date_results['documents'][0][:100] + "...")
    except ValueError:
        print("Invalid date format. Please use YYYY-MM-DD.")


    # 3. Combined query: channel and date
    print(f"\n3. Getting chunks for channel_id '{channel_id_to_find}' on date '{search_date_str}'")
    
    combined_results = collection.get(
        where={
            "$and": [
                {"channel_id": channel_id_to_find},
                {"published_at": {"$gte": start_timestamp}},
                {"published_at": {"$lte": end_timestamp}}
            ]
        }
    )
    print(f"Found {len(combined_results['ids'])} chunks.")


    # 4. Similarity search with metadata filter
    print("\n4. Similarity search for 'Tesla stock' within the specific channel")
    query_results = collection.query(
        query_texts=["Tesla stock"],
        n_results=2,
        where={"channel_id": channel_id_to_find}
    )

    print("Query results for 'Tesla stock':")
    for doc in query_results['documents'][0]:
        print("- " + doc.replace("\n", " ")[:150] + "...")


if __name__ == "__main__":
    main() 