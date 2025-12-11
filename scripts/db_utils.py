import os
import urllib.parse
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_postgres import PGVector  

# TODO: Remove debug
from dotenv import load_dotenv



def get_vector_store():
    
    # TODO: Remove debug
    load_dotenv()

    pg_user = urllib.parse.quote_plus(os.getenv("PG_USERNAME"))
    pg_pass = urllib.parse.quote_plus(os.getenv("PG_PASSWORD"))
    pg_host = os.getenv("PG_HOST")
    pg_port = os.getenv("PG_PORT", "5432")
    pg_db   = os.getenv("PG_DATABASE", "langchain")
    connection_string = f"postgresql+psycopg://{pg_user}:{pg_pass}@{pg_host}:{pg_port}/{pg_db}"


    vectorstore = PGVector(
        embeddings=VertexAIEmbeddings(model_name="text-embedding-004"), 
        collection_name="expense_tracking",
        connection=connection_string,
        use_jsonb=True                  # Stores metadata as a json column so we can filter
    )

    return vectorstore