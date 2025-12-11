import pandas as pd
import boto3
from google.cloud import bigquery
import os
import io
from langchain_core.documents import Document
import hashlib
import db_utils

# TODO: Remove debug
def get_local_data(filename):
    return pd.read_csv(filename)

def get_aws_data(bucket_name, file_key):
    s3_client = boto3.client('s3')
    response = s3_client.get_object(Bucket=bucket_name, Key=file_key)
    data = response['Body'].read().decode('utf-8')
    df = pd.read_csv(io.StringIO(data)) # Converts string retrieved from S3 to a file-like object
    return df



def get_gcp_data(project_id, dataset_id, table_id):
    client = bigquery.Client(project=project_id)
    query = f"SELECT * FROM `{project_id}.{dataset_id}.{table_id}`"
    df = client.query(query).to_dataframe()
    return df



# AWS and GCP data have different schemas; normalize them to a common format
def normalize_data(df, source):
    if source == 'aws':
        df = df.rename(columns={
            'lineItem/UnblendedCost': 'cost',
            'lineItem/UsageStartDate': 'date',
            'product/ProductName': 'service'
        })
    elif source == 'gcp':
        df = df.rename(columns={
            'cost': 'cost',
            'usage_start_time': 'date',
            'service.description': 'service'
        })

    # Record the source of the data
    df['source'] = source

    # Drop unneeded columns
    df = df[['cost', 'date', 'service', 'source']]

    df['text_summary'] = df.apply( lambda row: f"On {row['date']}, {row['source']} service {row['service']} cost {row['cost']}.", axis=1)

    return df



# We will only "upsert" data
def save_to_pgvector(df, vectorstore):

    documents = []
    ids = []

    if df.empty:
        print("DataFrame is empty. Skipping.")
        return
    
    for cost, date, service, source, text_summary in zip(df['cost'], df['date'], df['service'], df['source'], df['text_summary']):

        # get raw unique id
        str_date = str(date).split('+')[0] # Chop off timezone to avoid inadvertent hash changes
        raw_id = f"{source}_{str_date}_{service}_{cost}"
        id = hashlib.md5(raw_id.encode()).hexdigest() #generate a unique hash

        #create document
        doc = Document(
            page_content = text_summary,
            metadata = {
                "cost": float(cost),
                "date": date,
                "service": service,
                "source": source
            }
        )

        ids.append(id)
        documents.append(doc)

    # SANITY CHECK: Print the first document to verify format
    if documents:
        first_doc = documents[0]
        print(f"\n--- Quick Check: Document 1 of {len(documents)} ---")
        print(f"ID: {ids[0]}")
        print(f"Content: {first_doc.page_content}")
        print(f"Metadata: {first_doc.metadata}") 
        print("---------------------------------------------------\n")

    print(f"Upserting {len(documents)} documents to Postgres...")

    vectorstore.add_documents(
        ids=ids,
        documents=documents, 
    )

    print("Upload complete.")

    # E2E test to check whether the doc was written properly

    test_query = documents[0].page_content

    print(f"Verifying Upload by searching for: '{test_query[:30]}'...")

    results = vectorstore.similarity_search(
        query=test_query,
        k=1  # just get the top match
    )

    if results:
        match = results[0]
        print("Success! Document found in Postgres.")
        print(f"Retrieved Content: {match.page_content}")
        print(f"Retrieved Metadata: {match.metadata}")
    else:
        print("Warning: Document not found yet. It may still be indexing.")

AWS_BUCKET = os.getenv('AWS_BUCKET')
AWS_FILE_KEY = os.getenv('AWS_FILE_KEY')
GCP_PROJECT_ID = os.getenv('GCP_PROJECT_ID')
GCP_DATASET_ID = os.getenv('GCP_DATASET_ID')
GCP_TABLE_ID = os.getenv('GCP_TABLE_ID')



if __name__ == "__main__":

    # Connect to Postgres and get vector store
    vectorstore = db_utils.get_vector_store()

    # Ingest AWS data
    #aws_df = get_aws_data(AWS_BUCKET, AWS_FILE_KEY)
    aws_df = get_local_data("aws_billing.csv") # TODO: Remove debug
    aws_df = normalize_data(aws_df, 'aws')
    save_to_pgvector(aws_df, vectorstore) 

    # Ingest GCP data
    #gcp_df = get_gcp_data(GCP_PROJECT_ID, GCP_DATASET_ID, GCP_TABLE_ID)
    gcp_df = get_local_data("gcp_billing.csv") # TODO: Remove debug
    gcp_df = normalize_data(gcp_df, 'gcp')
    save_to_pgvector(gcp_df, vectorstore)