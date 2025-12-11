import db_utils
from langchain_google_vertexai import ChatVertexAI

def query_database(query):

    vectorstore = db_utils.get_vector_store()

    results = vectorstore.similarity_search(
        query=query,
        k=2  #TODO: Remove debug -- just get the top two matches for now  
    )

    context_data = ""

    for result in results:
        print(result.page_content) # TODO: Remove debug
        context_data += result.page_content + "\n" # put each 'fact' on a new line

    # Intialize Gemini
    llm = ChatVertexAI(
        model="gemini-2.5-pro",     # best model as of 2025/12/11
        temperature=0               # tells the llm to be accurate, not imaginative
    )

    prompt = f"""
    You are a financial analyst. Answer the question strictly based on the context provided below. 
    If the answer is not in the context, say "I do not have that information."
    
    Context Data:
    {context_data}


    Question: 
    {query}
    """

    # TODO: Remove debug -- print llm response
    print(llm.invoke(prompt).content)
    


if __name__ == "__main__":
    # Test 1: Data that exists
    query_database("How much did Amazon EC2 cost in October of 2023?")
    
    print("\n")

    # Test 2: Hallucination check
    query_database("What is the capital of Australia?") # TODO: Remove debug