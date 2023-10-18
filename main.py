from dotenv import load_dotenv
from Functions import *
import fitz
from langchain.document_loaders import PyPDFLoader
# Loaders
from langchain.schema import Document
# Splitters
from langchain.text_splitter import RecursiveCharacterTextSplitter
# Model
from langchain.chat_models import ChatOpenAI
# Embedding Support
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
# Summarizer we'll use for Map Reduce
from langchain.chains.summarize import load_summarize_chain
# Data Science
import numpy as np
from sklearn.cluster import KMeans
from langchain import PromptTemplate

# Taking out the warnings
import warnings
from warnings import simplefilter
from langchain.llms import HuggingFaceHub

load_dotenv()
llm = ChatOpenAI()
#llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature": 0.5, "max_length": 512})
def get_pdf_text(path):
    # Load the book
    loader = PyPDFLoader(path)
    pages = loader.load()

    # Cut out the open and closing parts
    pages = pages[26:277]

    # Combine the pages, and replace the tabs with spaces
    text = ""

    for page in pages:
        text += page.page_content

    text = text.replace('\t', ' ')
    num_tokens = llm.get_num_tokens(text)

    print(f"This book has {num_tokens} tokens in it")
    return text


def main():

    my_path = "data/test1.pdf"
    pdf = fitz.open(my_path)
    text = merge_text(pdf)

    num_tokens = llm.get_num_tokens(text)
    print(f"This book has {num_tokens} tokens in it")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000,
                                                   chunk_overlap=3000)

    docs = text_splitter.create_documents([text])
    num_documents = len(docs)

    print(f"Now our book is split up into {num_documents} documents")
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")

    vectors = embeddings.embed_documents([x.page_content for x in docs])

    # Assuming 'embeddings' is a list or array of 1536-dimensional embeddings

    # Choose the number of clusters, this can be adjusted based on the book's content.
    # I played around and found ~10 was the best.
    # Usually if you have 10 passages from a book you can tell what it's about
    num_clusters = 5

    # Perform K-means clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42).fit(vectors)

    labels = kmeans.labels_

    # Filter out FutureWarnings
    simplefilter(action='ignore', category=FutureWarning)

    # Find the closest embeddings to the centroids

    # Create an empty list that will hold your closest points
    closest_indices = []

    # Loop through the number of clusters you have
    for i in range(num_clusters):
        # Get the list of distances from that particular cluster center
        distances = np.linalg.norm(vectors - kmeans.cluster_centers_[i], axis=1)

        # Find the list position of the closest one (using argmin to find the smallest distance)
        closest_index = np.argmin(distances)

        # Append that position to your closest indices list
        closest_indices.append(closest_index)

    selected_indices = sorted(closest_indices)
    llm3 = ChatOpenAI(temperature=0,
                      max_tokens=1000,
                      model='gpt-3.5-turbo'
                      )
    map_prompt = """
    You will be given a single passage of a book. This section will be enclosed in triple backticks (```)
    Your goal is to give a summary of this section so that a reader will have a full understanding of what happened.
    Your response should be at least three paragraphs and fully encompass what was said in the passage.

    ```{text}```
    FULL SUMMARY:
    """
    map_prompt_template = PromptTemplate(template=map_prompt, input_variables=["text"])
    map_chain = load_summarize_chain(llm=llm3,
                                     chain_type="stuff",
                                     prompt=map_prompt_template)

    selected_docs = [docs[doc] for doc in selected_indices]

    # Make an empty list to hold your summaries
    summary_list = []

    # Loop through a range of the lenght of your selected docs
    for i, doc in enumerate(selected_docs):
        # Go get a summary of the chunk
        chunk_summary = map_chain.run([doc])

        # Append that summary to your list
        summary_list.append(chunk_summary)

    summaries = "\n".join(summary_list)

    # Convert it back to a document
    summaries = Document(page_content=summaries)

    combine_prompt = """
    You will be given a series of summaries from a book. The summaries will be enclosed in triple backticks (```)
    Your goal is to give a verbose summary of what happened in the story.
    The reader should be able to grasp what happened in the book.

    ```{text}```
    VERBOSE SUMMARY:
    """
    combine_prompt_template = PromptTemplate(template=combine_prompt, input_variables=["text"])

    reduce_chain = load_summarize_chain(llm=llm3,
                                        chain_type="stuff",
                                        prompt=combine_prompt_template,
                                        #                              verbose=True # Set this to true if you want to see the inner workings
                                        )
    output = reduce_chain.run([summaries])
    print(output)

if __name__ == '__main__':
    main()