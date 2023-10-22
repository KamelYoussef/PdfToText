from dotenv import load_dotenv
from Functions import *
import fitz

# Loaders
from langchain.schema import Document

# Splitters
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Model
from langchain.chat_models import ChatOpenAI
from langchain.llms import HuggingFaceHub

# Embedding Support
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings

# Summarizer we'll use for Map Reduce
from langchain.chains.summarize import load_summarize_chain

from langchain import PromptTemplate


load_dotenv()
llm = ChatOpenAI()

# llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature": 0.5, "max_length": 512})


def main():
    my_path = "data/book.pdf"
    pdf = fitz.open(my_path)
    text = merge_text(pdf)

    print(f"This book has {llm.get_num_tokens(text)} tokens in it")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=3000)

    docs = text_splitter.create_documents([text])

    print(f"Now our book is split up into {len(docs)} documents")

    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")

    vectors = embeddings.embed_documents([x.page_content for x in docs])

    selected_indices = clustering(vectors)

    llm3 = ChatOpenAI(temperature=0, max_tokens=1000, model="gpt-3.5-turbo")

    map_prompt = """
    You will be given a single passage of a book. This section will be enclosed in triple backticks (```)
    Your goal is to give a summary of this section so that a reader will have a full understanding of what happened.
    Your response should be at least three paragraphs and fully encompass what was said in the passage.

    ```{text}```
    FULL SUMMARY:
    """
    map_prompt_template = PromptTemplate(template=map_prompt, input_variables=["text"])
    map_chain = load_summarize_chain(
        llm=llm3, chain_type="stuff", prompt=map_prompt_template
    )

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
    combine_prompt_template = PromptTemplate(
        template=combine_prompt, input_variables=["text"]
    )

    reduce_chain = load_summarize_chain(
        llm=llm3,
        chain_type="stuff",
        prompt=combine_prompt_template,
        # verbose=True # Set this to true if you want to see the inner workings
    )
    output = reduce_chain.run([summaries])
    print(output)


if __name__ == "__main__":
    main()
