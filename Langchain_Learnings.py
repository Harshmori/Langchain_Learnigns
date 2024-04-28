
import os
import openai
import langchain
from pinecone import Pinecone 
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings 
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone as ps
from langchain_pinecone import PineconeVectorStore
from langchain.llms import OpenAI
#import lanchain and pinecode packages


from langchain.chains.question_answering import load_qa_chain
from langchain import OpenAI


def read_doc(directory):
    file_loader=PyPDFDirectoryLoader(directory)
    documents=file_loader.load()
    return documents
#for reading pdf file


doc=read_doc('docs/')


def chunk_data(docs,chunk_size=800,chunk_overlap=50):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap)
    doc=text_splitter.split_documents(docs)
    return doc
#to devide the pdf into chunks


documents=chunk_data(docs=doc)


os.environ['OPENAI_API_KEY'] = 'OPENAI_API_KEY'
embeddings=OpenAIEmbeddings(api_key=os.environ['OPENAI_API_KEY'])
#using openai to pass embeddings model


embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
#using open-source embeddings model


vectors=embeddings.embed_query("who is the prime minister of india?")
#this is how vectors get generated from single sentence 


os.environ['PINECONE_API_KEY'] = 'PINECONE_API_KEY'
index_name = 'FirstIndex'
index = PineconeVectorStore.from_documents(
        doc,
        index_name=index_name,
        embedding=embeddings,
        namespace="PDF1"
    )
#this is how you can injest data into pinecone index 


query = "what does krishna says to arjun about his form?"
index.similarity_search(query, k=2, namespace="gita")
#a simplle example of similarity search in an index


embeddings = OpenAIEmbeddings()
docsearch = Pinecone.from_existing_index(index_name=index_name, embedding=embeddings, namespace="your_namespace_name")
#this is how you can inmport index with vectors from pinecone


index.delete(namespace="user1", delete_all=True)
#delete an unwanted index


def retrieve_query(query,k=2):
    matching_results=index.similarity_search(query,k=k)
    return matching_results



llm=OpenAI()
chain=load_qa_chain(llm,chain_type="stuff")
#load QA chain


def retrieve_answers(query):
    doc_search=retrieve_query(query)
    print(doc_search)
    response=chain.run(input_documents=doc_search,question=query)
    return response


our_query = "how did arjuna observed from the middle of the battlefield, also you must give sanskrit shloka and chapter and verse numbers?"
answer = retrieve_answers(our_query)
print(answer)


#namespaces are one of the best features in the pinecone , this way you can store vectors of multiple docs into a single index
#thus you dont need to traverse whole inndex when searching answer in vectors of one pdf only


