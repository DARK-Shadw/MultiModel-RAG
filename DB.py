from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from util import check_exist
import time
import uuid
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from unstructured.partition.pdf import partition_pdf
from langchain_core.messages import HumanMessage
from langchain_together import ChatTogether
from langchain.storage import InMemoryStore
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.schema import Document
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class PDFLoader:
    def __init__(self, path):
        self.pdf_path = path

        # Retrieve API keys from environment variables
        together_api_key = os.getenv("TOGETHER_API_KEY")
        google_api_key = os.getenv("GOOGLE_API_KEY")

        # Validate API keys
        if not together_api_key:
            raise ValueError("TOGETHER_API_KEY not found in environment variables")
        if not google_api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")

        self.text_llm = ChatTogether(
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
            together_api_key=together_api_key
        )
        self.vision_llm = ChatTogether(
            model="meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo",
            together_api_key=together_api_key
        )

        self.embedding = GoogleGenerativeAIEmbeddings(
            google_api_key=google_api_key,
            model="models/embedding-001",
        )

        chunks = partition_pdf(
            filename=self.pdf_path,
            infer_table_structure=True,
            strategy="hi_res",
            extract_image_block_types=["Image"],   # Add 'Table' to list to extract image of tables
            extract_image_block_to_payload=True,   # if true, will extract base64 for API usage
            chunking_strategy="by_title",          # or 'basic'
            max_characters=10000,                  # defaults to 500
            combine_text_under_n_chars=2000,       # defaults to 0
            new_after_n_chars=6000,
        )
        
        self.chunks = chunks
        self.texts, self.tables = self.get_text_tables()
        self.images = self.get_imagesb64()
        self.text_summaries, self.table_summaries = self.create_summary()
        self.image_summaries = self.create_img_summary()

        self.vDB = Chroma(collection_name="ref-rag", embedding_function=self.embedding)
        self.docDB = InMemoryStore()
        self.id_key = "doc_id"
        self.retriever = MultiVectorRetriever(
            vectorstore=self.vDB,
            docstore=self.docDB,
            id_key=self.id_key,
        )
        self.bind_data()

    def bind_data(self):
        doc_ids = [str(uuid.uuid4()) for _ in self.texts]
        summary_texts = [
            Document(page_content=summary, metadata={self.id_key: doc_ids[i]}) for i, summary in enumerate(self.text_summaries)
        ]
        self.retriever.vectorstore.add_documents(summary_texts)
        self.retriever.docstore.mset(list(zip(doc_ids, self.texts)))

        if self.tables:
            table_ids = [str(uuid.uuid4()) for _ in self.tables]
            summary_tables = [
                Document(page_content=summary, metadata={self.id_key: table_ids[i]}) for i, summary in enumerate(self.table_summaries)
            ]
            self.retriever.vectorstore.add_documents(summary_tables)
            self.retriever.docstore.mset(list(zip(table_ids, self.tables)))

        img_ids = [str(uuid.uuid4()) for _ in self.images]
        summary_imgs = [
            Document(page_content=summary, metadata={self.id_key: img_ids[i]}) for i, summary in enumerate(self.image_summaries)
        ]
        self.retriever.vectorstore.add_documents(summary_imgs)
        self.retriever.docstore.mset(list(zip(img_ids, self.images)))

    def create_img_summary(self):
        img_summaries = []
        print(f"Running Summary for {len(self.images)} Images.")
        for element in self.images:
            prompt_text = HumanMessage(
                content=[
                    {"type": "text", "text": "describe the image in detail"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{element}"},
                    },
                ],
            )
            result = self.vision_llm.invoke([prompt_text])
            img_summaries.append(result.content)
        return img_summaries

    def create_summary(self):
        text_summaries = []
        table_summaries = []
        count = 0
        print(f"Running Summary for {len(self.texts)} texts.")
        for element in self.texts:
            prompt_text = f"""
            You are an assistant tasked with summarizing tables and text.
            Give a concise summary of the tables or text.

            Respond with only summary, no additional comment.
            Do not start your message by saying "Here is a summary" or anything like that.
            Just provide the summary as it is.

            Table or text chunk: {element}
            """
            count += 1
            print(f"API Usage: {count}")
            if count % 10 == 0:
                print(f"Sleeping..")
                time.sleep(0.5)
            result = self.text_llm.invoke([HumanMessage(prompt_text)])
            text_summaries.append(result.content)
        
        for element in self.tables:
            prompt_text = f"""
            You are an assistant tasked with summarizing tables and text.
            Give a concise summary of the tables or text.

            Respond with only summary, no additional comment.
            Do not start your message by saying "Here is a summary" or anything like that.
            Just provide the summary as it is.

            Table or text chunk: {element}
            """
            count += 1
            print(f"API Usage: {count}")
            result = self.text_llm.invoke([HumanMessage(prompt_text)])
            table_summaries.append(result.content)
        
        return text_summaries, table_summaries

    def get_text_tables(self):
        tables = []
        texts = []
        for chunk in self.chunks:
            if "Table" in str(type(chunk)):
                tables.append(chunk)
            if "CompositeElement" in str(type(chunk)):
                texts.append(chunk)
        return texts, tables
    
    def get_imagesb64(self):
        image_b64 = []
        for chunk in self.chunks:
            if "CompositeElement" in str(type(chunk)):
                chunk_elem = chunk.metadata.orig_elements
                for el in chunk_elem:
                    if "Image" in str(type(el)):
                        image_b64.append(el.metadata.image_base64)
        return image_b64

    def getchunks(self):
        return self.chunks
    
    def gettexts(self):
        return self.texts
