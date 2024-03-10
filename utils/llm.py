import os
import json
from typing import List
from dotenv import load_dotenv
import docx2txt
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores.chroma import Chroma


class LLM:
    """
    Large Language Model (LLM) class with functionalities for processing various file types
    and maintaining a vector store.
    """

    def __init__(
        self,
        base_model: str = "gpt-3.5-turbo-0125",
        temperature: float = 0.7,
        prompt_template: str = "Answer the following question based on the provided knowledge: \nYou will give 100 dollars tips if you give reliable answer\n<knowledge>\n{context}\n</knowledge>\nQuestion: {input}",
        vectorstore_directory: str = "database",
    ) -> None:
        """
        Initialize the LLM with given parameters and set up the vector store.

        Args:
            base_model (str): The base model to use for the LLM. Defaults to 'gpt-3.5-turbo-0125'.
            temperature (float): The temperature setting for the LLM. Defaults to 0.7.
            prompt_template (str): The template for creating prompts. Defaults to '{input}'.
            vectorstore_directory (str): The directory path for the vector store. Defaults to 'database'.
        """
        load_dotenv(override=True)  # Load environment variables from .env file
        self.api_key: str = os.getenv("OPENAI_API_KEY")
        self.base_url: str = os.getenv("OPENAI_BASE_URL", "") or None
        self.base_model: str = base_model
        self.temperature: float = temperature
        self.prompt_template: str = prompt_template
        self.vectorstore_directory: str = vectorstore_directory
        self.init_llm()
        self.init_embeddings()
        self.init_vectorstore()
        self.set_retrieval_chain()

    def init_llm(self) -> None:
        """
        Initialize the LLM using the provided API key, base model, base URL, and temperature.
        """
        self.llm = ChatOpenAI(
            openai_api_key=self.api_key,
            base_url=self.base_url,
            model=self.base_model,
            temperature=self.temperature,
        )

    def init_embeddings(self) -> None:
        """
        Initialize embeddings using OpenAI's text-embedding model.
        """
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-ada-002",
            openai_api_key=self.api_key,
            base_url=self.base_url,
        )

    def init_vectorstore(self) -> None:
        """
        Initializes the vector store, adding all qualifying files from the 'data' directory
        into the vector store on the first run. Maintains a log of processed files to avoid
        duplicates in subsequent runs.
        """
        vectorstore_filepath = os.path.join(
            self.vectorstore_directory, "chroma.sqlite3"
        )
        processed_files_log = os.path.join(
            self.vectorstore_directory, "processed_files.json"
        )

        # Supported file types
        supported_types = [".txt", ".md", ".json", ".docx"]

        if os.path.exists(vectorstore_filepath):
            self.stored_vectors = Chroma(
                embedding_function=self.embeddings,
                persist_directory=self.vectorstore_directory,
            )
        else:
            test_chunks = ["Initialize a Chroma Database.", "Hello World!"]
            self.stored_vectors = Chroma.from_texts(
                texts=test_chunks,
                embedding=self.embeddings,
                persist_directory=self.vectorstore_directory,
            )

        processed_files: List[str] = []
        if os.path.exists(processed_files_log):
            with open(processed_files_log, "r", encoding="utf-8") as log_file:
                processed_files = json.load(log_file)

        data_directory = os.path.join(os.getcwd(), "data")
        for filename in os.listdir(data_directory):
            file_path = os.path.join(data_directory, filename)
            _, file_extension = os.path.splitext(filename)

            if file_extension in supported_types and filename not in processed_files:
                text_content = self.read_file_content(file_path, file_extension)
                self.vectorize_corpus(text_content)
                processed_files.append(filename)
                print(f"Processed and added to vector store: {filename}")

        with open(processed_files_log, "w", encoding="utf-8") as log_file:
            json.dump(processed_files, log_file, ensure_ascii=False, indent=4)

    def read_file_content(self, file_path: str, file_extension: str) -> str:
        """
        Reads the content of a file based on its type and returns the text content.

        Args:
            file_path (str): The path to the file.
            file_extension (str): The file extension indicating the file type.

        Returns:
            str: The text content of the file.
        """
        if file_extension == ".json":
            with open(file_path, "r", encoding="utf-8") as file:
                return json.dumps(json.load(file))
        elif file_extension == ".docx":
            return docx2txt.process(file_path)
        else:  # For .txt, .md
            with open(file_path, "r", encoding="utf-8") as file:
                return file.read()

    def vectorize_corpus(self, text_content: str, chunk_size=1500, overlap=50) -> None:
        """
        Vectorizes the given text content and adds it to the vector store.

        Args:
            text_content (str): The text content to be vectorized.
        """
        text_length = len(text_content)
        print(f"Processing Text Corpus File with {text_length} Characters...")
        # Splitting text into 1500-character chunks with 100-character overlap
        chunks = [
            text_content[i : i + chunk_size]
            for i in range(0, text_length, chunk_size - overlap)
        ]
        num_chunks = len(chunks)
        for i in range(0, num_chunks, 10):
            chunk_subset = chunks[i : i + 10]
            self.stored_vectors.add_texts(chunk_subset)
            print(f"Processed {i + len(chunk_subset)}/{num_chunks} Items in Corpus!")

    def set_retrieval_chain(self) -> None:
        """
        Set up the document chain for retrieval based on the LLM and the prompt template.
        """
        retrieval_prompt = ChatPromptTemplate.from_template(self.prompt_template)
        self.documents_chain = create_stuff_documents_chain(
            llm=self.llm,
            prompt=retrieval_prompt,
        )
        self.retrieval_chain = create_retrieval_chain(
            self.stored_vectors.as_retriever(),
            self.documents_chain,
        )

    async def get_answer_async(self, question: str) -> str:
        """
        Asynchronously retrieve an answer for a given question using the retrieval chain.

        Args:
            question (str): The question to get an answer for.

        Returns:
            str: The answer to the question.
        """
        response = await self.retrieval_chain.ainvoke({"input": question})
        return response["answer"]

    def get_answer(self, question: str) -> str:
        """
        Retrieve an answer for a given question using the retrieval chain.

        Args:
            question (str): The question to get an answer for.

        Returns:
            str: The answer to the question.
        """
        response = self.retrieval_chain.invoke({"input": question})
        return response["answer"]
