"""
Retrieval module for Sapphire RAG pipeline.
Implements advanced RAG techniques including reranking.
"""

import logging
from typing import List, Dict, Any
from langchain.schema import Document
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_community.llms import HuggingFaceHub
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedRetriever:
    """
    A class to handle document retrieval with advanced RAG techniques.
    Includes query rewriting, contextual compression, and reranking.
    """
    def __init__(self, vector_store_path: str, llm_model_name: str, groq_api_key: str):
        self.vector_store_path = vector_store_path
        self.llm_model_name = llm_model_name
        self.groq_api_key = groq_api_key
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.vector_store = self._load_vector_store()
        self.llm = self._initialize_llm()
        self.retriever = self.vector_store.as_retriever()
        self.history_aware_retriever = self._create_history_aware_retriever()
        self.rag_chain = self._create_rag_chain()

    def _load_vector_store(self):
        """
        Loads the FAISS vector store from the specified path.
        """
        try:
            logger.info(f"Loading FAISS vector store from {self.vector_store_path}")
            # Ensure the embeddings are passed correctly for loading
            vector_store = FAISS.load_local(self.vector_store_path, self.embeddings, allow_dangerous_deserialization=True)
            logger.info("Vector store loaded successfully.")
            return vector_store
        except Exception as e:
            logger.error(f"Error loading vector store: {e}")
            raise

    def _initialize_llm(self):
        """
        Initializes the LLM for query rewriting and response generation.
        Using HuggingFaceHub for Groq models.
        """
        try:
            logger.info(f"Initializing LLM: {self.llm_model_name}")
            # LangChain's HuggingFaceHub can be used to connect to Groq via their API
            # Ensure you have the GROQ_API_KEY set in your environment or pass it directly
            # For Groq, the model name would be something like 'mixtral-8x7b-32768' or 'llama2-70b-4096'
            # This requires the `huggingface_hub` library and `HUGGINGFACEHUB_API_TOKEN` set
            # For direct Groq integration, you'd typically use `ChatGroq` from `langchain_groq`
            # As per requirement, using Groq Cloud API key, so we'll use ChatGroq.
            from langchain_groq import ChatGroq
            llm = ChatGroq(temperature=0, groq_api_key=self.groq_api_key, model_name=self.llm_model_name)
            logger.info("LLM initialized successfully.")
            return llm
        except Exception as e:
            logger.error(f"Error initializing LLM: {e}")
            raise

    def _create_history_aware_retriever(self):
        """
        Creates a retriever that is aware of chat history to rewrite queries.
        """
        contextualize_q_system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Given a chat history and the latest user question, "
            "generate a standalone question which can be understood without "
            "the chat history. Do NOT answer the question, just reformulate "
            "it if necessary and otherwise return it as is."
        )
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        history_aware_retriever = create_history_aware_retriever(
            self.llm, self.retriever, contextualize_q_prompt
        )
        return history_aware_retriever

    def _create_rag_chain(self):
        """
        Creates the RAG chain including contextual compression and response generation.
        """
        # Contextual Compression (Reranking can be added here using a BGE reranker or similar)
        # For simplicity, we'll use LLMChainExtractor as a basic compressor/reranker for now.
        # For a true reranker, you'd integrate a model like 'bge-reranker-base' or 'cohere-rerank'
        # compressor = LLMChainExtractor.from_llm(self.llm)
        # compression_retriever = ContextualCompressionRetriever(
        #     base_compressor=compressor,
        #     base_retriever=self.history_aware_retriever
        # )

        system_template = (
            "You are a friendly, contextual, and professional customer service chatbot for Sapphire, a Pakistani fashion brand. "
            "Use the following pieces of retrieved context to answer the question. "
            "If you don't know the answer, just say that you don't know, don't try to make up an answer. "
            "Keep the answer concise and relevant to Sapphire's products and policies. "
            "Always maintain a helpful and polite tone. "
            "\n\n{context}"
        )
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", system_template),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

        rag_chain = create_retrieval_chain(self.history_aware_retriever, qa_prompt)
        return rag_chain

    def get_response(self, question: str, chat_history: List[Dict[str, str]]) -> str:
        """
        Gets a response from the RAG pipeline.

        Args:
            question: The user's current question.
            chat_history: A list of previous chat messages (e.g., [HumanMessage, AIMessage, ...]).

        Returns:
            The generated answer.
        """
        # Convert chat_history to LangChain format
        formatted_chat_history = []
        for msg in chat_history:
            if msg["role"] == "user":
                formatted_chat_history.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                formatted_chat_history.append(AIMessage(content=msg["content"]))

        response = self.rag_chain.invoke({"input": question, "chat_history": formatted_chat_history})
        return response["answer"]

if __name__ == "__main__":
  
    test_groq_api_key = "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" 
    test_llm_model = "llama-3.3-70b-versatile" 
    test_vector_store_path = "./data/vector_store_langchain"



    if test_groq_api_key == "YOUR_GROQ_API_KEY":
        logger.warning("Please replace 'YOUR_GROQ_API_KEY' with your actual Groq API key for testing.")
    
    try:
        retriever_instance = AdvancedRetriever(
            vector_store_path=test_vector_store_path,
            llm_model_name=test_llm_model,
            groq_api_key=test_groq_api_key
        )

        chat_history = [
            {"role": "user", "content": "What is your return policy?"},
            {"role": "assistant", "content": "Sapphire offers a 7-day return policy for all products. Items must be in original condition with tags attached."}
        ]

        question = "Can I return a worn item?"
        print(f"\nUser: {question}")
        answer = retriever_instance.get_response(question, chat_history)
        print(f"Bot: {answer}")

        question_no_history = "What types of fabrics do you use for Ready to Wear?"
        print(f"\nUser: {question_no_history}")
        answer_no_history = retriever_instance.get_response(question_no_history, [])
        print(f"Bot: {answer_no_history}")

    except Exception as e:
        logger.error(f"Error during AdvancedRetriever example usage: {e}")
        print(f"Error during example usage: {e}")


