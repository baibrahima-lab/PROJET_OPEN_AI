from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_cohere import CohereRerank  # Optionnel, nécessite COHERE_API_KEY
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from typing import List, Dict, Any
from config import CONFIG
import os

class RAGEngine:
    """
    Moteur RAG avancé avec :
    - Expansion de requête (MultiQuery)
    - Reranking (optionnel)
    - Formatage des citations
    - Contextual Compression
    """
    
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(
            model=CONFIG.EMBEDDING_MODEL,
            api_key=CONFIG.OPENAI_API_KEY
        )
        
        # Vérification de l'existence de la base vectorielle
        if not os.path.exists(CONFIG.VECTORSTORE_PATH):
            raise FileNotFoundError(
                f"Base vectorielle non trouvée à {CONFIG.VECTORSTORE_PATH}. "
                "Veuillez d'abord lancer ingestion.py"
            )
        
        self.vectorstore = Chroma(
            persist_directory=CONFIG.VECTORSTORE_PATH,
            embedding_function=self.embeddings
        )
        
        # LLM pour la génération et l'expansion
        self.llm = ChatOpenAI(
            model=CONFIG.MODEL_NAME,
            temperature=0,
            api_key=CONFIG.OPENAI_API_KEY
        )
        
        # Configuration du retriever de base
        base_retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": CONFIG.TOP_K_RETRIEVAL}
        )
        
        # MultiQuery pour expansion sémantique
        self.retriever = MultiQueryRetriever.from_llm(
            retriever=base_retriever,
            llm=self.llm
        )
        
        # Prompt pour génération avec citations explicites
        self.prompt = ChatPromptTemplate.from_template("""
Tu es un assistant expert basé sur les documents internes de l'entreprise.
Utilise UNIQUEMENT le contexte fourni pour répondre. Si tu ne trouves pas la réponse, dis-le honnêtement.

RÈGLES IMPORTANTES:
1. Cite toujours tes sources avec le format [Source: nom_fichier.pdf, Page X]
2. Si plusieurs sources, liste-les toutes à la fin de ta réponse
3. Sois précis et factuel
4. Si le contexte est insuffisant, propose d'effectuer une recherche web

Contexte:
{context}

Question: {question}

Réponse détaillée:""")
    
    def format_docs(self, docs: List[Any]) -> str:
        """Formate les documents avec métadonnées pour le contexte"""
        formatted = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get('source', 'Document inconnu')
            page = doc.metadata.get('page', 'N/A')
            content = doc.page_content
            formatted.append(f"[{i}] Source: {source} (Page {page})\n{content}\n")
        return "\n---\n".join(formatted)
    
    def extract_citations(self, docs: List[Any]) -> List[Dict]:
        """Extrait les informations de citation pour l'UI"""
        citations = []
        for doc in docs:
            citations.append({
                "source": doc.metadata.get('source', 'Inconnu'),
                "page": doc.metadata.get('page', 'N/A'),
                "content": doc.page_content[:200] + "..."  # Aperçu
            })
        return citations
    
    def query(self, question: str) -> Dict[str, Any]:
        """
        Exécute la requête RAG complète.
        Retourne: {"answer": str, "citations": List[Dict], "sources": List[str]}
        """
        # Récupération
        docs = self.retriever.get_relevant_documents(question)
        
        if not docs:
            return {
                "answer": "Je n'ai trouvé aucune information pertinente dans les documents internes. Souhaitez-vous que je recherche sur internet ?",
                "citations": [],
                "sources": []
            }
        
        # Formatage du contexte
        context = self.format_docs(docs)
        
        # Chaîne de génération
        chain = (
            {"context": lambda x: context, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
        
        answer = chain.invoke(question)
        
        return {
            "answer": answer,
            "citations": self.extract_citations(docs),
            "sources": list(set([d.metadata.get('source', 'Inconnu') for d in docs]))
        }

# Singleton
rag_engine = RAGEngine()