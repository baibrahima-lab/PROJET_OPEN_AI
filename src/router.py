from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from enum import Enum
from typing import Literal
from config import CONFIG

class QueryType(str, Enum):
    DOCUMENT = "document"      # Requête nécessitant le RAG (documents internes)
    TOOL = "tool"             # Requête nécessitant un outil externe (météo, calcul, web)
    CHAT = "chat"             # Conversation générale

class SemanticRouter:
    """
    Routeur intelligent qui analyse l'intention de la requête 
    pour diriger vers le bon module (RAG, Agent ou LLM pur).
    """
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model=CONFIG.MODEL_NAME, 
            temperature=0,
            api_key=CONFIG.OPENAI_API_KEY
        )
        
        self.prompt = PromptTemplate(
            template="""Analyse la requête utilisateur et détermine son type.

RÈGLES DE CLASSIFICATION:
1. DOCUMENT : Si la question concerne des politiques internes, manuels, rapports, procédures, 
   historique de l'entreprise ou documents spécifiques à l'organisation.
   Ex: "Quelle est la politique de congés ?", "Selon le manuel..."

2. TOOL : Si la question nécessite une action externe, un calcul, données temps réel 
   (météo, actualités, calculs mathématiques complexes), ou recherche web.
   Ex: "Quelle est la météo à Paris ?", "Calcule le BSA", "Recherche les dernières news..."

3. CHAT : Si c'est une salutation, question générale, ou conversation sans besoin de données externes.
   Ex: "Bonjour", "Comment ça va ?", "Explique-moi la définition de..."

Réponds UNIQUEMENT au format JSON:
{{"type": "document|tool|chat", "confidence": 0.0-1.0, "reasoning": "explication courte"}}

Requête: {query}""",
            input_variables=["query"]
        )
        
        self.chain = self.prompt | self.llm | JsonOutputParser()
    
    def route(self, query: str) -> tuple[QueryType, float, str]:
        """Route la requête et retourne le type, la confiance et le raisonnement"""
        try:
            result = self.chain.invoke({"query": query})
            query_type = QueryType(result["type"])
            confidence = result["confidence"]
            reasoning = result["reasoning"]
            return query_type, confidence, reasoning
        except Exception as e:
            # Fallback sur CHAT en cas d'erreur
            return QueryType.CHAT, 0.0, f"Erreur de routing: {str(e)}"

# Singleton
router = SemanticRouter()