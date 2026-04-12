from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from typing import Dict, Any, List
from config import CONFIG
from router import router, QueryType
from query_engine import rag_engine
from tools import general_tools
from memory_manager import memory_manager

class SupervisorAgent:
    """
    Agent superviseur qui orchestre entre:
    1. RAG (documents internes)
    2. Outils externes (météo, web, calculs)
    3. Réponse directe LLM (conversation)
    """
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model=CONFIG.MODEL_NAME,
            temperature=CONFIG.TEMPERATURE,
            api_key=CONFIG.OPENAI_API_KEY
        )
        self.router = router
        
        # Agent pour les outils (quand on a besoin d'actions externes)
        self.tools_agent = self._create_tools_agent()
        
    def _create_tools_agent(self):
        """Crée l'agent spécialisé dans l'utilisation d'outils"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Tu es un assistant polyvalent capable d'utiliser des outils.
Lorsque tu utilises un outil, explique brièvement pourquoi tu l'utilises.
Si tu fais des calculs, montre les étapes intermédiaires quand c'est pertinent.
Sois précis et concis."""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        agent = create_openai_tools_agent(self.llm, general_tools, prompt)
        return AgentExecutor(
            agent=agent,
            tools=general_tools,
            verbose=True,
            handle_parsing_errors=True
        )
    
    def process(self, user_input: str, session_id: str = "default") -> Dict[str, Any]:
        """
        Point d'entrée principal. Route la requête vers le bon module.
        """
        # Récupération de l'historique
        chat_history = memory_manager.get_history(session_id)
        
        # 1. ROUTING - Déterminer le type de requête
        query_type, confidence, reasoning = self.router.route(user_input)
        
        print(f"[ROUTER] Type: {query_type.value}, Confiance: {confidence}, Raison: {reasoning}")
        
        response_data = {
            "input": user_input,
            "route_type": query_type.value,
            "confidence": confidence,
            "reasoning": reasoning,
            "output": "",
            "sources": [],
            "citations": []
        }
        
        # 2. EXECUTION selon le type
        try:
            if query_type == QueryType.DOCUMENT and confidence > 0.7:
                # RAG Pipeline
                rag_result = rag_engine.query(user_input)
                response_data["output"] = rag_result["answer"]
                response_data["citations"] = rag_result["citations"]
                response_data["sources"] = rag_result["sources"]
                
                # Fallback sur web search si RAG vide
                if not rag_result["sources"]:
                    response_data["output"] += "\n\nSouhaitez-vous que je recherche cette information sur internet ?"
                
            elif query_type == QueryType.TOOL and confidence > 0.7:
                # Agent avec outils
                result = self.tools_agent.invoke({
                    "input": user_input,
                    "chat_history": chat_history
                })
                response_data["output"] = result["output"]
                
            else:
                # Conversation directe avec LLM (CHAT ou confiance faible)
                messages = [
                    SystemMessage(content="""Tu es un assistant intelligent et bienveillant.
Tu peux discuter de sujets généraux. Si la question semble concerner des documents 
internes ou nécessiter des outils spécifiques que tu ne possèdes pas, suggère-le poliment."""),
                    *chat_history,
                    HumanMessage(content=user_input)
                ]
                
                response = self.llm.invoke(messages)
                response_data["output"] = response.content
                
        except Exception as e:
            response_data["output"] = f"Une erreur est survenue: {str(e)}"
            response_data["route_type"] = "error"
        
        # 3. MISE À JOUR MÉMOIRE
        memory_manager.add_exchange(session_id, user_input, response_data["output"])
        
        return response_data

# Singleton
supervisor = SupervisorAgent()