from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from src.config import OPENAI_API_KEY, MODEL_NAME
from src.tools import tools 
from src.query_engine import medical_knowledge_retrieval 

# 1. On combine tous les outils
all_tools = tools + [medical_knowledge_retrieval]

# 2. On définit le modèle
llm = ChatOpenAI(model=MODEL_NAME, temperature=0, openai_api_key=OPENAI_API_KEY)

# 3. Initialisation de la Mémoire
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# 4. Le Prompt de l'Agent (CORRIGÉ)
prompt = ChatPromptTemplate.from_messages([
    ("system", """Tu es 'Hémo-Expert', un assistant médical de précision spécialisé en onco-hématologie.
    Tu aides les soignants à naviguer dans les recommandations HAS et à réaliser des calculs cliniques.
    
    RÈGLES DE COMPORTEMENT :
    1. DOSAGE : Si l'utilisateur demande un dosage, vérifie d'abord si tu as besoin de calculer la surface corporelle (BSA) ou la clairance.
    2. RECHERCHE : Utilise 'medical_knowledge_retrieval' pour toute question sur les protocoles, avis de la HAS ou médicaments.
    3. CITATION : Cite toujours tes sources et reste factuel.
    4. MÉMOIRE : Utilise l'historique pour éviter de redemander les paramètres du patient (poids, taille, etc.).
    5. ANALYSE : Si l'utilisateur utilise des abréviations (LAL, VS, myélome, chimio), traduis-les mentalement en termes complets avant d'appeler tes outils.
    6. PERSÉVÉRANCE : Ne te contente pas d'une seule recherche. Si un outil ne donne rien, essaie avec une question plus large.
    7. TOLÉRANCE : Ignore les fautes d'orthographe. Concentre-toi sur l'intention clinique.
    8. CONTEXTE : Tu as accès à l'historique. Si le patient a une LAL, toutes les questions suivantes portent sur la LAL sauf indication contraire."""),
    
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# 5. Construction de l'agent
agent = create_openai_tools_agent(llm, all_tools, prompt)

# 6. L'exécuteur avec intégration de la mémoire
agent_executor = AgentExecutor(
    agent=agent, 
    tools=all_tools, 
    memory=memory, 
    verbose=True
)

if __name__ == "__main__":
    # Test de vérification
    print("--- Test de fonctionnement ---")
    agent_executor.invoke({"input": "Le patient pèse 80kg et mesure 1m75. Calcule son BSA."})