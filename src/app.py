import chainlit as cl
from agent_supervisor import supervisor
from memory_manager import memory_manager

@cl.on_chat_start
async def start():
    """Initialisation de la session"""
    session_id = cl.user_session.get("id")
    
    await cl.Message(
        content="""👋 **Bienvenue sur l'Assistant Intelligent Multi-Compétences !**

Je peux vous aider de 3 façons:
1. 📚 **Questions sur documents internes** (politiques, manuels, rapports)
2. 🛠️ **Actions** (calculs, météo, recherche web, todo list)
3. 💬 **Conversation générale**


_Développé avec LangChain & RAG_""",
        author="Assistant"
    ).send()
    
    # Stockage du session_id
    cl.user_session.set("session_id", session_id)

@cl.on_message
async def main(message: cl.Message):
    """Gestion des messages"""
    session_id = cl.user_session.get("session_id")
    user_input = message.content
    
    # Indicateur de chargement
    msg = cl.Message(content="")
    await msg.send()
    
    try:
        # Traitement par le superviseur
        result = supervisor.process(user_input, session_id)
        
        # Construction de la réponse
        output = result["output"]
        
        # Ajout des sources si présentes (RAG)
        if result.get("citations"):
            sources_text = "\n\n---\n📚 **Sources consultées:**\n"
            for i, citation in enumerate(result["citations"][:3], 1):
                sources_text += f"{i}. {citation['source']} (p.{citation['page']})\n"
            output += sources_text
        
        # Badge indiquant le type de traitement
        route_badge = {
            "document": "📚 RAG",
            "tool": "🛠️ Outils",
            "chat": "💬 Conversation",
            "error": "❌ Erreur"
        }.get(result["route_type"], "❓")
        
        msg.content = f"{output}\n\n*{route_badge} (confiance: {result['confidence']:.0%})*"
        await msg.update()
        
        # Ajout d'actions si RAG vide
        if result["route_type"] == "document" and not result.get("sources"):
            actions = [
                cl.Action(
                    name="web_search",
                    value=user_input,
                    description="🔍 Rechercher sur internet",
                    label="Rechercher en ligne"
                )
            ]
            msg.actions = actions
            await msg.update()
            
    except Exception as e:
        msg.content = f"❌ Erreur: {str(e)}"
        await msg.update()

@cl.action_callback("web_search")
async def on_action(action):
    """Callback pour l'action de recherche web"""
    session_id = cl.user_session.get("session_id")
    query = action.value
    
    await cl.Message(content=f"🔍 Recherche web pour: *{query}*...").send()
    
    # Forcer l'utilisation du tool web_search
    from tools_extended import web_search
    result = web_search.invoke(query)
    
    await cl.Message(content=result).send()

@cl.on_chat_end
async def end():
    """Cleanup si nécessaire"""
    pass

if __name__ == "__main__":
    pass