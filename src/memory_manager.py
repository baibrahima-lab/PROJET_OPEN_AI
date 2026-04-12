from langchain.memory import ConversationBufferWindowMemory
from langchain_core.messages import HumanMessage, AIMessage
from typing import List, Dict
import json
import os

class MemoryManager:
    """
    Gestionnaire de mémoire conversationnelle avec persistance optionnelle.
    Utilise BufferWindow pour garder uniquement les N derniers échanges (token efficient).
    """
    
    def __init__(self, window_size: int = 5):
        self.memories: Dict[str, ConversationBufferWindowMemory] = {}
        self.window_size = window_size
        self.persist_dir = "memory_sessions/"
        
        # Création du dossier de persistance si besoin
        os.makedirs(self.persist_dir, exist_ok=True)
    
    def _get_memory(self, session_id: str) -> ConversationBufferWindowMemory:
        """Récupère ou crée la mémoire pour une session"""
        if session_id not in self.memories:
            self.memories[session_id] = ConversationBufferWindowMemory(
                k=self.window_size,
                memory_key="chat_history",
                return_messages=True
            )
            
            # Tentative de chargement depuis fichier
            self._load_session(session_id)
            
        return self.memories[session_id]
    
    def get_history(self, session_id: str) -> List:
        """Retourne l'historique sous forme de liste de Messages"""
        memory = self._get_memory(session_id)
        return memory.load_memory_variables({})["chat_history"]
    
    def add_exchange(self, session_id: str, user_msg: str, ai_msg: str):
        """Ajoute un échange à la mémoire"""
        memory = self._get_memory(session_id)
        memory.save_context(
            {"input": user_msg},
            {"output": ai_msg}
        )
        self._save_session(session_id, memory)
    
    def clear_session(self, session_id: str):
        """Réinitialise une session"""
        if session_id in self.memories:
            del self.memories[session_id]
        
        filepath = os.path.join(self.persist_dir, f"{session_id}.json")
        if os.path.exists(filepath):
            os.remove(filepath)
    
    def _save_session(self, session_id: str, memory: ConversationBufferWindowMemory):
        """Persistance sur disque (optionnel)"""
        try:
            history = memory.load_memory_variables({})["chat_history"]
            data = []
            for msg in history:
                if isinstance(msg, HumanMessage):
                    data.append({"role": "human", "content": msg.content})
                elif isinstance(msg, AIMessage):
                    data.append({"role": "ai", "content": msg.content})
            
            filepath = os.path.join(self.persist_dir, f"{session_id}.json")
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)
        except Exception as e:
            print(f"Erreur sauvegarde mémoire: {e}")
    
    def _load_session(self, session_id: str):
        """Chargement depuis disque"""
        filepath = os.path.join(self.persist_dir, f"{session_id}.json")
        if os.path.exists(filepath):
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                memory = self.memories[session_id]
                for item in data:
                    if item["role"] == "human":
                        memory.chat_memory.add_user_message(item["content"])
                    else:
                        memory.chat_memory.add_ai_message(item["content"])
            except Exception as e:
                print(f"Erreur chargement mémoire: {e}")

# Singleton
memory_manager = MemoryManager()