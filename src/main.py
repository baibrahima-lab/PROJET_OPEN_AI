import argparse
import sys
from agent_supervisor import supervisor
from ingestion import ingest_documents
from config import CONFIG

def cli_mode():
    """Mode interactif en ligne de commande"""
    print("=" * 50)
    print("🤖 Assistant Intelligent Multi-Compétences")
    print("=" * 50)
    print("Commandes spéciales:")
    print("  /reset  - Réinitialiser la mémoire")
    print("  /exit   - Quitter")
    print("=" * 50)
    
    session_id = "cli_session"
    
    while True:
        try:
            user_input = input("\n👤 Vous: ").strip()
            
            if not user_input:
                continue
                
            if user_input.lower() == "/exit":
                print("👋 Au revoir!")
                break
            elif user_input.lower() == "/reset":
                from src.memory_manager import memory_manager
                memory_manager.clear_session(session_id)
                print("🧹 Mémoire réinitialisée")
                continue
            
            print("\n🤖 Assistant: ", end="")
            
            result = supervisor.process(user_input, session_id)
            
            print(result["output"])
            print(f"\n[Mode: {result['route_type']} | Confiance: {result['confidence']:.0%}]")
            
            if result.get("sources"):
                print(f"📚 Sources: {', '.join(result['sources'])}")
                
        except KeyboardInterrupt:
            print("\n👋 Au revoir!")
            break
        except Exception as e:
            print(f"❌ Erreur: {e}")

def main():
    parser = argparse.ArgumentParser(description="Assistant Intelligent RAG + Agents")
    parser.add_argument("--ingest", action="store_true", help="Lancer l'ingestion des documents")
    parser.add_argument("--cli", action="store_true", help="Mode ligne de commande")
    parser.add_argument("--web", action="store_true", help="Lancer l'interface web (Streamlit)")
    
    args = parser.parse_args()
    
    if args.ingest:
        print("📥 Démarrage de l'ingestion...")
        ingest_documents()
    elif args.cli:
        cli_mode()
    elif args.web:
        import subprocess
        subprocess.run(["streamlit", "run", "app.py"])
    else:
        # Par défaut: mode CLI
        cli_mode()

if __name__ == "__main__":
    main()