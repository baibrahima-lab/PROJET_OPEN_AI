from langchain.tools import tool
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.utilities import OpenWeatherMapAPIWrapper
from typing import Optional
import math
import requests
from datetime import datetime
from config import CONFIG

# Outil de recherche Web (fallback si RAG échoue)
@tool
def web_search(query: str) -> str:
    """
    Recherche des informations à jour sur internet via DuckDuckGo.
    À utiliser lorsque l'information n'est pas disponible dans les documents internes 
    ou pour actualités récentes.
    """
    try:
        search = DuckDuckGoSearchResults()
        results = search.run(query)
        return f"Résultats de recherche web:\n{results}"
    except Exception as e:
        return f"Erreur de recherche web: {str(e)}"

@tool
def get_weather(city: str) -> str:
    """
    Récupère la météo actuelle pour une ville donnée.
    Utilise l'API OpenWeatherMap (nécessite OPENWEATHER_API_KEY).
    """
    if not CONFIG.OPENWEATHER_API_KEY:
        return "Clé API météo non configurée. Veuillez définir OPENWEATHER_API_KEY."
    
    try:
        weather = OpenWeatherMapAPIWrapper(
            openweathermap_api_key=CONFIG.OPENWEATHER_API_KEY
        )
        result = weather.run(city)
        return result
    except Exception as e:
        return f"Erreur météo: {str(e)}"

@tool
def calculate_bsa(height_cm: int, weight_kg: float) -> str:
    """
    Calcule la Surface Corporelle (BSA) formule de Mosteller.
    Usage: calculate_bsa(height_cm=180, weight_kg=80)
    """
    if height_cm <= 0 or weight_kg <= 0:
        return "Erreur: Les valeurs doivent être positives"
    
    bsa = math.sqrt((height_cm * weight_kg) / 3600)
    return f"Surface Corporelle (BSA): {bsa:.2f} m²\nFormule: √({height_cm} × {weight_kg} / 3600)"

@tool
def calculate_creatinine_clearance(age: int, weight_kg: float, 
                                   creatinine_umol: float, is_female: bool) -> str:
    """
    Calcule la clairance de créatinine (Cockcroft-Gault).
    Usage: calculate_creatinine_clearance(age=65, weight_kg=70, creatinine_umol=100, is_female=True)
    """
    if creatinine_umol <= 0:
        return "Erreur: Créatinine doit être > 0"
    
    # Facteur de correction féminin
    k = 0.85 if is_female else 1.0
    
    clearance = ((140 - age) * weight_kg * k) / (creatinine_umol * 0.0113)
    
    interpretation = "Normale" if clearance >= 60 else "Insuffisance modérée" if clearance >= 30 else "Insuffisance sévère"
    
    return (f"Clairance créatinine (Cockcroft-Gault): {clearance:.2f} ml/min\n"
            f"Interprétation: {interpretation}")

@tool
def calculator(expression: str) -> str:
    """
    Calculateur mathématique sécurisé pour expressions simples.
    Usage: calculator(expression="(150 * 2.5) / 3")
    """
    try:
        # Liste blanche de caractères autorisés pour sécurité
        allowed_chars = set('0123456789+-*/.() ')
        if not all(c in allowed_chars for c in expression):
            return "Erreur: Caractères non autorisés. Utilisez uniquement: 0-9, +, -, *, /, (, ), ."
        
        result = eval(expression, {"__builtins__": {}}, {})
        return f"Résultat: {result}"
    except Exception as e:
        return f"Erreur de calcul: {str(e)}"

@tool
def save_to_todo(task: str, priority: str = "normal") -> str:
    """
    Sauvegarde une tâche dans une todo list locale (fichier texte).
    Usage: save_to_todo(task="Relire le rapport", priority="high")
    """
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        with open("todo_list.txt", "a", encoding="utf-8") as f:
            f.write(f"[{timestamp}] [{priority.upper()}] {task}\n")
        return f"✅ Tâche ajoutée: '{task}' (Priorité: {priority})"
    except Exception as e:
        return f"Erreur sauvegarde: {str(e)}"

@tool
def get_current_date() -> str:
    """
    Retourne la date et l'heure actuelles.
    """
    now = datetime.now()
    return f"Date actuelle: {now.strftime('%d/%m/%Y %H:%M')}"

# Liste des outils disponibles pour l'agent
general_tools = [
    web_search,
    get_weather,
    calculator,
    calculate_bsa,
    calculate_creatinine_clearance,
    save_to_todo,
    get_current_date
]