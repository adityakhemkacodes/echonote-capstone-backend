# app/modules/entity_recognition.py
import spacy
import warnings
warnings.filterwarnings('ignore')

# Try to load spacy model
_nlp = None

def get_nlp():
    """Lazy load spacy model"""
    global _nlp
    if _nlp is None:
        try:
            print("Loading spaCy model...")
            _nlp = spacy.load("en_core_web_sm")
        except:
            print("⚠ spaCy model not found. Run: python -m spacy download en_core_web_sm")
            _nlp = "unavailable"
    return _nlp


def extract_entities(text: str):
    """
    Extract named entities (people, orgs, locations, etc.) using spaCy.
    """
    print("Extracting named entities...")
    
    text = (text or "").strip()
    if not text:
        print("✗ Entity extraction skipped (empty text)")
        return []

    nlp = get_nlp()
    
    if nlp == "unavailable":
        print("✗ Entity extraction skipped (spaCy not available)")
        return []
    
    try:
        # Truncate text if too long
        text_to_process = text[:10000]
        
        doc = nlp(text_to_process)
        
        entities = []
        seen = set()
        
        for ent in doc.ents:
            # Avoid duplicates
            key = (ent.text.lower(), ent.label_)
            if key not in seen:
                seen.add(key)
                entities.append({
                    "text": ent.text,
                    "label": ent.label_
                })
        
        print(f"✓ Extracted {len(entities)} entities")
        return entities
        
    except Exception as e:
        print(f"✗ Entity extraction error: {e}")
        return []