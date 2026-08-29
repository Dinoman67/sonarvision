import json
from typing import Dict, Any

def generate_json_report(analysis_data: Dict[str, Any]) -> str:
    """
    Serializes analysis response dictionary to pretty formatted JSON.
    """
    return json.dumps(analysis_data, indent=2, default=str)
