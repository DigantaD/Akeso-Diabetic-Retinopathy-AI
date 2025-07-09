import re

def normalize_id(image_id):
    """
    Extracts the numeric ID from any image identifier (e.g., 'IDRiD_041', 'IDRiD_41', '41')
    and returns it as an integer.
    
    Parameters:
        image_id (str): The image identifier string.

    Returns:
        int: The normalized integer ID (e.g., 41).
    """
    match = re.search(r'\d+', image_id)
    if not match:
        raise ValueError(f"Invalid image ID format: {image_id}")
    return int(match.group())