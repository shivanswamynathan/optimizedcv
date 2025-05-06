import os
import json
import re
import uuid
import logging
from pathlib import Path
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
TYPST_TEMPLATES_DIR = os.path.join(BASE_DIR, 'typst_templates')
SE_TYPST_TEMPLATE_FILE = os.path.join(TYPST_TEMPLATES_DIR, 'SE_typest.typ')
OUTPUT_DIR = os.path.join(BASE_DIR, 'enhanced_resumes')

def update_typst_template(json_data, typst_file_path=None):
    """
    Update the Typst template file with enhanced JSON data.
    
    Args:
        json_data (dict): Enhanced JSON resume data
        typst_file_path (str, optional): Path to the Typst template file. If None, uses default path.
        
    Returns:
        dict: Dictionary containing paths to the updated files
    """
    try:
        if typst_file_path is None:
            typst_file_path = SE_TYPST_TEMPLATE_FILE
            
        if not os.path.exists(typst_file_path):
            logger.error(f"Typst template not found at: {typst_file_path}")
            raise FileNotFoundError(f"Typst template not found at: {typst_file_path}")
            
        os.makedirs(OUTPUT_DIR, exist_ok=True)
            
        with open(typst_file_path, 'r', encoding='utf-8') as f:
            typst_content = f.read()
        
        typst_data = json_to_typst_dict(json_data)
        
        patterns = [
            r'#let data = \(\s*details:\s*\(.*?^\)\s*\)',  # Standard pattern
            r'#let data = \(.*?^\)',                      # More general pattern
            r'#let data = \([\s\S]*?\n\)',                # Very general pattern
        ]
        
        replaced = False
        for pattern in patterns:
            if re.search(pattern, typst_content, re.DOTALL | re.MULTILINE):
                updated_content = re.sub(pattern, f'#let data = {typst_data}', typst_content, 
                                         flags=re.DOTALL | re.MULTILINE)
                replaced = True
                break
        
        if not replaced:
            lines = typst_content.split('\n')
            data_start_idx = -1
            data_end_idx = -1
            
            for i, line in enumerate(lines):
                if '#let data =' in line:
                    data_start_idx = i
                    open_count = line.count('(')
                    close_count = line.count(')')
                    balance = open_count - close_count
                    
                    if balance == 0:
                        data_end_idx = i
                        break
                    
                    j = i + 1
                    while j < len(lines) and balance > 0:
                        open_count = lines[j].count('(')
                        close_count = lines[j].count(')')
                        balance += open_count - close_count
                        if balance == 0:
                            data_end_idx = j
                            break
                        j += 1
                    break
            
            if data_start_idx >= 0 and data_end_idx >= 0:
                lines[data_start_idx:data_end_idx+1] = [f'#let data = {typst_data}']
                updated_content = '\n'.join(lines)
                replaced = True
            
        if not replaced:
            logger.error("Could not find data declaration pattern in template")
            raise ValueError("Could not find data declaration pattern in template")
        
        # unique_id = str(uuid.uuid4())
        # output_dir = os.path.join(OUTPUT_DIR, unique_id)
        # os.makedirs(output_dir, exist_ok=True)
        
        updated_typst_path = os.path.join('resume.typ')
        with open(updated_typst_path, 'w', encoding='utf-8') as f:
            f.write(updated_content)

        # json_path = os.path.join(output_dir, 'resume.json')
        # with open(json_path, 'w', encoding='utf-8') as f:
        #     json.dump(json_data, f, indent=2)
        return {
            'typst_path': updated_typst_path
        }
    
    except Exception as e:
        logger.error(f"Error updating Typst template: {str(e)}")
        raise

def json_to_typst_dict(json_data):
    """
    Convert JSON data to Typst dictionary format.
    
    Args:
        json_data (dict): JSON resume data
        
    Returns:
        str: Typst formatted dictionary string
    """
    if 'details' in json_data:
        data = json_data['details']
    else:
        data = json_data
    
    result = "(\n  details:   (\n"
    
    # Process each section of the resume
    for section_name, section_data in data.items():
        result += f"    {section_name}: "
        result += _convert_value_to_typst(section_data, indent_level=4)
        result += ",\n"
    
    result += "  )\n)"
    return result

def _convert_value_to_typst(value, indent_level=0):
    """
    Recursively convert JSON values to Typst format.
    
    Args:
        value: The value to convert
        indent_level (int): Current indentation level
        
    Returns:
        str: Typst formatted value
    """
    indent = "  " * indent_level
    
    if isinstance(value, dict):
        if not value:  # Empty dict
            return "()"
            
        result = "(\n"
        for k, v in value.items():
            if v is None:  # Skip None values
                continue
            result += f"{indent}  {k}: {_convert_value_to_typst(v, indent_level + 1)},\n"
        result += f"{indent})"
        return result
    
    elif isinstance(value, list):
        if not value:  # Empty list
            return "()"
        
        result = "(\n"
        for item in value:
            if item is None:  # Skip None values
                continue
            result += f"{indent}  {_convert_value_to_typst(item, indent_level + 1)},\n"
        result += f"{indent})"
        return result
    
    elif isinstance(value, str):
        # Escape quotes and backslashes
        escaped = value.replace("\\", "\\\\").replace("\"", "\\\"")
        return f"\"{escaped}\""
    
    elif value is None:
        return "\"\""
    
    else:
        return str(value)