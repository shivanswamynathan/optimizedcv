from typing import Dict, Any, Tuple, Optional
import io
import json
import logging
from langchain_core.language_models.base import BaseLanguageModel
from langchain_google_genai import GoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_deepseek import ChatDeepSeek
from PyPDF2 import PdfReader
from .llm_logger import LLMLogger
from .modelmanager import SimpleModelManager
from dotenv import load_dotenv
import os

load_dotenv()
llm_logger = LLMLogger()
logger = logging.getLogger(__name__)

def clean_text(text: str) -> str:
    """Clean and normalize extracted text"""
    return " ".join(text.split()).replace('\x00', '')

def extract_text_and_hyperlinks(pdf_file) -> Tuple[str, list]:
    """
    Extracts text and hyperlinks from a PDF file.
    Returns tuple of (text_content, hyperlinks)
    """
    reader = PdfReader(pdf_file)
    text_content = ""
    hyperlinks = []

    for page_num, page in enumerate(reader.pages):
        text_content += page.extract_text() or ""
        if "/Annots" in page:
            for annot in page["/Annots"]:
                annot_obj = annot.get_object()
                if "/A" in annot_obj and "/URI" in annot_obj["/A"]:
                    hyperlinks.append({
                        "page": page_num + 1,
                        "url": annot_obj["/A"]["/URI"]
                    })
    print(text_content)
    return text_content, hyperlinks

import json

def create_extraction_prompt(resume_text: str, hyperlinks: list) -> str:
    """Create the prompt for resume information extraction"""
    
    schema_template = {
        "userId": None,
        "profileImage": None,
        "firstName": "",
        "lastName": "",
        "dob": None,
        "email": "",
        "address": "",
        "city": "",
        "state": "",
        "country": "",
        "zipcode": "",
        "phone": "",
        "currentTitle": "",
        "industry": "",
        "summary": "",
        "websites": [
            {
                "url": "",
                "type": ""
            }
        ],
        "languages": [
            {
                "name": "",
                "proficiency": "",
                "skills": []
            }
        ],
        "projects": [
            {
                "projectName": "",
                "projectUrl": "",
                "projectDescription": []
            }
        ],
        "experience": [
            {
                "companyName": "",
                "jobTitle": "",
                "skills": [],
                "startDate": "",
                "endDate": "",
                "currentlyWorking": False,
                "responsibilities": []
            }
        ],
        "education": [
            {
                "level": "",
                "university": "",
                "major": "",
                "specialization": "",
                "startDate": "",
                "endDate": "",
                "score": [
                    {
                        "type": "",
                        "value": ""
                    }
                ],
                "description": []
            }
        ],
        "certifications": [
            {
                "name": "",
                "completionId": "",
                "url": "",
                "startDate": "",
                "endDate": ""
            }
        ],
        "publications": [
            {
                "author": "",
                "description": [],
                "publishedDate": "",
                "publisher": "",
                "publisherUrl": "",
                "title": ""
            }
        ]
    }

    return f"""Extract and structure resume information from the provided text into JSON format.
Below is the resume content:
{resume_text}

Hyperlinks Extracted:
{json.dumps(hyperlinks, indent=2)}

Follow this exact structure:
{json.dumps(schema_template, indent=2)}

### Instructions:
- Extract ALL available information present in the resume, excluding dashes like -,–.
- If the fields can't be filled with the extracted information, mark them as null.
- Strictly adhere to the provided JSON structure for all extracted data.
- All extracted fields must map directly to the keys and nesting provided in the schema.
- Do not add or remove keys, even if no data is found (keep empty lists or nulls where applicable, matching the schema format).
- Extract personal details including firstName, lastName, dob, email, address, city, state, country, zipcode, phone, currentTitle, and industry exactly as found.
- If available, extract a professional summary or objective statement into the "summary" field.
- Format all dates strictly as 'YYYY-MM-DDTHH:MM:SS.000Z', defaulting to January 1st if only year is provided.
- For education, map each qualification to the 'education' array, ensuring correct nesting with level, university, major, specialization, startDate, endDate, gpa, and description.
- Standardize education titles (e.g., Bachelor, Master variations should be converted to: 'BE', 'BS', 'MS', 'B.Tech', 'MBA', etc.) and ensure they are placed under the 'level' field.
- Ensure all phone numbers are formatted in international format: '+[Country Code] [Number]'.
- For websites, categorize them by type (e.g., "LinkedIn", "Portfolio", "GitHub", "Personal") and include the full URL.
- Projects must contain projectName, projectUrl, and projectDescription — break down compound information into these fields wherever possible.
- Experience must contain companyName, jobTitle, skills, startDate, endDate, currentlyWorking, and responsibilities — each experience entry must map to this exact structure.
- Extract technical keywords as skills and group similar ones under appropriate categories (experience.skills).
- Languages must contain name, proficiency, and skills (reading, writing, speaking, etc.), matching the schema's nested format.
- Certifications must contain name, completionId, url, startDate, and endDate — ensure these fields match the schema.
- Publications must contain author, description, publishedDate, publisher, publisherUrl, and title — follow the exact schema structure.
- Preserve original text and spelling for important fields such as project descriptions, responsibilities, and publication titles, unless standardization is explicitly required (dates, phone numbers, education levels).
- All date fields must follow the exact ISO 8601 format provided in the schema ('2025-02-02T18:30:00.000Z').
- All arrays and objects must strictly match the schema in type (lists, strings, objects), even if no data is available (use empty arrays where necessary).
- Ensure all fields required by the schema are present, even if they are null or empty.
- Leave userId and profileImage as null values - these will be populated later.
- Return only valid JSON that matches the schema exactly — do not include extra commentary, headers, or explanations.
"""

def remove_null_values(obj):
    """Replace null values with empty strings or arrays"""
    if isinstance(obj, dict):
        return {k: remove_null_values(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [remove_null_values(item) for item in obj]
    return "" if obj is None else obj

def get_model_name(model: BaseLanguageModel) -> str:
    """Get the standardized name for the model type"""
    model_mapping = {
        GoogleGenerativeAI: "gemini",
        ChatOpenAI: "openai",
        ChatDeepSeek: "deepseek"
    }
    return model_mapping.get(type(model), "unknown")

def extract_response_text(response: Any, model: BaseLanguageModel) -> str:
    """Extract text content from model response based on model type"""
    if isinstance(model, GoogleGenerativeAI):
        return response.text if hasattr(response, 'text') else str(response)
    elif isinstance(model, (ChatOpenAI, ChatDeepSeek)):
        return response.content if hasattr(response, 'content') else str(response)
    return str(response)

def save_input_json(json_data: Dict[str, Any], filename: str = "resume1input.json") -> None:
    """Save the input JSON to the specified file"""
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        input_dir = os.path.join(base_dir, "tests", "assets", "input_json")
        os.makedirs(input_dir, exist_ok=True)
        
        input_path = os.path.join(input_dir, filename)
        with open(input_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving input JSON: {str(e)}")

async def convert_pdf_to_json_schema(
    pdf_content: bytes,
    save_input: bool = False
) -> Dict[str, Any]:
    """
    Convert PDF content to structured JSON schema.
    
    Args:
        pdf_content: Raw PDF file content in bytes
        save_input: Whether to save the extracted JSON to a file
        model: Optional Modelmanager instance.
    
    Returns:
        Dict containing the structured resume data or error message
    """
    try:
        
        instance = SimpleModelManager()
        model = instance.get_model()
        
        resume_text, hyperlinks = extract_text_and_hyperlinks(io.BytesIO(pdf_content))
        if not resume_text.strip():
            return {"error": "No text could be extracted from the PDF"}

        prompt = create_extraction_prompt(resume_text, hyperlinks)
        
        try:
            # Use the properly initialized model
            result = await model.ainvoke(prompt)
            logger.info(f"Extraction result: {result.response_metadata.get('token_usage').get('completion_tokens')}")
            
            response_text = extract_response_text(result, model)
            
            cleaned_result = response_text.replace('```json', '').replace('```', '').strip()
            
            try:
                response_json = json.loads(cleaned_result)
                cleaned_json = remove_null_values(response_json)

                # Validation for schema 
                if not isinstance(cleaned_json, dict) or \
                        "firstName" not in cleaned_json or \
                        "lastName" not in cleaned_json or \
                        "email" not in cleaned_json:
                    error_msg = "Invalid JSON structure - missing required fields (firstName, lastName, email)"
                    llm_logger.log_interaction(
                        model_name=instance.current_model_type,
                        input_text=prompt,
                        output_text=cleaned_result,
                        metadata={"error": error_msg, "status": "failed"}
                    )
                    return {"error": error_msg}

                # Log success
                llm_logger.log_interaction(
                    model_name=instance.current_model_type,
                    input_text=prompt,
                    output_text=json.dumps(cleaned_json, indent=2),
                    metadata={"status": "success"}
                )

                if save_input:
                    save_input_json(cleaned_json)

                return cleaned_json

            except json.JSONDecodeError as e:
                llm_logger.log_interaction(
                    model_name=instance.current_model_type,
                    input_text=prompt,
                    output_text=cleaned_result,
                    metadata={"error": str(e), "status": "failed"}
                )
                return {"error": f"Invalid JSON response: {str(e)}"}
            
        except Exception as e:
            logger.error(f"PDF processing error: {str(e)}")
            return {"error": f"Extraction failed: {str(e)}"}
    except Exception as e:
        logger.error(f"PDF processing error: {str(e)}")
        return {"error": f"Extraction failed: {str(e)}"}