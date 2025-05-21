import json
import logging
import re
from typing import Dict, Any, List, Optional, Union, Tuple
from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.output_parsers import PydanticOutputParser
from .modelmanager import SimpleModelManager
from .enhance import enhance_resume_section
from ..models.resume_models import (
    SECTION_MODELS, SECTION_GUIDELINES, SECTION_NAME_MAPPING
)

logger = logging.getLogger(__name__)

# Define formatters for each section type
SUMMARY_FORMATTERS = {
    "work": lambda item: f"{item.get('position', 'Role')} at {item.get('name', 'Company')}",
    "education": lambda item: f"{item.get('degree', 'Degree')} in {item.get('area', 'Field')} at {item.get('institution', 'Institution')}",
    "projects": lambda item: f"{item.get('name', 'Project')}",
    "skills": lambda item: f"{item.get('name', 'Skill Category')}",
    "certifications": lambda item: f"{item.get('certificateName', 'Certificate')} from {item.get('completionId', 'Unknown ID')}",
    "publications": lambda item: f"\"{item.get('title', 'Publication')}\" published by {item.get('publisher', 'Publisher')}",
    "languages": lambda item: f"{item.get('name', 'Language')} - {item.get('proficiency', 'Proficiency level')}",
    "awards": lambda item: f"{item.get('title', 'Award')} from {item.get('awarder', 'Organization')}"
}

def get_section_prompt(section_name: str, parser: Optional[PydanticOutputParser] = None) -> str:
    """Create a prompt for a specific section with correct formatting instructions."""
    
    section_format = SECTION_GUIDELINES.get(
        "experience" if section_name == "work" else section_name, 
        "Create a well-structured entry with all required information."
    )
    
    # Special handling for list types
    is_list_wrapper = section_name in [
        "work", "education", "skills", "projects", 
        "awards", "publications", "languages", "certifications"
    ]
    
    format_instructions = parser.get_format_instructions() if parser else ""
    
    return f"""
    FORMAT GUIDELINES:
    {section_format}
    
    {format_instructions}
    
    INSTRUCTIONS:
    1. Extract all information mentioned in the query
    2. Format dates consistently as "YYYY-MM" (e.g., "2020-01")
    3. For current positions, use "present" for the endDate
    4. Do not invent information not present in the query
    5. For lists (like highlights, courses, keywords), format as a proper JSON array
    6. Every required field must have a value - use reasonable defaults if necessary
    
    IMPORTANT: Return ONLY valid JSON. 
    {f"Response must be a JSON object with an 'items' list." if is_list_wrapper else ""}
    Do not include explanations or extra text.
    """

def create_summarized_entries(section_name: str, items: List[Dict[str, Any]]) -> List[str]:
    """Create human-readable summaries of resume entries."""
    entries_summary = []
    
    # Default formatter for section types not in the dictionary
    def default_formatter(item):
        return list(item.values())[0] if item and item.values() else 'Entry'
    
    for idx, item in enumerate(items):
        # Get the appropriate formatter for this section type, or use default
        formatter = SUMMARY_FORMATTERS.get(section_name, default_formatter)
        
        # Format the summary with the entry number
        summary = f"{idx+1}. {formatter(item)}"
        entries_summary.append(summary)
    
    return entries_summary

def extract_response_text(response: Any) -> str:
    """Extract text content from a model response."""
    if hasattr(response, 'content'):
        return response.content
    elif hasattr(response, 'text'):
        return response.text
    else:
        return str(response)

async def process_resume_query(
    json_data: Dict[str, Any],
    user_query: str,
    job_description: Optional[str] = None
) -> Dict[str, Any]:
    """
    Process a user's query to modify their resume and return the updated resume.
    This is the main entry point that uses helper functions and manages model access.
    
    Args:
        json_data: The resume data in JSON format
        user_query: The user's natural language query about what to add/modify
        job_description: Optional job description for context
        
    Returns:
        Dict: Updated resume JSON with the user's requested modifications
    """
    try:
        # Initialize model once for use across all functions
        model_manager = SimpleModelManager()
        model = model_manager.get_model()
        
        # Log the incoming request
        logger.info(f"Processing resume query: {user_query[:50]}...")
        
        # Determine whether we're working with the inner details or the full object
        if 'details' in json_data and isinstance(json_data['details'], dict):
            resume_data = json_data['details']
            has_details_wrapper = True
        else:
            resume_data = json_data
            has_details_wrapper = False
        
        # 1. Identify which section the user wants to modify using LLM
        # Create prompt for section identification
        section_prompt = f"""
        Determine which section of a resume the following query is referring to.
        The query may be about adding or modifying content in a specific section.
        
        Valid resume sections are: 
        - work (job experience, employment history, professional experience)
        - education (schooling, degrees, academic background)
        - skills (abilities, competencies, technical skills)
        - projects (personal projects, side projects)
        - awards (honors, achievements, recognitions)
        - languages (spoken or programming languages)
        - publications (papers, articles, research)
        - certifications (certificates, licenses, credentials)
        
        User query: "{user_query}"
        
        Analyze the query and determine which section it most likely refers to.
        Only respond with one of these exact section names: work, education, skills, projects, awards, languages, publications, certifications.
        Do not provide any explanation, just output the section name.
        """
        
        try:
            # Call the model for section identification
            response = await model.ainvoke(section_prompt)
            section_result = extract_response_text(response).strip().lower()
            
            # Clean up the response to get just the section name
            section_result = section_result.replace('"', '').replace("'", "").strip()
            
            # Validate against known sections
            valid_sections = [
                "work", "education", "skills", "projects", 
                "awards", "languages", "publications", "certifications"
            ]
            
            if section_result in valid_sections:
                target_section = section_result
                logger.info(f"Dynamically identified section: {target_section} from query: {user_query[:50]}...")
            else:
                # Fall back to keyword matching
                logger.warning(f"LLM returned invalid section: {section_result}. Falling back to keyword matching.")
                
                # Direct keyword matching implementation
                default_section = "work"
                query_lower = user_query.lower()
                
                for keyword, section in SECTION_NAME_MAPPING.items():
                    if keyword in query_lower:
                        target_section = section
                        logger.info(f"Identified section via keywords: {section} (keyword: {keyword})")
                        break
                else:  # This else belongs to the for loop (executes if no break occurs)
                    target_section = default_section
                    logger.info(f"No section keywords found in query. Using default: {default_section}")
                    
        except Exception as e:
            logger.error(f"Error in section identification: {str(e)}")
            
            # Direct keyword matching as fallback
            default_section = "work"
            query_lower = user_query.lower()
            
            target_section = default_section
            for keyword, section in SECTION_NAME_MAPPING.items():
                if keyword in query_lower:
                    target_section = section
                    logger.info(f"Identified section via keywords (fallback): {section} (keyword: {keyword})")
                    break
            else:
                logger.info(f"No section keywords found in query (fallback). Using default: {default_section}")
        
        # 2. Get the current section data
        if target_section not in resume_data:
            resume_data[target_section] = []
        
        section_data = resume_data[target_section]
        
        # 3. Determine if adding new or modifying existing
        # Default to adding a new entry
        action = "add"
        entry_index = None

        # Check if there are existing entries to potentially modify
        items = []
        if isinstance(section_data, list):
            items = section_data
        elif isinstance(section_data, dict) and 'items' in section_data:
            items = section_data['items']

        # If there are existing items, use LLM to determine intent
        if items:
            # Create summaries of existing entries
            entries_summary = create_summarized_entries(target_section, items)
            
            # Create prompt to determine action and entry
            action_prompt = f"""
            Based on the user's query, determine whether they want to ADD a new entry to their resume's {target_section} section or MODIFY an existing entry.
            
            USER QUERY:
            "{user_query}"
            
            EXISTING ENTRIES in {target_section} section:
            {chr(10).join(entries_summary)}
            
            First, determine if the user is more likely trying to:
            1. ADD a completely new entry
            2. MODIFY an existing entry
            
            If you think they're trying to MODIFY an existing entry, identify which entry number (1, 2, 3, etc.) they most likely want to modify.
            
            If you think they're trying to ADD a new entry, or it's unclear which entry they want to modify, respond with "new".
            
            Your response should be ONLY one of:
            - A single number (e.g., "1", "2", "3") representing the entry to modify
            - The word "new" if they want to add a new entry
            
            Respond with just this one word or number, no explanation.
            """
            
            try:
                # Call the model to determine action
                response = await model.ainvoke(action_prompt)
                result = extract_response_text(response).strip().lower()
                
                # Process the result
                if result.isdigit() and 1 <= int(result) <= len(items):
                    action = "modify"
                    entry_index = int(result) - 1  # Convert to 0-based index
                    logger.info(f"Identified intent to modify entry {entry_index+1} in {target_section} section")
                else:
                    action = "add"
                    logger.info(f"Identified intent to add new entry to {target_section} section")
            except Exception as e:
                logger.error(f"Error determining action: {str(e)}")
                # Default to adding on error
                action = "add"
                entry_index = None
        else:
            # If no existing entries, must be adding a new one
            action = "add"
            logger.info(f"No existing entries in {target_section} section. Must be adding new entry.")
        
        # 4. Extract structured content from the query
        # Get the appropriate schema model for this section
        section_model = SECTION_MODELS.get(target_section)
        
        if not section_model:
            logger.warning(f"No schema defined for section '{target_section}', using generic extraction")
            content_prompt = f"""
            Extract information from the user's query to create a new entry for the {target_section} section of a resume.
            
            USER QUERY:
            "{user_query}"
            
            Extract all relevant information and return a single JSON object.
            Only return valid JSON that can be directly parsed, no explanations or text outside the JSON.
            """
            
            # No parser for sections without a model
            parser = None
        else:
            # Create Pydantic parser
            parser = PydanticOutputParser(pydantic_object=section_model)
            
            # Get section prompt with formatting instructions
            section_instructions = get_section_prompt(target_section, parser)
            
            content_prompt = f"""
            Extract information from the user's query to create a new entry for the {target_section} section of a resume.
            
            USER QUERY:
            "{user_query}"
            
            SECTION: {target_section}
            
            {section_instructions}
            """
        
        try:
            # Call model to extract content
            response = await model.ainvoke(content_prompt)
            response_text = extract_response_text(response)
            
            logger.debug(f"Raw response from model: {response_text[:500]}...")
            
            # Parse the content - following pattern from enhance.py
            extracted_content = {}  # Default empty content
            
            try:
                if parser:
                    # Use Pydantic parser if available
                    parsed_data = parser.parse(response_text)
                    
                    # Convert to dict if it's a Pydantic model
                    if hasattr(parsed_data, "model_dump"):
                        parsed_dict = parsed_data.model_dump()
                        
                        # Special handling for list types as in enhance.py
                        if target_section in ["work", "education", "skills", "projects", "awards", "publications", "languages", "certifications"]:
                            if "items" in parsed_dict:
                                # When extracting from query, we typically want just the first item
                                if parsed_dict["items"] and len(parsed_dict["items"]) > 0:
                                    extracted_content = parsed_dict["items"][0]
                                else:
                                    logger.warning(f"No items found in parsed response for {target_section}")
                            else:
                                logger.warning(f"Expected 'items' field in {target_section} response, but not found")
                                extracted_content = parsed_dict
                        else:
                            extracted_content = parsed_dict
                    else:
                        logger.warning(f"Unexpected parsed data type for {target_section}: {type(parsed_data)}")
                        extracted_content = parsed_data
                else:
                    # Fallback to JSON parsing if no parser (similar to enhance.py)
                    match = re.search(r'(\{.*\}|\[.*\])', response_text, re.DOTALL)
                    if match:
                        json_str = match.group(0)
                        extracted_content = json.loads(json_str)
            except Exception as e:
                logger.error(f"Failed to parse response for section {target_section}: {str(e)}")
                logger.debug(f"Response text: {response_text[:500]}...")
                
                # Last resort fallback parsing
                try:
                    # Try to extract JSON from code blocks
                    json_match = re.search(r'```json\s*([\s\S]*?)\s*```', response_text)
                    if json_match:
                        extracted_content = json.loads(json_match.group(1).strip())
                    else:
                        # Try to extract JSON without code blocks
                        json_match = re.search(r'(\{[\s\S]*\}|\[[\s\S]*\])', response_text.strip(), re.DOTALL)
                        if json_match:
                            extracted_content = json.loads(json_match.group(1).strip())
                        else:
                            raise ValueError("Could not extract valid JSON from model response")
                except Exception as nested_e:
                    logger.error(f"All parsing methods failed: {str(nested_e)}")
                    return {
                        "error": f"Failed to parse content from query: {str(e)} -> {str(nested_e)}",
                        "data": json_data,
                        "success": False
                    }
            
            # Once we have the extracted content, enhance it using the enhance_resume_section function
            try:
                # Prepare extracted content for enhancement
                # If this is a list wrapper section, we need to wrap it in an items list
                is_list_wrapper = target_section in [
                    "work", "education", "skills", "projects", 
                    "awards", "publications", "languages", "certifications"
                ]
                
                content_to_enhance = extracted_content
                if is_list_wrapper:
                    content_to_enhance = {"items": [extracted_content]}
                
                # Call enhance_resume_section to improve the extracted content
                enhanced_result = await enhance_resume_section(
                    section_name=target_section,
                    section_data=content_to_enhance,
                    model=model,
                    job_description=job_description
                )
                
                # Extract the enhanced content
                if "section_data" in enhanced_result:
                    if is_list_wrapper:
                        # For list wrappers, get the first item from the enhanced items
                        if isinstance(enhanced_result["section_data"], list):
                            new_content = enhanced_result["section_data"][0] if enhanced_result["section_data"] else extracted_content
                        elif isinstance(enhanced_result["section_data"], dict) and "items" in enhanced_result["section_data"]:
                            items = enhanced_result["section_data"]["items"]
                            new_content = items[0] if items else extracted_content
                        else:
                            new_content = enhanced_result["section_data"]
                    else:
                        new_content = enhanced_result["section_data"]
                    
                    logger.info(f"Successfully enhanced extracted content for {target_section} section")
                else:
                    # If enhancement fails, use the original extracted content
                    new_content = extracted_content
                    logger.warning(f"Enhancement failed, using original extracted content for {target_section} section")
            except Exception as e:
                # If enhancement fails, use the original extracted content
                new_content = extracted_content
                logger.error(f"Error enhancing extracted content: {str(e)}. Using original content.")
            
        except Exception as e:
            logger.error(f"Error extracting content from query: {str(e)}")
            return {
                "error": f"Failed to extract content: {str(e)}",
                "data": json_data,
                "success": False
            }
        
        # 5. Update the resume data based on the action
        if action == "add":
            logger.info(f"Adding new entry to {target_section} section")
            
            # If section doesn't exist yet, create it
            if target_section not in resume_data:
                resume_data[target_section] = []
            
            # Add the new entry
            if isinstance(resume_data[target_section], list):
                resume_data[target_section].append(new_content)
            else:
                # Handle case where it might be a dict with 'items' key
                if 'items' in resume_data[target_section]:
                    resume_data[target_section]['items'].append(new_content)
                else:
                    # Convert to list if it's not already
                    resume_data[target_section] = [new_content]
                    
        elif action == "modify" and entry_index is not None:
            logger.info(f"Modifying entry {entry_index} in {target_section} section")
            
            # Modify existing entry
            if isinstance(resume_data[target_section], list):
                if 0 <= entry_index < len(resume_data[target_section]):
                    resume_data[target_section][entry_index] = new_content
                else:
                    raise ValueError(f"Invalid entry index: {entry_index}")
            else:
                # Handle case where it might be a dict with 'items' key
                if 'items' in resume_data[target_section]:
                    if 0 <= entry_index < len(resume_data[target_section]['items']):
                        resume_data[target_section]['items'][entry_index] = new_content
                    else:
                        raise ValueError(f"Invalid entry index: {entry_index}")
        
        # Return the updated resume
        if has_details_wrapper:
            json_data['details'] = resume_data
            result = json_data
        else:
            result = resume_data
            
        return {
            "message": "Resume updated successfully based on your query",
            "data": result,
            "success": True
        }
        
    except Exception as e:
        logger.error(f"Error in process_resume_query: {str(e)}")
        return {
            "error": f"Query processing failed: {str(e)}",
            "data": json_data,
            "success": False
        }