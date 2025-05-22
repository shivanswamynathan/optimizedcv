import json
import logging
import re
from typing import Dict, Any, List, Optional, Union, Tuple, Literal
from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from .modelmanager import SimpleModelManager
from .enhance import enhance_resume_section
from ..models.resume_models import (SpecificChanges,ModificationRequest,QueryAnalysis,
    SECTION_MODELS
)

logger = logging.getLogger(__name__)

def extract_response_text(response: Any) -> str:
    """Extract text content from a model response.
        
    Key Features:
        - Handles different LLM response formats automatically
        - Supports OpenAI/DeepSeek (content attribute) and Gemini (text attribute)
        - Provides fallback conversion to string for unknown formats
        - Returns clean text content ready for processing
        - Used across all model interactions for consistency
    """
    if hasattr(response, 'content'):
        return response.content
    elif hasattr(response, 'text'):
        return response.text
    else:
        return str(response)

async def analyze_complex_query(user_query: str, model: BaseLanguageModel) -> QueryAnalysis:
    """
    Analyze a complex user query to identify all the changes they want to make.
    
    Args:
        user_query: Natural language query from user about resume modifications
        model: Language model instance for query processing
        
    Returns:
        QueryAnalysis: Pydantic model containing structured list of modifications
        
    Key Features:
        - Uses Pydantic parser for structured output validation
        - Identifies multiple modification requests in single query
        - Categorizes actions (modify_existing, add_new, enhance_content)
        - Maps requests to specific resume sections (basics, work, education, etc.)
        - Provides fallback analysis if parsing fails for robust operation
    """
    parser = PydanticOutputParser(pydantic_object=QueryAnalysis)
    
    prompt_template = PromptTemplate(
        template="""
        Analyze the following user query about resume modifications and break it down into specific, actionable tasks.
        
        USER QUERY:
        "{user_query}"
        
        Identify ALL the different modifications the user is requesting. For each modification, determine:
        1. Which resume section it affects (basics, work, education, skills, projects, awards, languages, publications, certifications)
        2. What type of action (modify_existing, add_new, enhance_content, delete_item)
        3. Specific details about what needs to be changed
        
        For specific_changes, use these guidelines:
        - field: The specific field to modify (phone, summary, name, position, title, etc.)
        - identifier: Use for finding specific entries (company name, school name, project name, etc.)
        - search_term: Any term that helps identify what to modify
        - enhancement_focus: What aspect to enhance (technical, leadership, impact, metrics, etc.)
        - data: New data to add for any section type
        
        
        Be thorough and identify every single modification request in the query.
        
        {format_instructions}
        """,
        input_variables=["user_query"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    
    try:
        formatted_prompt = prompt_template.format(user_query=user_query)
        response = await model.ainvoke(formatted_prompt)
        response_text = extract_response_text(response).strip()
        
        logger.debug(f"Query analysis response: {response_text[:500]}...")
        
        analysis = parser.parse(response_text)
        logger.info(f"Successfully analyzed query into {len(analysis.modifications)} modifications")
        return analysis
        
    except Exception as e:
        logger.error(f"Error analyzing complex query: {str(e)}")
        # Fallback analysis
        fallback_modification = ModificationRequest(
            section="work",
            action="enhance_content", 
            description=user_query[:100],
            specific_changes=SpecificChanges()
        )
        return QueryAnalysis(modifications=[fallback_modification])

async def modify_section_generic(
    section_name: str,
    section_data: Any,
    modifications: List[ModificationRequest],
    model: BaseLanguageModel,
    job_description: Optional[str] = None
) -> Any:
    """
    Generic function to handle modifications for ANY section using existing models from SECTION_MODELS.
    
    Args:
        section_name: Name of resume section (basics, work, education, skills, etc.)
        section_data: Current data for the section
        modifications: List of modification requests for this section
        model: Language model instance for enhancements
        job_description: Optional job description for targeted improvements
        
    Returns:
        Any: Modified section data in appropriate format (list or dict)
        
    Key Features:
        - Handles all resume sections uniformly using existing SECTION_MODELS
        - Distinguishes between list-type sections (work, education) and object sections (basics)
        - Processes multiple modifications per section in sequence
        - Leverages existing Pydantic models for type safety and validation
        - Replaces need for separate section-specific functions
    """
    section_model = SECTION_MODELS.get(section_name)
    if not section_model:
        logger.warning(f"No existing model found for section: {section_name}")
        return section_data
    
    relevant_mods = [mod for mod in modifications if mod.section == section_name]
    if not relevant_mods:
        return section_data
    
    logger.info(f"Processing {len(relevant_mods)} modifications for {section_name} section")

    is_list_section = section_name in ["work", "education", "skills", "projects", "awards", "publications", "languages", "certifications"]
    
    if is_list_section:
        # For list-type sections
        modified_data = section_data.copy() if isinstance(section_data, list) else []
        
        for mod in relevant_mods:
            modified_data = await handle_list_section_modification(
                section_name, modified_data, mod, model, job_description
            )
        
        return modified_data
    else:
        # For non-list sections (like basics)
        modified_data = section_data.copy() if isinstance(section_data, dict) else {}
        
        for mod in relevant_mods:
            modified_data = await handle_object_section_modification(
                section_name, modified_data, mod, model, job_description
            )
        
        return modified_data

async def handle_list_section_modification(
    section_name: str,
    section_data: List[Dict[str, Any]],
    modification: ModificationRequest,
    model: BaseLanguageModel,
    job_description: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Handle modifications for list-type sections (work, education, skills, etc.).
    
    Args:
        section_name: Name of the list-type section to modify
        section_data: Current list of items in the section
        modification: Single modification request to process
        model: Language model for content generation and enhancement
        job_description: Optional JD for context-aware modifications
        
    Returns:
        List[Dict[str, Any]]: Updated list of section items
        
    Key Features:
        - Supports add_new, modify_existing, and enhance_content actions
        - Smart item identification using section-specific fields (name, institution, title)
        - Position-aware insertion for new items (beginning, end, or specific index)
        - Uses existing enhance_resume_section function for consistency
        - Handles all list-type sections with unified logic
    """
    
    if modification.action == "add_new":
        # Add new item to the section
        new_item = await create_new_section_item(
            section_name, modification, model, job_description
        )
        if new_item:
            position = modification.specific_changes.position if modification.specific_changes else None
            if position is not None:
                if position == -1:
                    section_data.append(new_item)
                else:
                    section_data.insert(position, new_item)
            else:
                section_data.append(new_item)
            
            logger.info(f"Added new item to {section_name} section")
    
    elif modification.action == "modify_existing":
        if modification.specific_changes and modification.specific_changes.identifier:
            identifier = modification.specific_changes.identifier.upper()

            for i, item in enumerate(section_data):
                item_match = False

                if section_name == "work":
                    item_match = item.get("name", "").upper() == identifier
                elif section_name == "education":
                    item_match = item.get("institution", "").upper() == identifier
                elif section_name == "projects":
                    item_match = item.get("name", "").upper() == identifier or item.get("projectName", "").upper() == identifier
                elif section_name == "skills":
                    item_match = item.get("name", "").upper() == identifier
                elif section_name == "certifications":
                    item_match = item.get("certificateName", "").upper() == identifier
                elif section_name == "publications":
                    item_match = item.get("title", "").upper() == identifier
                elif section_name == "awards":
                    item_match = item.get("title", "").upper() == identifier
                elif section_name == "languages":
                    item_match = item.get("name", "").upper() == identifier
                
                if item_match:
                    enhanced_item = await enhance_specific_item(
                        section_name, item, modification, model, job_description
                    )
                    section_data[i] = enhanced_item
                    logger.info(f"Modified existing item in {section_name} section: {identifier}")
                    break
    
    elif modification.action == "enhance_content":
        try:
            enhancement_result = await enhance_resume_section(
                section_name=section_name,
                section_data={"items": section_data},
                model=model,
                job_description=job_description
            )
            
            if "section_data" in enhancement_result and enhancement_result["section_data"]:
                enhanced_data = enhancement_result["section_data"]
                if isinstance(enhanced_data, list):
                    section_data = enhanced_data
                logger.info(f"Enhanced entire {section_name} section")
        except Exception as e:
            logger.error(f"Error enhancing {section_name} section: {str(e)}")
    
    return section_data

async def handle_object_section_modification(
    section_name: str,
    section_data: Dict[str, Any],
    modification: ModificationRequest,
    model: BaseLanguageModel,
    job_description: Optional[str] = None
) -> Dict[str, Any]:
    """
    Handle modifications for object-type sections (like basics).
    
    Args:
        section_name: Name of the object-type section (typically 'basics')
        section_data: Current dictionary data for the section
        modification: Single modification request to process
        model: Language model for content enhancement and generation
        job_description: Optional JD for targeted content improvements
        
    Returns:
        Dict[str, Any]: Updated section data dictionary
        
    Key Features:
        - Handles direct field modifications (phone, email, summary updates)
        - Supports adding new data like social profiles with proper structure
        - Uses existing SECTION_MODELS for type-safe content enhancement
        - Manages special cases like profiles list initialization and conversion
        - Provides fallback for old_value/new_value replacement operations
    """
    
    if modification.specific_changes:
        field = modification.specific_changes.field

        if field and modification.specific_changes.old_value and modification.specific_changes.new_value:
            old_value = modification.specific_changes.old_value
            new_value = modification.specific_changes.new_value
            
            if field in section_data:
                section_data[field] = str(section_data[field]).replace(old_value, new_value)
                logger.info(f"Updated {field} in {section_name}: {old_value} -> {new_value}")

        elif field and modification.specific_changes.data:
            if field == "profiles":
                if field not in section_data:
                    section_data[field] = []
                elif not isinstance(section_data[field], list):
                    existing = section_data[field]
                    section_data[field] = [existing] if existing else []
                
                section_data[field].append(modification.specific_changes.data)
                logger.info(f"Added new profile to {section_name}")
            else:
                section_data[field] = modification.specific_changes.data
                logger.info(f"Added new data to {field} in {section_name}")
    
    # Handle content enhancement
    if modification.action == "enhance_content":
        try:
            section_model = SECTION_MODELS.get(section_name)
            if section_model:
                parser = PydanticOutputParser(pydantic_object=section_model)
                
                enhancement_prompt = PromptTemplate(
                    template="""
                    Enhance this {section_name} section based on the modification request.
                    
                    CURRENT SECTION DATA:
                    {section_data}
                    
                    MODIFICATION REQUEST:
                    {modification_description}
                    
                    JOB DESCRIPTION (for context):
                    {job_description}
                    
                    ENHANCEMENT FOCUS:
                    {enhancement_focus}
                    
                    {format_instructions}
                    """,
                    input_variables=["section_name", "section_data", "modification_description", "job_description", "enhancement_focus"],
                    partial_variables={"format_instructions": parser.get_format_instructions()}
                )
                
                formatted_prompt = enhancement_prompt.format(
                    section_name=section_name,
                    section_data=json.dumps(section_data, indent=2),
                    modification_description=modification.description,
                    job_description=job_description or 'Not provided',
                    enhancement_focus=modification.specific_changes.enhancement_focus if modification.specific_changes else 'general improvement'
                )
                
                response = await model.ainvoke(formatted_prompt)
                response_text = extract_response_text(response)
                
                enhanced_obj = parser.parse(response_text)
                section_data = enhanced_obj.model_dump()
                logger.info(f"Enhanced {section_name} section using existing model")
                
        except Exception as e:
            logger.error(f"Error enhancing {section_name} section: {str(e)}")
    
    return section_data

async def create_new_section_item(
    section_name: str,
    modification: ModificationRequest,
    model: BaseLanguageModel,
    job_description: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """Create a new item for any section using the appropriate existing model.
    
    Args:
        section_name: Name of section for which to create new item
        modification: ModificationRequest containing creation details and data
        model: Language model for intelligent content generation
        job_description: Optional JD for role-aligned content creation
        
    Returns:
        Optional[Dict[str, Any]]: New item dictionary or None if creation fails
        
    Key Features:
        - Dynamically selects correct Pydantic model based on section type
        - Uses existing models (ExperienceItem, EducationItem, etc.) for consistency
        - Incorporates user-provided data with intelligent enhancement
        - Generates realistic content aligned with job requirements when JD provided
        - Handles all section types with unified creation logic
    """
    
    # Get the item model (not the list wrapper)
    if section_name == "work":
        from ..models.resume_models import ExperienceItem
        item_model = ExperienceItem
    elif section_name == "education":
        from ..models.resume_models import EducationItem
        item_model = EducationItem
    elif section_name == "projects":
        from ..models.resume_models import ProjectItem
        item_model = ProjectItem
    elif section_name == "skills":
        from ..models.resume_models import SkillCategory
        item_model = SkillCategory
    elif section_name == "certifications":
        from ..models.resume_models import Certification
        item_model = Certification
    elif section_name == "publications":
        from ..models.resume_models import Publication
        item_model = Publication
    elif section_name == "awards":
        from ..models.resume_models import Award
        item_model = Award
    elif section_name == "languages":
        from ..models.resume_models import Language
        item_model = Language
    else:
        logger.warning(f"No item model found for section: {section_name}")
        return None
    
    try:
        parser = PydanticOutputParser(pydantic_object=item_model)
        
        creation_prompt = PromptTemplate(
            template="""
            Create a new {section_name} entry based on the modification request.
            
            MODIFICATION REQUEST:
            {modification_description}
            
            PROVIDED DATA:
            {provided_data}
            
            JOB DESCRIPTION (for alignment):
            {job_description}
            
            REQUIREMENTS:
            - Create realistic and relevant content
            - Use appropriate dates in YYYY-MM format where needed
            - Include relevant details that align with the job description
            - Follow professional standards for this type of entry
            
            {format_instructions}
            """,
            input_variables=["section_name", "modification_description", "provided_data", "job_description"],
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )
        
        provided_data = modification.specific_changes.data if modification.specific_changes else {}
        
        formatted_prompt = creation_prompt.format(
            section_name=section_name,
            modification_description=modification.description,
            provided_data=json.dumps(provided_data, indent=2),
            job_description=job_description or 'Not provided'
        )
        
        response = await model.ainvoke(formatted_prompt)
        response_text = extract_response_text(response)
        
        new_item = parser.parse(response_text)
        return new_item.model_dump()
        
    except Exception as e:
        logger.error(f"Error creating new {section_name} item: {str(e)}")
        return None

async def enhance_specific_item(
    section_name: str,
    item_data: Dict[str, Any],
    modification: ModificationRequest,
    model: BaseLanguageModel,
    job_description: Optional[str] = None
) -> Dict[str, Any]:
    """
    Enhance a specific item within a section.
    
    Args:
        section_name: Name of section containing the item to enhance
        item_data: Current data dictionary for the specific item
        modification: ModificationRequest with enhancement instructions
        model: Language model for content improvement
        job_description: Optional JD for targeted enhancement
        
    Returns:
        Dict[str, Any]: Enhanced item data or original data if enhancement fails
        
    Key Features:
        - Uses existing enhance_resume_section function for consistency
        - Maintains single-item focus while leveraging section-level logic
        - Provides graceful fallback to original data on enhancement failure
        - Ensures enhanced content follows existing validation patterns
        - Integrates seamlessly with broader section modification workflow
    """
    
    try:
        enhancement_result = await enhance_resume_section(
            section_name=section_name,
            section_data={"items": [item_data]},
            model=model,
            job_description=job_description
        )
        
        if "section_data" in enhancement_result and enhancement_result["section_data"]:
            enhanced_items = enhancement_result["section_data"]
            if isinstance(enhanced_items, list) and enhanced_items:
                return enhanced_items[0]
        
    except Exception as e:
        logger.error(f"Error enhancing specific {section_name} item: {str(e)}")
    
    return item_data

async def ensure_section_exists(
    resume_data: Dict[str, Any], 
    section_name: str, 
    job_description: Optional[str] = None,
    model: Optional[BaseLanguageModel] = None
) -> Dict[str, Any]:
    """Ensure a section exists in the resume, creating it if necessary.
    
    Args:
        resume_data: Complete resume data dictionary
        section_name: Name of section to verify/create
        job_description: Optional JD for context (currently unused but future-ready)
        model: Optional model instance (currently unused but future-ready)
        
    Returns:
        Dict[str, Any]: Resume data with guaranteed section existence
        
    Key Features:
        - Creates sections with appropriate data structure based on SECTION_MODELS
        - Distinguishes between list sections and object sections automatically
        - Uses existing model information to determine correct initial structure
        - Provides logging for section creation tracking
        - Ensures no operation fails due to missing sections
    """
    if section_name in resume_data:
        return resume_data
    
    section_model = SECTION_MODELS.get(section_name)
    if section_model:
        if section_name in ["work", "education", "skills", "projects", "awards", "publications", "languages", "certifications"]:
            resume_data[section_name] = []
        else:
            resume_data[section_name] = {}
        
        logger.info(f"Created new section: {section_name} with structure based on {section_model.__name__}")
    else:
        resume_data[section_name] = []
        logger.info(f"Created new section: {section_name} with default list structure")
    
    return resume_data

async def process_resume_query(
    json_data: Dict[str, Any],
    user_query: str,
    job_description: Optional[str] = None
) -> Dict[str, Any]:
    """
    Process a user's complex query to modify their resume across multiple sections.
    
    Args:
        json_data: Complete resume data in JSON format (with or without 'details' wrapper)
        user_query: Natural language query describing desired resume modifications
        job_description: Optional job description for targeted content alignment
        
    Returns:
        Dict[str, Any]: Response containing updated resume data and operation results
        
    Key Features:
        - Handles complex multi-section queries in single operation
        - Uses existing Pydantic models from SECTION_MODELS for all data validation
        - Supports both 'details' wrapped and direct resume data formats
        - Provides detailed success/failure reporting with modification tracking
        - Processes all resume sections equally using unified generic functions
    """
    try:
        model_manager = SimpleModelManager()
        model = model_manager.get_model()
        
        logger.info(f"Processing complex resume query: {user_query[:50]}...")
        
        if 'details' in json_data and isinstance(json_data['details'], dict):
            resume_data = json_data['details']
            has_details_wrapper = True
        else:
            resume_data = json_data
            has_details_wrapper = False
    
        query_analysis = await analyze_complex_query(user_query, model)
        
        logger.info(f"Identified {len(query_analysis.modifications)} modifications:")
        for mod in query_analysis.modifications:
            logger.info(f"  - {mod.section}: {mod.action} - {mod.description}")
        
        sections_to_modify = set(mod.section for mod in query_analysis.modifications)

        for section in sections_to_modify:
            resume_data = await ensure_section_exists(resume_data, section, job_description, model)

        for section_name in sections_to_modify:
            if section_name in resume_data:
                resume_data[section_name] = await modify_section_generic(
                    section_name,
                    resume_data[section_name],
                    query_analysis.modifications,
                    model,
                    job_description
                )
                
        if has_details_wrapper:
            json_data['details'] = resume_data
            result = json_data
        else:
            result = resume_data
            
        return {
            "message": f"Resume updated successfully with {len(query_analysis.modifications)} modifications",
            "data": result,
            "success": True,
            "modifications_applied": [
                {
                    "section": mod.section,
                    "action": mod.action,
                    "description": mod.description
                }
                for mod in query_analysis.modifications
            ]
        }
        
    except Exception as e:
        logger.error(f"Error in process_resume_query: {str(e)}")
        return {
            "error": f"Query processing failed: {str(e)}",
            "data": json_data,
            "success": False
        }