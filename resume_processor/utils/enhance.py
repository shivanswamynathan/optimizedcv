import json
import logging
import asyncio
from typing import Dict, Any, Optional, List, Union, Literal
from langchain_core.language_models.base import BaseLanguageModel
from langchain_openai import ChatOpenAI, OpenAI
from langchain_google_genai import GoogleGenerativeAI
from langchain_deepseek import ChatDeepSeek
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field, field_validator, model_validator
import os
import re
from .modelmanager import SimpleModelManager

logger = logging.getLogger(__name__)


# Define Pydantic models for each resume section
class Location(BaseModel):
    city: Optional[str] = Field(None, description="City of residence.")
    region: Optional[str] = Field(None, description="Region or area (e.g., 'California', 'ALASKA','New York').")
    State: Optional[str] = Field(None, description="Statecode (e.g., 'NY', 'CO', 'NV').")

class Profile(BaseModel):
    network: str = Field(description="Name of the social network.")
    username: str = Field(description="Username on the network.")
    url: Optional[str] = Field(None, description="URL to the profile page.")

class Summary(BaseModel):
    """Model for the summary section of a resume."""
    name: str = Field(description="Name of the individual.")
    label: str = Field(description="Label or title of the individual.")
    email: str = Field(description="Email address of the individual.")
    phone: str = Field(description="Phone number of the individual.")
    url: Optional[str] = Field(None, description="URL to the individual's personal website or portfolio.")
    summary: str = Field(description="A concise, impactful professional summary of 45-50 words highlighting core skills and measurable impact.")
    location: Optional[Location] = Field(None, description="Location information.")
    profiles: Optional[List[Profile]] = Field(None, description="List of social media profiles.")

class ExperienceItem(BaseModel):
    """Model for a single work experience entry."""
    name: str = Field(description="Name of the company or organization.")
    position: str = Field(description="Job title or position held.")
    location: str = Field(description="Location of the job.")
    startDate: str = Field(description="Start date of employment.")
    endDate: Optional[str] = Field(None, description="End date of employment, or 'present' if current.")
    highlights: List[str] = Field(
        description="Bullet points demonstrating skills and responsibilities. Only include quantifiable results or metrics if they already exist in the original content for that specific point. Each bullet should begin with a strong action verb. Maintain original number of points."
    )
    
    @field_validator('highlights')
    @classmethod
    def validate_highlights(cls, v):
        if not v:
            return [] 
        return v

class EducationItem(BaseModel):
    """Model for a single education entry."""
    institution: str = Field(description="Name of the educational institution.")
    area: str = Field(description="Field of study or major.")
    degree: str = Field(description="Degree type (e.g., 'Bachelor', 'Master').")
    specialization: Optional[str] = Field(None, description="Specialization or minor, if applicable.")
    startDate: str = Field(description="Start date of education.")
    endDate: Optional[str] = Field(None, description="End date of education, or 'present' if current.")
    score: Optional[List[Dict[str, str]]] = Field(None, description="Academic scores (e.g., GPA, percentage).")
    courses: Optional[List[str]] = Field(
        None, 
        description="Bullet points emphasizing achievements, skills, or projects. Maintain original number of points."
    )
    
    @field_validator('courses')
    @classmethod
    def validate_courses(cls, v):
        if v is None:
            return None
        return v

class ProjectItem(BaseModel):
    """Model for a single project entry."""
    name: str = Field(description="Name of the project.")
    description: List[str] = Field(
        description="Bullet points emphasizing impact and technologies used. Only include measurable outcomes or metrics if they already exist in the original content for that specific point. Each bullet should begin with a strong action verb. Maintain original number of points."
    )
    startDate: str = Field(None, description="Start date of the project.")
    endDate: Optional[str] = Field(None, description="End date of the project, or 'present' if current.")

    @field_validator('description')
    @classmethod
    def validate_description(cls, v):
        if not v :
            return []
        return v
class SkillCategory(BaseModel):
    """Model for a category of skills."""
    name: str = Field(description="Name of the skill category.")
    keywords: List[str] = Field(description="List of core skills in this category. Maintain original number of skills.")
    
    @field_validator('keywords')
    @classmethod
    def validate_keywords(cls, v):
        if not v:
            return [] 
        return v
    
class Award(BaseModel):
    """Model for an award entry."""
    title: str = Field(description="Title of the award.")
    awarder: str = Field(description="Name of the award issuer.")

class Publication(BaseModel):
    """Model for a publication entry."""
    title: str = Field(description="Title of the publication.")
    publisher: str = Field(description="Publisher of the publication.")
    publishedDate: Optional[str] = Field(None, description="Date of publication.")
    publisherUrl: Optional[str] = Field(None, description="URL to the publisher's page.")
    authors: List[str] = Field(description="List of authors of the publication.")
    description:list[str] = Field(description="Description of the publication.")

class Language(BaseModel):
    """Model for a language entry."""
    name: str = Field(description="Name of the language.")
    proficiency: str = Field(description="Proficiency level in the language.")

class Certification(BaseModel):
    """Model for a certification entry."""
    certificateName: str = Field(description="Name of the certificate.")
    completionId: str = Field(description="Certificate ID or completion code.")
    date: str = Field(description="Date of certification.")
    url: Optional[str] = Field(None, description="URL to the certificate or issuer.")

    
# Wrapper models 
class ExperienceList(BaseModel):
    """Model for a list of work experiences."""
    items: List[ExperienceItem] = Field(description="List of work experience items.")

class EducationList(BaseModel):
    """Model for a list of education entries."""
    items: List[EducationItem] = Field(description="List of education entries.")

class ProjectList(BaseModel):
    """Model for a list of projects."""
    items: List[ProjectItem] = Field(description="List of project entries.")

class SkillList(BaseModel):
    """Model for a list of skill categories."""
    items: List[SkillCategory] = Field(description="List of skill categories.")

class AwardList(BaseModel):
    """Model for a list of awards."""
    items: List[Award] = Field(description="List of award entries.")

class PublicationList(BaseModel):
    """Model for a list of publications."""
    items: List[Publication] = Field(description="List of publication entries.")

class LanguageList(BaseModel):
    """Model for a list of languages."""
    items: List[Language] = Field(description="List of language entries.")

class CertificationList(BaseModel):
    """Model for a list of certifications."""
    items: List[Certification] = Field(description="List of certification entries.")


# Define combined model for all sections
class ResumeSection(BaseModel):
    """Base model for a resume section with its enhancement parameters."""
    section_name: str = Field(description="Name of the resume section being enhanced.")
    section_data: Dict[str, Any] = Field(description="Enhanced data for this section following exact format requirements.")

# Map section names to appropriate models
SECTION_MODELS = {
    "basics": Summary,
    "work": ExperienceList,  
    "education": EducationList,  
    "projects": ProjectList,  
    "skills": SkillList,  
    "awards": AwardList,  
    "publications": PublicationList,
    "languages": LanguageList,
    "certifications": CertificationList  
}

# Standard enhancement guidelines for each section
SECTION_GUIDELINES = {
    "summary": "Enhance the summary to be more professional while maintaining the original tone and meaning. Correct grammar and spelling errors. Write out all acronyms with abbreviations in parentheses (e.g., 'Customer Relationship Management (CRM)'). If a job description is provided, naturally incorporate relevant keywords that match the candidate's actual experience. Include the target role title if specified. Keep the summary concise and impactful without adding fabricated metrics or claims. NEVER add metrics that aren't explicitly mentioned in the original content.",
    "experience": " Correct grammar and spelling while preserving the original meaning and tone. Write out all acronyms with abbreviations. If a job description is provided, highlight relevant responsibilities that match the candidate's experience using similar terminology. Format each point to be clear and professional, beginning with strong action verbs. IMPORTANT: NEVER invent metrics, responsibilities, or accomplishments not mentioned in the original content. Only include quantitative metrics (numbers, percentages) if they already exist in the original content for that specific point. MAINTAIN THE ORIGINAL NUMBER OF POINTS - do not create new ones.",
    "education": "Ensure correct spelling and grammar while preserving original content. Emphasize real achievements, skills, or projects mentioned. NEVER add accomplishments not present in the original. MAINTAIN THE ORIGINAL NUMBER OF POINTS - do not create new ones.",
    "projects": " Correct grammar and spelling while preserving original meaning. Write out acronyms with abbreviations. Begin each point with a strong action verb, focusing on actual technologies and real outcomes. Highlight projects that match job description if applicable. NEVER invent unmentioned technologies, responsibilities, or metrics. MAINTAIN THE ORIGINAL NUMBER OF POINTS - do not create new ones.",
    "skills": "Organize skills into logical categories using concise, professional terminology. Write out acronyms with abbreviations. Include ONLY skills from the original content. If a job description is provided, prioritize matching skills only if mentioned in the resume. MAINTAIN THE ORIGINAL NUMBER OF SKILLS - do not create new ones. No descriptions or extra words required.",
    "awards": "Transform award listings into impactful achievements that demonstrate professional recognition. For each award, create a title that clearly communicates the significance and includes relevant context (year, scope, or criteria if available). Ensure proper formatting and consistent structure. If possible, relate awards to skills or achievements relevant to the target role without fabricating details. NEVER add awards or details not present in the original content.",
    "languages": "Format language proficiency information consistently. For each language, ensure proper capitalization and standardize proficiency levels using professional terminology (e.g., 'Native', 'Fluent', 'Professional', 'Intermediate', 'Basic'). Prioritize languages that might be relevant to the job description. Add brief context if appropriate (e.g., 'Professional working proficiency'). DO NOT add languages not mentioned in the original content.",
    "publications": "Enhance publication entries with proper academic formatting. Ensure each publication has a meaningful title, correct publisher information, and properly formatted publication date. If author information is provided as a string, convert it to a proper authors array. Fix formatting inconsistencies. DO NOT add publications or details not mentioned in the original content.",
    "certifications": "Enhance certification entries with proper naming conventions, standardized dates, and complete information. Ensure each certification name is descriptive and professional. If abbreviated or informal names are used, expand them to their full, proper titles. Add clear completion IDs and URLs if available. Format dates consistently. Prioritize certifications relevant to the target role. NEVER add certifications not present in the original content."
}

# Creates a prompt for enhancing a specific resume section based on section type
def create_section_prompt(
    section_name: str, 
    section_data: Any, 
    job_description: Optional[str] = None    
) -> Dict[str, Any]:
    """
    Create a prompt for enhancing a specific resume section.
    
    Args:
        section_name: Name of the section to enhance
        section_data: Current section data
        job_description: Optional job description for tailoring
        
    Returns:
        Dict containing the prompt text and output parser
    """
    section_format = SECTION_GUIDELINES.get(section_name, "Enhance this section to be more impactful and ATS-friendly.")
    
    section_model = SECTION_MODELS.get(section_name)
    if not section_model:
        logger.warning(f"No schema defined for section '{section_name}', using generic structure")
        return {
            "prompt": PromptTemplate(
                template=(
                    f"Enhance the {section_name} section of this resume.\n\n"
                    f"FORMAT GUIDELINES:\n{section_format}\n\n"
                    f"JOB DESCRIPTION:\n{job_description}\n\n"
                    f"ORIGINAL CONTENT:\n{section_data}\n\n"
                    "CRITICAL INSTRUCTION: DO NOT ADD ANY NEW FIELDS THAT DON'T EXIST IN THE ORIGINAL CONTENT. "
                    "ONLY enhance fields that already exist in the original data.\n\n"
                    "Return the enhanced content in the same structure as the original."
                ),
                input_variables=["section_name", "section_format", "job_description", "original_content"]
            ),
            "parser": None
        }
    # Special handling for list types - pack them into wrapper models
    if section_name in ["work", "education", "skills", "projects", "awards", "publications", "languages", "certifications"]:
        if isinstance(section_data, list):
            modified_data = {"items": section_data}
        else:
            logger.warning(f"Unexpected data format for {section_name}: {type(section_data)}")
            modified_data = {"items": section_data} if section_data else {"items": []}
    else:
        modified_data = section_data
    
    # Create Pydantic parser
    parser = PydanticOutputParser(pydantic_object=section_model)
    
    is_list_wrapper = section_name in ["work", "education", "skills", "projects", "awards", "publications", "languages", "certifications"]
    
    if is_list_wrapper:
        
        prompt_template = PromptTemplate(
        template=(
            "Enhance the '{section_name}' section of this resume to improve clarity, impact, and alignment with the job description.\n\n"
            "FORMAT GUIDELINES:\n{section_format}\n\n"
            "JOB DESCRIPTION:\n{job_description}\n\n"
            "ORIGINAL CONTENT:\n{original_content}\n\n"
            "{format_instructions}\n\n"
            "CRITICAL INSTRUCTION: DO NOT ADD ANY NEW FIELDS THAT DON'T EXIST IN THE ORIGINAL CONTENT. "
            "ONLY enhance fields that already exist in the original data.\n\n"
            "IMPORTANT:\n"
            "Return ONLY valid JSON. "
            + ("Response must be a JSON object with an \"items\" list.\n" if is_list_wrapper else "") +
            "Do not include explanations or extra text."
        ),
        input_variables=["section_name", "section_format", "job_description", "original_content"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
        
        return {
        "prompt": prompt_template,
        "parser": parser
    }
    else:
        return {
            "prompt": f"""
                Enhance the {section_name} section of this resume according to the provided guidelines.
                
                FORMAT GUIDELINES:
                {section_format}
                
                JOB DESCRIPTION:
                {job_description or "Not provided"}
                
                ORIGINAL CONTENT:
                {json.dumps(section_data, indent=2)}
                
                {parser.get_format_instructions()}
                
                IMPORTANT: Make sure your response can be parsed into the required format. Return ONLY valid JSON.
                Do not include any explanations, just the enhanced content following the schema exactly.
            """,
            "parser": parser
        }

# Extracts text content from model responses based on the model type
def extract_response_text(response: Any, model: BaseLanguageModel) -> str:
    """Extract the response text based on model type."""
    if isinstance(model, GoogleGenerativeAI):
        return response.text if hasattr(response, 'text') else str(response)
    elif isinstance(model, (ChatOpenAI, ChatDeepSeek, OpenAI)):
        return response.content if hasattr(response, 'content') else str(response)
    return str(response)

# Extracts token usage information from model responses
def get_token_counts_from_response(response: Any) -> Dict[str, int]:
    """Extract token counts from model response."""
    token_counts = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    
    if hasattr(response, 'response_metadata'):
        metadata = response.response_metadata
        if hasattr(metadata, 'token_usage') or 'token_usage' in metadata:
            token_usage = metadata.get('token_usage', {})
            token_counts["input_tokens"] = token_usage.get('prompt_tokens', 0)
            token_counts["output_tokens"] = token_usage.get('completion_tokens', 0)
            token_counts["total_tokens"] = token_usage.get('total_tokens', 0)
    
    return token_counts

# Enhances a single section of the resume using the LLM with Pydantic validation
async def enhance_resume_section(
    section_name: str,
    section_data: Any,
    model: BaseLanguageModel,
    job_description: Optional[str] = None
) -> Dict[str, Any]:
    """
    Enhance a single section of the resume using the LLM with Pydantic validation.
    
    Args:
        section_name: Name of the section to enhance
        section_data: Current section data
        model: Language model to use
        job_description: Optional job description
        
    Returns:
        Dict containing enhanced section data and token metrics
    """
    try:
        logger.info(f"[DEBUG] Enhancing section: {section_name}")
        
        # Create prompt with schema
        prompt_data = create_section_prompt(
            section_name, 
            section_data, 
            job_description
        )
        
        logger.debug(f"Section model prompt for {section_name}: {prompt_data['prompt']}")

        # Prepare prompt for model invocation
        prompt = prompt_data["prompt"]
        if isinstance(prompt, PromptTemplate):
            section_format = SECTION_GUIDELINES.get(section_name, "Enhance this section.")
            if section_name in ["work", "education", "skills", "projects", "awards", "publications", "languages", "certifications"] and isinstance(section_data, dict) and "items" in section_data:
                original_content = json.dumps(section_data["items"], indent=2)
            else:
                original_content = json.dumps(section_data, indent=2)
            prompt = prompt.format(
                section_name=section_name,
                section_format=section_format,
                job_description=job_description or "Not provided",
                original_content=original_content
            )
        
        # Call the model
        response = await model.ainvoke(prompt)
        response_text = extract_response_text(response, model)
        
        logger.debug(f"Section model response for {section_name}: {response_text[:500]}")
        
        # Get token counts
        token_counts = get_token_counts_from_response(response)

        logger.info(f"Section: {section_name} | Input tokens: {token_counts['input_tokens']} | Output tokens: {token_counts['output_tokens']} | Total tokens: {token_counts['total_tokens']}")
        
        # Parse the enhanced content
        enhanced_section = section_data  
        try:
            if prompt_data["parser"]:
                parsed_data = prompt_data["parser"].parse(response_text)
                
                # Convert to dict if it's a Pydantic model
                if hasattr(parsed_data, "model_dump"):
                    parsed_dict = parsed_data.model_dump()
                    
                    if section_name in ["work", "education", "skills", "projects", "awards", "publications", "languages", "certifications"]:
                        if "items" in parsed_dict:
                            enhanced_section = parsed_dict["items"]
                        else:
                            logger.warning(f"Expected 'items' field in {section_name} response, but not found")
                            enhanced_section = parsed_dict 
                    else:
                        enhanced_section = parsed_dict
                else:
                    logger.warning(f"Unexpected parsed data type for {section_name}: {type(parsed_data)}")
                    enhanced_section = parsed_data
            else:
                # Fallback to JSON parsing if no parser
                match = re.search(r'(\{.*\}|\[.*\])', response_text, re.DOTALL)
                if match:
                    json_str = match.group(0)
                    enhanced_section = json.loads(json_str)
        except Exception as e:
            logger.error(f"Failed to parse response for section {section_name}: {str(e)}")
            logger.debug(f"Response text: {response_text[:500]}...")
            
        # Post-process to remove hallucinated fields for education section
        if section_name == "education":
            original_courses = []
            if isinstance(section_data, dict) and "items" in section_data:
                input_items = section_data["items"]
            else:
                input_items = section_data
            for entry in input_items:
                original_courses.append(bool(entry.get("courses")))
            if isinstance(enhanced_section, list):
                for idx, entry in enumerate(enhanced_section):
                    if idx < len(original_courses) and not original_courses[idx]:
                        if "courses" in entry:
                            entry["courses"] = None

        return {
            "section_data": enhanced_section, 
            "input_tokens": token_counts['input_tokens'], 
            "output_tokens": token_counts['output_tokens'],
            "total_tokens": token_counts['total_tokens']
        }
    except Exception as e:
        logger.error(f"Error enhancing section {section_name}: {str(e)}")
        return {"section_data": section_data, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

# Enhances each section of the resume asynchronously and combines results
async def enhance_resume_by_sections(
    json_data: Dict[str, Any],
    model: BaseLanguageModel,
    job_description: Optional[str] = None
) -> Dict[str, Any]:
    """
    Enhance each section of the resume asynchronously and combine results.
    """
    if 'details' in json_data and isinstance(json_data['details'], dict):
        resume_data = json_data['details']
        has_details_wrapper = True
    else:
        resume_data = json_data
        has_details_wrapper = False
    
    logger.info(f"Starting resume enhancement")
    
    allowed_sections = list(SECTION_GUIDELINES.keys())
    
    # Map from JSON section names to template parameter names
    section_mapping = {
        "basics": "summary",
        "work": "experience",
        "skills": "skills",
        "education": "education",
        "projects": "projects",
        "awards": "awards",
        "languages": "languages",
        "publications": "publications",
        "certifications": "certifications"
    }
    
    enhancement_tasks = {}
    for section_name, section_data in resume_data.items():
        if section_name in ['token_metrics', 'JD']:
            continue

        mapped_section = section_mapping.get(section_name, section_name)
        
        if mapped_section in allowed_sections:
            logger.info(f"Enhancing section {section_name} (maps to template section {mapped_section})")
            task = asyncio.create_task(
                enhance_resume_section(
                    section_name, 
                    section_data, 
                    model, 
                    job_description
                )
            )
            enhancement_tasks[section_name] = task
        else:
            logger.info(f"Skipping enhancement for section {section_name} - not defined")
    
    enhanced_sections = {}
    total_input_tokens = 0
    total_output_tokens = 0
    total_tokens = 0
    
    # Process enhanced sections and add original unenhanced sections
    for section_name, section_data in resume_data.items():
        if section_name in ['token_metrics', 'JD']:
            continue
            
        if section_name in enhancement_tasks:
            result = await enhancement_tasks[section_name]
            enhanced_sections[section_name] = result["section_data"]
            total_input_tokens += result["input_tokens"]
            total_output_tokens += result["output_tokens"]
            total_tokens += result.get("total_tokens", 0)
        else:
            enhanced_sections[section_name] = section_data

    logger.info(f"Resume enhancement completed |"
                f"Total input tokens: {total_input_tokens} | "
                f"Total output tokens: {total_output_tokens} | "
                f"Total tokens: {total_tokens}")
    
    if has_details_wrapper:
        result = {'details': enhanced_sections}
        if 'JD' in json_data:
            result['JD'] = json_data['JD']
        result['token_metrics'] = {
            'total_input_tokens': total_input_tokens,
            'total_output_tokens': total_output_tokens,
            'total_tokens': total_tokens
        }
        return result
    
    enhanced_sections['token_metrics'] = {
        'total_input_tokens': total_input_tokens,
        'total_output_tokens': total_output_tokens,
        'total_tokens': total_tokens
    }
    
    return enhanced_sections

# Pydantic model for validating a resume enhancement request
class ResumeEnhancementRequest(BaseModel):
    """Model for a resume enhancement request with validation."""
    json_data: Dict[str, Any] = Field(description="Resume data in JSON format")
    job_description: Optional[str] = Field(None, description="Optional job description to tailor the resume")
    
    @field_validator('json_data')
    @classmethod
    def validate_json_data(cls, v):
        """Validate that the JSON data has the required structure."""
        if not v:
            raise ValueError("Resume data cannot be empty")
        
        # Check if we have either details or required sections
        if 'details' not in v and not any(key in v for key in ['basics', 'work', 'education', 'skills', 'awards', 'languages', 'publications', 'certifications']):
            raise ValueError("Resume data must contain 'details' or at least one resume section")
            
        return v

async def generate_enhancement_suggestions(original_resume: Dict[str, Any], enhanced_resume: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Generate 3 enhancement suggestions by comparing original vs enhanced resume.
    Shows what improvements were made and suggests further enhancements.
    
    Args:
        original_resume: The original resume data
        enhanced_resume: The enhanced resume data
        
    Returns:
        List of 3 enhancement suggestions
    """
    try:
        # Initialize model
        model_manager = SimpleModelManager()
        model = model_manager.get_model()
        
        # Create comparison prompt
        prompt = f"""You are a resume enhancement expert. Based on comparing the original and enhanced resume below, generate exactly 3 questions that the user might want to ask for further enhancement.

ORIGINAL RESUME:
{json.dumps(original_resume, indent=2)}

ENHANCED RESUME:
{json.dumps(enhanced_resume, indent=2)}

Generate 3 questions that represent what the user might ask to enhance their resume further. These should be natural questions a user would ask, like:

**Examples of good enhancement questions:**
- "Can you make my work experience more impactful?"
- "How can I improve my project descriptions?"
- "Can you add metrics to my achievements?"
- "How can I make my skills section stand out more?"
- "Can you strengthen my summary to be more compelling?"
- "How can I quantify my accomplishments better?"
- "Can you make my job responsibilities more specific?"
- "How can I highlight my technical expertise better?"
- "Can you improve the impact of my project descriptions?"

Based on what areas still need improvement in this specific resume, generate 3 questions the user might ask. Focus on:
- Areas that could use more metrics/quantification
- Sections that could be more impactful
- Skills that could be better organized
- Achievements that could be highlighted better

Return in this JSON format:
{{
    "enhancement_suggestions": [
        {{
            "suggestion": "Can you make my work experience more impactful?",
        }},
        {{
            "suggestion": "How can I improve my project descriptions?",
        }},
        {{
            "suggestion": "Can you add more metrics to my achievements?",
        }}
    ]
}}

IMPORTANT:
- Frame as questions the user would naturally ask
- Start with phrases like "Can you...", "How can I...", "Could you help me..."
- Focus on specific areas that still need improvement
- Make them actionable and specific to this resume
- These are questions for further enhancement, not just general advice

Return only valid JSON without additional text."""

        # Get response from model
        response = await model.ainvoke(prompt)
        response_text = extract_response_text(response, model)
        
        # Clean and parse JSON
        cleaned_response = response_text.strip()
        if cleaned_response.startswith('```json'):
            cleaned_response = cleaned_response.replace('```json', '').replace('```', '').strip()
        elif cleaned_response.startswith('```'):
            cleaned_response = cleaned_response.replace('```', '').strip()
        
        try:
            result = json.loads(cleaned_response)
            suggestions = result.get('enhancement_suggestions', [])
            
            # Validate we have exactly 3 suggestions
            if len(suggestions) == 3:
                logger.info("Successfully generated 3 enhancement suggestions")
                return suggestions
            else:
                logger.warning(f"Expected 3 suggestions, got {len(suggestions)}")
                return []
                
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse enhancement suggestions JSON: {str(e)}")
            return []
            
    except Exception as e:
        logger.error(f"Error generating enhancement suggestions: {str(e)}")
        return []

 # Enhances resume data using the specified LLM model, section by section   
async def enhance_resume_with_model(
    json_data: Dict[str, Any],
    job_description: Optional[str] = None
) -> Dict[str, Any]:
    """
    Enhance resume data using the specified LLM model, section by section.
    Generate enhancement suggestions during the process.
    
    Args:
        json_data: The resume data in JSON format
        job_description: Optional job description to tailor the resume
    
    Returns:
        Dict containing the enhanced resume data with token metrics
    """
    try:
        # Validate request with Pydantic
        request = ResumeEnhancementRequest(
            json_data=json_data,
            job_description=job_description
        )
        
        # Set validated values
        json_data = request.json_data

        original_resume = json_data.copy()
        
        # Initialize model
        model = SimpleModelManager().get_model()
        
        jd = request.job_description
        if not jd and 'JD' in json_data:
            jd = json_data.get('JD')
            
        enhanced_json = await enhance_resume_by_sections(
            json_data, 
            model, 
            jd
        )

        try:
            # Extract resume data for comparison
            original_data = original_resume.get('details', original_resume)
            enhanced_data = enhanced_json.get('details', enhanced_json)
            
            enhancement_suggestions = await generate_enhancement_suggestions(
                original_data, 
                enhanced_data
            )

            logger.info(f"DEBUG: Generated {len(enhancement_suggestions) if enhancement_suggestions else 0} suggestions: {enhancement_suggestions}")
            
            if enhancement_suggestions:
                # Store suggestions in the response (temporarily during enhancement)
                enhanced_json['enhancement_suggestions'] = enhancement_suggestions
                logger.info("Enhancement suggestions generated successfully")
                print ("aaaaaa")
            
        except Exception as e:
            logger.error(f"Error generating enhancement suggestions: {str(e)}")
            # Don't fail the entire enhancement if suggestions fail
        
        logger.info("Resume enhancement completed successfully")
        return enhanced_json
        
    except ValueError as e:
        logger.error(f"Resume enhancement validation failed: {str(e)}")
        if 'token_metrics' not in json_data:
            json_data['token_metrics'] = {
                'total_input_tokens': 0,
                'total_output_tokens': 0,
                'total_tokens': 0,
                'validation_error': str(e)
            }
        return json_data
    except Exception as e:
        logger.error(f"Resume enhancement failed: {str(e)}")
        if 'token_metrics' not in json_data:
            json_data['token_metrics'] = {
                'total_input_tokens': 0,
                'total_output_tokens': 0,
                'total_tokens': 0,
                'error': str(e)
            }
        return json_data    

# Pydantic model for validating a process resume request
class ProcessResumeRequest(BaseModel):
    """Model for validating a process resume request."""
    json_data: Dict[str, Any] = Field(description="Resume data in JSON format")
    job_description: Optional[str] = Field(None, description="Optional job description to tailor the resume")
    
    model_config = {
        "extra": "ignore"
    }

# Processes a resume by enhancing it section by section with Pydantic validation    
async def process_resume(
    json_data: Dict[str, Any],
    job_description: Optional[str] = None
) -> Dict[str, Any]:
    """
    Process a resume by enhancing it section by section with Pydantic validation.
    
    Args:
        json_data: The resume data in JSON format
        job_description: Optional job description to tailor the resume
        
    Returns:
        Dict containing the enhanced resume data
    """
    try:
        request = ProcessResumeRequest(
            json_data=json_data,
            job_description=job_description
        )

        enhanced_json = await enhance_resume_with_model(
            request.json_data,
            request.job_description
        )
        
        return enhanced_json
        
    except Exception as e:
        logger.error(f"Error processing resume: {str(e)}")
        if 'token_metrics' not in json_data:
            json_data['token_metrics'] = {
                'total_input_tokens': 0,
                'total_output_tokens': 0,
                'total_tokens': 0,
                'process_error': str(e)
            }
        return json_data