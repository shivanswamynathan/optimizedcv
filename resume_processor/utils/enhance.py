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
        description="Exactly 5 bullet points, each demonstrating skills and quantifiable results. Each bullet must begin with a strong action verb."
    )
    
    @field_validator('highlights')
    @classmethod
    def validate_highlights(cls, v):
        if not v or len(v) != 5:
            # Ensure we have exactly 5 items, padding with empty strings if necessary
            v = (v or [])[:5] + [''] * (5 - len(v or []))
        return v

# Wrapper model for list of work experiences to fix Pydantic v2 compatibility
class ExperienceList(BaseModel):
    """Model for a list of work experiences."""
    items: List[ExperienceItem] = Field(description="List of work experience items.")

class EducationItem(BaseModel):
    """Model for a single education entry."""
    institution: str = Field(description="Name of the educational institution.")
    area: str = Field(description="Field of study or major.")
    degree: str = Field(description="Degree type (e.g., 'BSc', 'MSc', 'BA').")
    specialization: Optional[str] = Field(None, description="Specialization or minor, if applicable.")
    startDate: str = Field(description="Start date of education.")
    endDate: Optional[str] = Field(None, description="End date of education, or 'present' if current.")
    score: Optional[List[Dict[str, str]]] = Field(None, description="Academic scores (e.g., GPA, percentage).")
    courses: Optional[List[str]] = Field(
        None, 
        description="Exactly 3 bullet points emphasizing achievements, skills, or projects. Each must be 10-13 words only."
    )
    
    @field_validator('courses')
    @classmethod
    def validate_courses(cls, v):
        if v is not None and len(v) != 3:
            # Ensure we have exactly 3 items, padding with empty strings if necessary
            v = (v or [])[:3] + [''] * (3 - len(v or []))
        return v

# Wrapper model for list of education entries
class EducationList(BaseModel):
    """Model for a list of education entries."""
    items: List[EducationItem] = Field(description="List of education entries.")

class ProjectItem(BaseModel):
    """Model for a single project entry."""
    name: str = Field(description="Name of the project.")
    description: List[str] = Field(
        description="Exactly 5 bullet points emphasizing impact, technologies used, and measurable outcomes. Each bullet must begin with a strong action verb."
    )
    startDate: str = Field(None, description="Start date of the project.")
    endDate: Optional[str] = Field(None, description="End date of the project, or 'present' if current.")

    @field_validator('description')
    @classmethod
    def validate_description(cls, v):
        if not v or len(v) != 5:
            # Ensure we have exactly 5 items, padding with empty strings if necessary
            v = (v or [])[:5] + [''] * (5 - len(v or []))
        return v

# Wrapper model for list of projects
class ProjectList(BaseModel):
    """Model for a list of projects."""
    items: List[ProjectItem] = Field(description="List of project entries.")

class SkillCategory(BaseModel):
    """Model for a category of skills."""
    name: str = Field(description="Name of the skill category.")
    keywords: List[str] = Field(description="List of exactly 4 core skills in this category.")
    
    @field_validator('keywords')
    @classmethod
    def validate_keywords(cls, v):
        if not v or len(v) != 4:
            # Ensure we have exactly 4 items, padding with empty strings if necessary
            v = (v or [])[:4] + [''] * (4 - len(v or []))
        return v

# Wrapper model for list of skill categories
class SkillList(BaseModel):
    """Model for a list of skill categories."""
    items: List[SkillCategory] = Field(description="List of skill categories.")

class Award(BaseModel):
    """Model for an award entry."""
    title: str = Field(description="Title of the award.")
    awarder: str = Field(description="Name of the award issuer.")
# Wrapper model for list of awards
class AwardList(BaseModel):
    """Model for a list of awards."""
    items: List[Award] = Field(description="List of award entries.")

class Publication(BaseModel):
    """Model for a publication entry."""
    title: str = Field(description="Title of the publication.")
    publisher: str = Field(description="Publisher of the publication.")
    publishedDate: Optional[str] = Field(None, description="Date of publication.")
    publisherUrl: Optional[str] = Field(None, description="URL to the publisher's page.")
    authors: List[str] = Field(description="List of authors of the publication.")
    description:list[str] = Field(description="Description of the publication.")
    
# Wrapper model for list of publications
class PublicationList(BaseModel):
    """Model for a list of publications."""
    items: List[Publication] = Field(description="List of publication entries.")

class Language(BaseModel):
    """Model for a language entry."""
    name: str = Field(description="Name of the language.")
    proficiency: str = Field(description="Proficiency level in the language.")

class LanguageList(BaseModel):
    """Model for a list of languages."""
    items: List[Language] = Field(description="List of language entries.")

class Certification(BaseModel):
    """Model for a certification entry."""
    certificateName: str = Field(description="Name of the certificate.")
    completionId: str = Field(description="Certificate ID or completion code.")
    date: str = Field(description="Date of certification.")
    url: Optional[str] = Field(None, description="URL to the certificate or issuer.")

class CertificationList(BaseModel):
    """Model for a list of certifications."""
    items: List[Certification] = Field(description="List of certification entries.")



# Define combined model for all sections
class ResumeSection(BaseModel):
    """Base model for a resume section with its enhancement parameters."""
    section_name: str = Field(description="Name of the resume section being enhanced.")
    section_data: Dict[str, Any] = Field(description="Enhanced data for this section following exact format requirements.")

# Template specific models
class SoftwareEngineerResume(BaseModel):
    """Complete enhanced resume for a Software Engineer."""
    optimized_summary: str = Field(
        description="Technical summary of exactly 55-60 words focusing on specialized programming skills, achievements, and measurable impact."
    )
    optimized_experience: List[str] = Field(
        description="List of exactly 5 bullet points for each experience entry, formatted according to given constraints."
    )
    optimized_education: List[str] = Field(
        description="List of 3 education bullet points with strict adherence to word limits and abbreviations."
    )
    optimized_projects: List[str] = Field(
        description="List of 5 formatted project bullet points with technical depth."
    )
    optimized_skills: Dict[str, List[str]] = Field(
        description="Skills categorized into exactly 5 technical categories, each containing exactly 4 specialized skills."
    )
    optimized_awards: Optional[List[str]] = Field(
        None,
        description="List of optimized award descriptions following the required format."
    )

class SimpleResume(BaseModel):
    """Complete enhanced resume for a simple template."""
    optimized_summary: str = Field(
        description="Rewritten summary that follows the STRICT word limit and content constraints."
    )
    optimized_experience: List[str] = Field(
        description="List of EXACTLY 5 bullet points per experience entry, formatted according to the given constraints."
    )
    optimized_education: List[str] = Field(
        description="List of EXACTLY 3 education bullet points with STRICT adherence to word limits and abbreviations."
    )
    optimized_projects: List[str] = Field(
        description="List of EXACTLY 5 formatted project bullet points with technical depth."
    )
    optimized_skills: Dict[str, List[str]] = Field(
        description="Skills categorized into EXACTLY 4 categories, each containing EXACTLY 4 specialized skills."
    )
    optimized_awards: Optional[List[str]] = Field(
        None,
        description="List of optimized award descriptions following the STRICT format."
    )
    optimized_languages: Optional[List[Dict[str, str]]] = Field(
        None,
        description="List of languages with accurate proficiency levels."
    )
    optimized_publications: Optional[List[Dict[str, Any]]] = Field(
        None,
        description="List of enhanced publication descriptions with proper academic formatting."
    )
    optimized_certifications: Optional[List[Dict[str, str]]] = Field(
        None,
        description="List of certifications with proper names, issuing organizations, and dates."
    )

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

# Map template types to complete resume models
TEMPLATE_MODELS = {
    "simple": SimpleResume,
    "software_engineer": SoftwareEngineerResume
}

# Pydantic models for template parameters
class TemplateParameter(BaseModel):
    """Model for a single template parameter."""
    name: str = Field(description="Name of the parameter (section)")
    type: str = Field(description="Data type of the parameter")
    description: str = Field(description="Formatting guidelines for this section")

class TemplatePrompt(BaseModel):
    """Model for a template prompt with its parameters."""
    parameters: List[TemplateParameter] = Field(description="List of parameters for each section in the template")
    
    @field_validator('parameters')
    @classmethod
    def validate_parameters(cls, v):
        """Ensure all required section parameters are present."""
        required_sections = ["summary", "experience", "education", "projects", "skills"]
        optional_sections = ["awards", "languages", "publications", "certifications"]
        section_names = [param.name for param in v]
        
        for section in required_sections:
            if section not in section_names:
                raise ValueError(f"Required section '{section}' is missing from template parameters")
        
        for section in optional_sections:
            if section in section_names:
            # Add any specific validation for these optional sections if needed
                pass
        
        return v

class SimpleTemplate(TemplatePrompt):
    """Template prompt for the 'simple' resume style."""
    
    @model_validator(mode='after')
    def validate_simple_format(self) -> 'SimpleTemplate':
        """Validate specific requirements for the simple template."""
        parameters = self.parameters
        
        # Verify skills parameter requires exactly 4 categories
        skills_param = next((p for p in parameters if p.name == "skills"), None)
        if skills_param and "EXACTLY 4 categories" not in skills_param.description:
            raise ValueError("Simple template must specify EXACTLY 4 skill categories")
            
        return self

class SoftwareEngineerTemplate(TemplatePrompt):
    """Template prompt for the 'software_engineer' resume style."""
    
    @model_validator(mode='after')
    def validate_se_format(self) -> 'SoftwareEngineerTemplate':
        """Validate specific requirements for the software engineer template."""
        parameters = self.parameters
        
        # Verify skills parameter requires exactly 5 categories for software engineer
        skills_param = next((p for p in parameters if p.name == "skills"), None)
        if skills_param and "exactly 5 technical categories" not in skills_param.description:
            raise ValueError("Software engineer template must specify exactly 5 technical skill categories")
            
        return self

# Define template prompt instances with validation
simple_template = SimpleTemplate(
    parameters=[
        TemplateParameter(
            name="summary",
            type="string",
            description="Generate a concise and impactful summary, ideally between 45–50 words. Enhance the summary to be more professional while maintaining the original tone and meaning. Correct grammar and spelling errors. Write out all acronyms with abbreviations in parentheses (e.g., 'Customer Relationship Management (CRM)'). If a job description is provided, naturally incorporate relevant keywords that match the candidate's actual experience. Include the target role title if specified. Keep the summary concise and impactful without adding fabricated metrics or claims."
        ),
        TemplateParameter(
            name="experience",
            type="string",
            description="Each experience section should contain 5 bullet points. Prefer two-line entries with around 28–32 words each showcasing applied skills and quantifiable results. Correct grammar and spelling while preserving the original meaning and tone. Write out all acronyms with abbreviations in parentheses. If a job description is provided, highlight relevant responsibilities that match the candidate's experience using similar terminology. Format each point to be clear and professional, beginning with strong action verbs. Avoid inventing metrics, responsibilities, or accomplishments not mentioned in the original content."
        ),
        TemplateParameter(
            name="education",
            type="string",
            description="Provide 3 bullet points for education details. Aim to keep each point within 10–13 words. Use degree abbreviations such as 'BSc', 'MSc', 'BA', 'MA', and 'BE' for uniformity. Ensure correct spelling and grammar while preserving original content. Emphasize real achievements, skills, or projects mentioned. Avoid adding accomplishments not present in the original."
        ),
        TemplateParameter(
            name="projects",
            type="string",
            description="Include 5 bullet points per project. Prefer two-line entries of 28–32 words emphasizing impact, technologies used, and measurable outcomes. Use one-line entries (15–16 words) only when necessary. Correct grammar and spelling while preserving original meaning. Write out acronyms with abbreviations. Begin each point with a strong action verb, focusing on actual technologies and real outcomes. Highlight projects that match job description if applicable. Do not invent unmentioned technologies, responsibilities, or metrics."
        ),
        TemplateParameter(
            name="skills",
            type="string",
            description="Organize skills into EXACTLY 4 categories, ideally listing 4 core skills per category. Use concise, professional terminology. Write out acronyms with abbreviations. Include only skills from the original content. If a job description is provided, prioritize matching skills only if mentioned in the resume. No descriptions or extra words required."
        ),
        TemplateParameter(
            name="awards",
            type="string",
            description="Enhance each award description with professional language while keeping the original meaning. Emphasize significance and recognition using factual information. Avoid fabricating or exaggerating metrics or impact. Highlight relevance to the target role if genuinely applicable."
        ),
        TemplateParameter(
            name="languages",
            type="string",
            description="List languages accurately with their actual proficiency levels. Maintain the original language information without adding languages not listed in the original resume. Format consistently and professionally."
        ),
        TemplateParameter(
            name="publications",
            type="string",
            description="Enhance publication descriptions with proper academic formatting and accurate citation information. Maintain all original authors and content details. Do NOT invent or add publications not present in the original resume."
        ),
        TemplateParameter(
            name="certifications",
            type="string",
            description="List certifications with their proper names, issuing organizations, and dates. Correct any formatting or spelling issues while maintaining the original certification details. Do NOT add certifications not present in the original resume."
        )
    ]
)

software_engineer_template = SoftwareEngineerTemplate(
    parameters=[
        TemplateParameter(
            name="summary",
            type="string",
            description="Generate a concise and impactful summary, ideally between 55–60 words. Enhance the summary to be more professional while maintaining the original tone and focusing on technical skills. Correct grammar and spelling errors. Write out all technical acronyms with abbreviations (e.g., 'Application Programming Interface (API)'). If a job description is provided, naturally incorporate relevant technical keywords that match the candidate's actual experience. Include the target software role title if specified. Avoid fabricated technical expertise or metrics."
        ),
        TemplateParameter(
            name="experience",
            type="string",
            description="Each experience section should include 5 bullet points. Prefer two-line entries with 28–32 words. Use one-line entries (15–16 words) only when content cannot be expanded. Correct grammar and spelling while maintaining technical accuracy. Write out acronyms with abbreviations. Highlight relevant technical responsibilities that align with the job description if provided. Begin each point with a technical action verb. Avoid inventing technical skills, project involvement, or metrics."
        ),
        TemplateParameter(
            name="education",
            type="string",
            description="Provide 3 bullet points for education details. Ensure correct grammar and spelling. Emphasize actual technical coursework, certifications, or projects from the original content. Use appropriate academic terms and abbreviations like 'BSc', 'MSc', 'BE', 'BTech'. Avoid adding educational details not present in the original."
        ),
        TemplateParameter(
            name="projects",
            type="string",
            description="Organize skills into 5 categories, each containing 4 specialized technical skills. Use precise technical terms. Write out acronyms with abbreviations. Include only original resume skills. Prioritize skills matching the job description only if mentioned in the resume."
        ),
        TemplateParameter(
            name="skills",
            type="string",
            description="List skills under exactly 5 technical categories with exactly 4 specialized skills per category. Organize technical skills into EXACTLY 5 categories, with EXACTLY 4 specialized skills per category. Use precise technical terminology. Write out all technical acronyms with abbreviations in parentheses. Include only skills present in the original content. If a job description is provided, prioritize listing technical skills that match the requirements"
        ),
        TemplateParameter(
            name="awards",
            type="string",
            description="Enhance award descriptions professionally while retaining original meaning. Emphasize technical contributions using actual information. Avoid fabricating achievements. Highlight awards relevant to the technical role if applicable."
        )
    ]
)

# Template prompts dictionary using validated Pydantic models
TEMPLATE_PROMPTS = {
    "simple": simple_template.model_dump(),
    "software_engineer": software_engineer_template.model_dump()
}

class TemplateSectionConfig(BaseModel):
    """Configuration for a specific section in a template."""
    section_name: str = Field(description="Name of the section")
    template_type: str = Field(description="Type of template")
    format_description: str = Field(description="Formatting guidelines for this section")
    
    @field_validator('template_type')
    @classmethod
    def validate_template_type(cls, v):
        """Ensure template type is valid."""
        if v not in TEMPLATE_PROMPTS:
            raise ValueError(f"Template type '{v}' is not valid")
        return v

def get_section_format(section_name: str, template_type: str = "simple") -> str:
    """
    Get formatting guidelines for a specific section from the template.
    
    Args:
        section_name: Name of the section to get format for
        template_type: Resume template type
        
    Returns:
        str: Formatting guidelines for the section
    """
    template_params = TEMPLATE_PROMPTS.get(template_type, TEMPLATE_PROMPTS["simple"])
    
    section_name_mapping = {
        "basics": "summary",
        "work": "experience"
    }
    
    mapped_section = section_name_mapping.get(section_name, section_name)
    
    for param in template_params.get("parameters", []):
        if param.get("name") == mapped_section:
            try:
                # Validate configuration with Pydantic
                config = TemplateSectionConfig(
                    section_name=section_name,
                    template_type=template_type,
                    format_description=param.get("description", "")
                )
                return config.format_description
            except Exception as e:
                logger.warning(f"Invalid section format configuration: {e}")
                break
    
    return "Enhance this section to be more impactful and ATS-friendly."

def create_section_prompt_with_schema(
    section_name: str, 
    section_data: Any, 
    job_description: Optional[str] = None,
    template_type: str = "simple"
) -> Dict[str, Any]:
    """
    Create a prompt for enhancing a specific resume section with Pydantic schema.
    
    Args:
        section_name: Name of the section to enhance
        section_data: Current section data
        job_description: Optional job description for tailoring
        template_type: Resume template type
        
    Returns:
        Dict containing the prompt text and output parser
    """
    section_format = get_section_format(section_name, template_type)
    
    # Map section name to appropriate model
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
                    "Return the enhanced content in the same structure as the original."
                ),
                input_variables=["section_name", "section_format", "job_description", "original_content"]
            ),
            "parser": None
        }
    
    # Special handling for list types - pack them into wrapper models
    if section_name == "work":
        # Handle conversion to the right format
        if isinstance(section_data, list):
            # Already a list, wrap in items
            modified_data = {"items": section_data}
        else:
            # Not sure what format this is, try to handle gracefully
            logger.warning(f"Unexpected data format for {section_name}: {type(section_data)}")
            modified_data = {"items": section_data} if section_data else {"items": []}
    elif section_name == "education":
        if isinstance(section_data, list):
            modified_data = {"items": section_data}
        else:
            modified_data = {"items": section_data} if section_data else {"items": []}
    elif section_name == "skills":
        if isinstance(section_data, list):
            modified_data = {"items": section_data}
        else:
            modified_data = {"items": section_data} if section_data else {"items": []}
    elif section_name == "projects":
        if isinstance(section_data, list):
            modified_data = {"items": section_data}
        else:
            modified_data = {"items": section_data} if section_data else {"items": []}
    elif section_name == "awards":
        if isinstance(section_data, list):
            modified_data = {"items": section_data}
        else:
            modified_data = {"items": section_data} if section_data else {"items": []}
    elif section_name == "publications":
        if isinstance(section_data, list):
            modified_data = {"items": section_data}
        else:
            modified_data = {"items": section_data} if section_data else {"items": []}
    elif section_name == "languages":
        if isinstance(section_data, list):
            modified_data = {"items": section_data}
        else:
            modified_data = {"items": section_data} if section_data else {"items": []}
    elif section_name == "certifications":
        if isinstance(section_data, list):
            modified_data = {"items": section_data}
        else:
            modified_data = {"items": section_data} if section_data else {"items": []}
    else:
        # For non-list sections, use the data as is
        modified_data = section_data
    
    # Create Pydantic parser
    parser = PydanticOutputParser(pydantic_object=section_model)
    
    # Check if this is a list wrapper model and adjust prompt accordingly
    is_list_wrapper = section_name in ["work", "education", "skills", "projects", "awards", "publications", "languages", "certifications"]
    
    if is_list_wrapper:
        
        prompt_template = PromptTemplate(
        template=(
            "Enhance the '{section_name}' section of this resume to improve clarity, impact, and alignment with the job description.\n\n"
            "FORMAT GUIDELINES:\n{section_format}\n\n"
            "JOB DESCRIPTION:\n{job_description}\n\n"
            "ORIGINAL CONTENT:\n{original_content}\n\n"
            "{format_instructions}\n\n"
            "IMPORTANT:\n"
            "Return ONLY valid JSON. "
            + ("Response must be a JSON object with an \"items\" list.\n" if is_list_wrapper else "") +
            "Do not include explanations or extra text."
        ),
        input_variables=["section_name", "section_format", "job_description", "original_content"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
        
        # For list types, we need to modify the prompt to make it clear we're using a wrapper
        return {
        "prompt": prompt_template,
        "parser": parser
    }
    else:
        # For non-list types, use the standard prompt
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

def extract_response_text(response: Any, model: BaseLanguageModel) -> str:
    """Extract the response text based on model type."""
    if isinstance(model, GoogleGenerativeAI):
        return response.text if hasattr(response, 'text') else str(response)
    elif isinstance(model, (ChatOpenAI, ChatDeepSeek, OpenAI)):
        return response.content if hasattr(response, 'content') else str(response)
    return str(response)

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

async def enhance_resume_section(
    section_name: str,
    section_data: Any,
    model: BaseLanguageModel,
    job_description: Optional[str] = None,
    template_type: str = "simple"
) -> Dict[str, Any]:
    """
    Enhance a single section of the resume using the LLM with Pydantic validation.
    
    Args:
        section_name: Name of the section to enhance
        section_data: Current section data
        model: Language model to use
        job_description: Optional job description
        template_type: Resume template type
        
    Returns:
        Dict containing enhanced section data and token metrics
    """
    try:
        logger.info(f"[DEBUG] Enhancing section: {section_name}")
        logger.info(f"Enhancing section: {section_name} with template: {template_type}")
        
        # Create prompt with schema
        prompt_data = create_section_prompt_with_schema(
            section_name, 
            section_data, 
            job_description,
            template_type
        )
        
        # Log the prompt for debugging
        logger.debug(f"Section model prompt for {section_name}: {prompt_data['prompt']}")

        # Prepare prompt for model invocation
        prompt = prompt_data["prompt"]
        if isinstance(prompt, PromptTemplate):
            # Render the template with required variables
            section_format = get_section_format(section_name, template_type)
            # For list wrappers, section_data is wrapped as 'items', so unwrap for original_content
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
        logger.info(f"[DEBUG] Section model response for {section_name}: {response_text}")
        
        # Log the response for debugging
        logger.debug(f"Section model response for {section_name}: {response_text[:500]}")
        
        # Get token counts
        token_counts = get_token_counts_from_response(response)
        
        # Log token usage
        logger.info(f"Section: {section_name} | Input tokens: {token_counts['input_tokens']} | Output tokens: {token_counts['output_tokens']} | Total tokens: {token_counts['total_tokens']}")
        
        # Parse the enhanced content
        enhanced_section = section_data  # Default to original if parsing fails
        try:
            if prompt_data["parser"]:
                # Use Pydantic parser if available
                parsed_data = prompt_data["parser"].parse(response_text)
                
                # Convert to dict if it's a Pydantic model
                if hasattr(parsed_data, "model_dump"):
                    parsed_dict = parsed_data.model_dump()
                    
                    # For list wrapper models, extract the items
                    if section_name in ["work", "education", "skills", "projects", "awards", "publications", "languages", "certifications"]:
                        if "items" in parsed_dict:
                            enhanced_section = parsed_dict["items"]
                        else:
                            logger.warning(f"Expected 'items' field in {section_name} response, but not found")
                            enhanced_section = parsed_dict  # Use as is in this case
                    else:
                        enhanced_section = parsed_dict
                else:
                    # Not a model with model_dump - try to handle gracefully
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
            # Fall back to original data
            
        return {
            "section_data": enhanced_section, 
            "input_tokens": token_counts['input_tokens'], 
            "output_tokens": token_counts['output_tokens'],
            "total_tokens": token_counts['total_tokens']
        }
    except Exception as e:
        logger.error(f"Error enhancing section {section_name}: {str(e)}")
        return {"section_data": section_data, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

async def enhance_resume_by_sections(
    json_data: Dict[str, Any],
    model: BaseLanguageModel,
    job_description: Optional[str] = None,
    template_type: str = "simple"
) -> Dict[str, Any]:
    """
    Enhance each section of the resume asynchronously and combine results.
    
    Args:
        json_data: Resume data to enhance
        model: Language model to use
        job_description: Optional job description
        template_type: Resume template type
        
    Returns:
        Dict containing enhanced resume data with token metrics
    """
    if 'details' in json_data and isinstance(json_data['details'], dict):
        resume_data = json_data['details']
        has_details_wrapper = True
    else:
        resume_data = json_data
        has_details_wrapper = False
    
    logger.info(f"Starting resume enhancement with template_type: {template_type}")
    
    # Convert template_type to lowercase for case-insensitive comparison
    template_type = template_type.lower()
    
    enhancement_tasks = {}
    for section_name, section_data in resume_data.items():
        # Skip non-resume sections
        if section_name in ['token_metrics', 'Theme', 'JD']:
            continue
            
        task = asyncio.create_task(
            enhance_resume_section(
                section_name, 
                section_data, 
                model, 
                job_description,
                template_type
            )
        )
        enhancement_tasks[section_name] = task
    
    enhanced_sections = {}
    total_input_tokens = 0
    total_output_tokens = 0
    total_tokens = 0
    
    for section_name, task in enhancement_tasks.items():
        result = await task
        enhanced_sections[section_name] = result["section_data"]
        total_input_tokens += result["input_tokens"]
        total_output_tokens += result["output_tokens"]
        total_tokens += result.get("total_tokens", 0)

    # Log total token usage
    logger.info(f"Resume enhancement completed | Template: {template_type} | "
                f"Total input tokens: {total_input_tokens} | "
                f"Total output tokens: {total_output_tokens} | "
                f"Total tokens: {total_tokens}")
    
    enhanced_resume = enhanced_sections
    
    if has_details_wrapper:
        result = {'details': enhanced_resume}
        if 'JD' in json_data:
            result['JD'] = json_data['JD']
        if 'Theme' in json_data:
            result['Theme'] = json_data['Theme']
        result['token_metrics'] = {
            'template_type': template_type,
            'total_input_tokens': total_input_tokens,
            'total_output_tokens': total_output_tokens,
            'total_tokens': total_tokens
        }
        return result
    
    # Add token metrics to the result
    enhanced_resume['token_metrics'] = {
        'template_type': template_type,
        'total_input_tokens': total_input_tokens,
        'total_output_tokens': total_output_tokens,
        'total_tokens': total_tokens
    }
    
    return enhanced_resume

class ResumeEnhancementRequest(BaseModel):
    """Model for a resume enhancement request with validation."""
    json_data: Dict[str, Any] = Field(description="Resume data in JSON format")
    job_description: Optional[str] = Field(None, description="Optional job description to tailor the resume")
    template_type: str = Field("simple", description="Style template to use (simple or software_engineer)")
    
    @field_validator('template_type')
    @classmethod
    def validate_template_type(cls, v):
        """Validate that the template type exists."""
        v = v.lower()
        if v not in TEMPLATE_PROMPTS:
            logger.warning(f"Template type '{v}' not found in TEMPLATE_PROMPTS. Using 'simple' instead.")
            return "simple"
        return v
    
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

async def enhance_resume_with_model(
    json_data: Dict[str, Any],
    job_description: Optional[str] = None,
    template_type: str = "simple"
) -> Dict[str, Any]:
    """
    Enhance resume data using the specified LLM model, section by section.
    
    Args:
        json_data: The resume data in JSON format
        job_description: Optional job description to tailor the resume
        template_type: Style template to use (simple or software_engineer)
    
    Returns:
        Dict containing the enhanced resume data with token metrics
    """
    try:
        # Validate request with Pydantic
        request = ResumeEnhancementRequest(
            json_data=json_data,
            job_description=job_description,
            template_type=template_type
        )
        
        # Set validated values
        json_data = request.json_data
        template_type = request.template_type
        
        # Initialize model
        model = SimpleModelManager().get_model()
        
        # Get job description from request or JSON data
        jd = request.job_description
        if not jd and 'JD' in json_data:
            jd = json_data.get('JD')
        
        # Check if Theme is in json_data and use it if template_type wasn't specified
        if template_type == "simple" and "Theme" in json_data:
            theme = json_data["Theme"].lower()
            if theme in TEMPLATE_PROMPTS:
                template_type = theme
            else:
                logger.warning(f"Theme '{theme}' in JSON not found in TEMPLATE_PROMPTS")
            
        # Log the template type being used
        logger.info(f"Using template_type: {template_type} for resume enhancement")
            
        enhanced_json = await enhance_resume_by_sections(
            json_data, 
            model, 
            jd,
            template_type
        )
        
        logger.info("Resume enhancement completed successfully")
        return enhanced_json
        
    except ValueError as e:
        # Handle validation errors
        logger.error(f"Resume enhancement validation failed: {str(e)}")
        if 'token_metrics' not in json_data:
            json_data['token_metrics'] = {
                'template_type': template_type,
                'total_input_tokens': 0,
                'total_output_tokens': 0,
                'total_tokens': 0,
                'validation_error': str(e)
            }
        return json_data
    except Exception as e:
        # Handle other errors
        logger.error(f"Resume enhancement failed: {str(e)}")
        # Return original data with token metrics to ensure consistent response structure
        if 'token_metrics' not in json_data:
            json_data['token_metrics'] = {
                'template_type': template_type,
                'total_input_tokens': 0,
                'total_output_tokens': 0,
                'total_tokens': 0,
                'error': str(e)
            }
        return json_data    

class ProcessResumeRequest(BaseModel):
    """Model for validating a process resume request."""
    json_data: Dict[str, Any] = Field(description="Resume data in JSON format")
    job_description: Optional[str] = Field(None, description="Optional job description to tailor the resume")
    template_type: str = Field("simple", description="Style template to use")
    
    @field_validator('template_type')
    @classmethod
    def validate_template_type(cls, v):
        """Validate and normalize template type."""
        v = v.lower()
        allowed_templates = list(TEMPLATE_PROMPTS.keys())
        if v not in allowed_templates:
            logger.warning(f"Template type '{v}' not allowed. Must be one of {allowed_templates}. Using 'simple'.")
            return "simple"
        return v
    
    model_config = {
        "extra": "ignore"
    }

async def process_resume(
    json_data: Dict[str, Any],
    job_description: Optional[str] = None,
    template_type: str = "simple"
) -> Dict[str, Any]:
    """
    Process a resume by enhancing it section by section with Pydantic validation.
    
    Args:
        json_data: The resume data in JSON format
        job_description: Optional job description to tailor the resume
        template_type: Style template to use (simple or software_engineer)
        
    Returns:
        Dict containing the enhanced resume data
    """
    try:
        # Extract Theme from JSON if present
        if "Theme" in json_data:
            theme_from_json = json_data["Theme"].lower()
            if theme_from_json in TEMPLATE_PROMPTS:
                template_type = theme_from_json
                logger.info(f"Using theme from JSON: {template_type}")
                
        # Validate request
        request = ProcessResumeRequest(
            json_data=json_data,
            job_description=job_description,
            template_type=template_type
        )
        
        # Process with validated data
        enhanced_json = await enhance_resume_with_model(
            request.json_data,
            request.job_description,
            request.template_type
        )
        
        return enhanced_json
        
    except Exception as e:
        logger.error(f"Error processing resume: {str(e)}")
        # Return original data with error info
        if 'token_metrics' not in json_data:
            json_data['token_metrics'] = {
                'template_type': template_type,
                'total_input_tokens': 0,
                'total_output_tokens': 0,
                'total_tokens': 0,
                'process_error': str(e)
            }
        return json_data