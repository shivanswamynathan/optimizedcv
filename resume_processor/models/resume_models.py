from typing import Dict, Any, Optional, List, Union, Literal
from pydantic import BaseModel, Field, field_validator, model_validator

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
        if not v:
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
    description: list[str] = Field(description="Description of the publication.")

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

# Validation models
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

class ProcessResumeRequest(BaseModel):
    """Model for validating a process resume request."""
    json_data: Dict[str, Any] = Field(description="Resume data in JSON format")
    job_description: Optional[str] = Field(None, description="Optional job description to tailor the resume")
    
    model_config = {
        "extra": "ignore"
    }

# Dictionary mapping section names to appropriate models
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

# Generic Pydantic models for query analysis
class SpecificChanges(BaseModel):
    """Model for specific changes in a modification request - generic for all sections."""
    field: Optional[str] = Field(None, description="The specific field to modify (e.g., phone, summary, name, title)")
    old_value: Optional[str] = Field(None, description="The old value to replace")
    new_value: Optional[str] = Field(None, description="The new value to use")
    identifier: Optional[str] = Field(None, description="Generic identifier (company name, institution, project name, skill category, etc.)")
    search_term: Optional[str] = Field(None, description="Term to search for when finding specific entries")
    enhancement_focus: Optional[str] = Field(None, description="Focus area for enhancements (technical, leadership, impact, etc.)")
    data: Optional[Dict[str, Any]] = Field(None, description="New data to add for any section")
    position: Optional[int] = Field(None, description="Position to insert new item (0 for beginning, -1 for end)")

class ModificationRequest(BaseModel):
    """Model for a single modification request - works for all sections."""
    section: Literal["basics", "work", "education", "skills", "projects", "awards", "languages", "publications", "certifications"] = Field(
        description="The resume section to modify"
    )
    action: Literal["modify_existing", "add_new", "enhance_content", "delete_item"] = Field(
        description="The type of action to perform"
    )
    description: str = Field(description="Human-readable description of the modification")
    specific_changes: Optional[SpecificChanges] = Field(None, description="Specific details about the changes")

class QueryAnalysis(BaseModel):
    """Model for the complete query analysis."""
    modifications: List[ModificationRequest] = Field(
        description="List of all modification requests identified in the query"
    )

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