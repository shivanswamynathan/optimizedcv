# Resume processor models package
# This makes the directory a proper Python package

from .resume_models import (
    # Base section models
    Location, Profile, Summary,
    ExperienceItem, EducationItem, ProjectItem, SkillCategory,
    Award, Publication, Language, Certification,
    
    # List wrapper models
    ExperienceList, EducationList, ProjectList, SkillList,
    AwardList, PublicationList, LanguageList, CertificationList,
    
    # Combined model
    ResumeSection,
    
    # Request validation models
    ResumeEnhancementRequest, ProcessResumeRequest,

    # Generic Pydantic models for query analysis
    SpecificChanges,ModificationRequest,QueryAnalysis,

    
    # Configuration dictionaries
    SECTION_MODELS, SECTION_GUIDELINES 
)

__all__ = [
    # Base models
    'Location', 'Profile', 'Summary',
    'ExperienceItem', 'EducationItem', 'ProjectItem', 'SkillCategory',
    'Award', 'Publication', 'Language', 'Certification',
    
    # List wrapper models
    'ExperienceList', 'EducationList', 'ProjectList', 'SkillList',
    'AwardList', 'PublicationList', 'LanguageList', 'CertificationList',
    
    # Combined model
    'ResumeSection',
    
    # Request validation models
    'ResumeEnhancementRequest', 'ProcessResumeRequest',

    # Generic Pydantic models for query analysis
    'SpecificChanges','ModificationRequest','QueryAnalysis',
    
    # Configuration
    'SECTION_MODELS', 'SECTION_GUIDELINES' 
]