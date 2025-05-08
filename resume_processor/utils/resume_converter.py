import json
import re
import logging
from collections import OrderedDict
from typing import Dict, List, Optional, Union
from datetime import datetime


# Get logger for this module
logger = logging.getLogger(__name__)

class JSONResumeConverter:
    def __init__(self, json_resume: Dict):
        self.render_cv = OrderedDict()
        self.process_basics(json_resume.get("basics", {}))
        sections = self._build_sections(json_resume)
        if sections:
            self.render_cv["sections"] = sections

    def format_phone_number(self, phone_str: str) -> Optional[str]:
        """
        Formats a phone number string to meet RenderCV requirements.
        Returns in format: '+1 1234567890'
        """
        if not phone_str:
            return None
            
        # Remove all non-numeric characters except '+'
        cleaned = ''.join(char for char in str(phone_str) if char.isdigit() or char == '+')
        
        # If no '+' prefix, assume US number and add +1
        if not cleaned.startswith('+'):
            numbers_only = ''.join(filter(str.isdigit, cleaned))
            if len(numbers_only) == 10:
                return f'+1 {numbers_only}'
            return None
            
        # If already has '+' prefix, just add space after country code
        match = re.match(r'^\+(\d{1,3})(\d{10})$', cleaned)
        if match:
            country_code, number = match.groups()
            return f'+{country_code} {number}'
            
        return None

    def validate_and_format_website(self, url: str) -> Optional[str]:
        """
        Validate and format website URL. Returns None if no valid URL provided.
        """
        if not url:
            return None
        
        # Ensure URL has a scheme
        if not url.startswith(('http://', 'https://')):
            url = f'https://{url}'
        
        # URL validation pattern
        url_pattern = re.compile(
            r'^https?://'
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'
            r'localhost|'
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'
            r'(?::\d+)?'
            r'(?:/?|[/?]\S+)$', re.IGNORECASE)
        
        return url if url_pattern.match(url) else None

    def process_basics(self, basics: Dict) -> None:
        """Process basic information from JSON resume"""
        if basics.get("name"):
            self.render_cv["name"] = str(basics["name"])
    
        location = self._format_location(basics.get("location", {}))
        if location:
            self.render_cv["location"] = location
        
        if basics.get("email"):
            self.render_cv["email"] = str(basics["email"])
        
        if basics.get("phone"):
            formatted_phone = self.format_phone_number(basics["phone"])
            if formatted_phone:
                self.render_cv["phone"] = formatted_phone
            else:
                self.render_cv["phone"] = str(basics["phone"])
        
        # Handle website URL - only include if provided in JSON
        if basics.get("url"):
            website = self.validate_and_format_website(basics["url"])
            if website:
                self.render_cv["website"] = website
    
        profiles = basics.get("profiles", [])
        if profiles:
            social_networks = self._format_social_networks(profiles)
            if social_networks:
                self.render_cv["social_networks"] = social_networks

    def _validate_date(self, date_str: str) -> Optional[str]:
        """
        Validates and formats date strings with more flexible input handling
        """
        if not date_str:
            return None
        
        if str(date_str).lower() == 'present':
            return 'present'
    
        # Handle month year format
        try:
            # Try various date formats
            for fmt in ['%Y-%m', '%B %Y', '%b %Y']:
                try:
                    parsed_date = datetime.strptime(str(date_str), fmt)
                    return parsed_date.strftime('%Y-%m')
                except ValueError:
                    continue
                
            # If no format matches, extract year and month if possible
            match = re.search(r'(\w+)\s*(\d{4})', str(date_str))
            if match:
                month, year = match.groups()
                try:
                    parsed_date = datetime.strptime(f"{month} {year}", "%B %Y")
                    return parsed_date.strftime('%Y-%m')
                except ValueError:
                    try:
                        parsed_date = datetime.strptime(f"{month} {year}", "%b %Y")
                        return parsed_date.strftime('%Y-%m')
                    except ValueError:
                        return None
        except Exception:
                return None
    
        return None

    def _format_location(self, location: Dict) -> Optional[str]:
        """
        Format location string with more flexible handling
        """
        if not location:
            return None
        
        if isinstance(location, str):
            return location
        
        if not isinstance(location, dict):
            return None
        
        components = []
        if location.get('city'):
            components.append(location['city'])
        if location.get('region'):
            components.append(location['region'])
        if location.get('countryCode'):
            components.append(location['countryCode'])
    
        return ', '.join(components) if components else None

    def _format_social_networks(self, profiles: List[Dict]) -> Optional[List[Dict[str, str]]]:
        """
        Format social network profiles with proper network naming and URL handling.
        Excludes profiles that don't match known social networks.
        """
        if not profiles:
            return None
    
        formatted_profiles = []
        networks_mapping = {
            "linkedin": "LinkedIn",
            "github": "GitHub",
            "twitter": "Twitter",
            "default": "Other"
        }
    
        for profile in profiles:
            if not isinstance(profile, dict):
                continue
        
            network = profile.get("network", "").lower()
            username = profile.get("username", "")
            url = profile.get("url", "").strip()
        
            # Skip if no username
            if not username:
                continue
        
        # Determine network name
            network_name = networks_mapping.get(network, networks_mapping["default"])
        
        # Only add if network is not "Other"
            if network_name != "Other":
                formatted_profiles.append({
                    "network": network_name,
                    "username": username
                })
    
        return formatted_profiles if formatted_profiles else None
    
    
    def _build_summary(self, json_resume: Dict) -> Optional[List[str]]:
        """Build summary section"""
        summary = json_resume.get("basics", {}).get("summary")
        return [str(summary)] if summary else None
    
    def _build_experience(self, json_resume: Dict) -> Optional[List[Dict]]:
        """Build experience section according to template requirements"""
        work = json_resume.get("work", [])
        if not work:
            return None
            
        experience = []
        for job in work:
            if not isinstance(job, dict):
                continue
                
            if not job.get("name") or not job.get("position"):
                continue
                
            entry = {
                "company": str(job["name"]),
                "position": str(job["position"])
            }
            
            start_date = self._validate_date(job.get("startDate"))
            if start_date:
                entry["start_date"] = start_date
            else:
                continue  # Skip entries without valid start date
                
            if job.get("location"):
                entry["location"] = str(job["location"])
                
            end_date = self._validate_date(job.get("endDate"))
            if end_date:
                entry["end_date"] = end_date
                
            highlights = job.get("highlights", [])
            if highlights and isinstance(highlights, list):
                entry["highlights"] = [str(h) for h in highlights if h]
                
            experience.append(entry)
            
        return experience if experience else None
    
    def _build_education(self, json_resume: Dict) -> Optional[List[Dict]]:
        """Build education section according to template requirements"""
        education_list = json_resume.get("education", [])
        if not education_list:
            return None
        
        education = []
        for edu in education_list:
            if not isinstance(edu, dict):
                continue
            
        # Check for required fields
            required_fields = ["institution", "area"]
            if not all(edu.get(field) for field in required_fields):
                continue
            
            entry = {
                "institution": str(edu["institution"]).strip(),
                "area": str(edu["area"]).strip(),
            }
        
        # Handle degree/studyType field
            if edu.get("studyType"):
                entry["degree"] = str(edu["studyType"]).strip()
            elif edu.get("degree"):  # Add fallback to "degree" field
                entry["degree"] = str(edu["degree"]).strip()
            else:
                continue  # Skip entries without degree information
            
            start_date = self._validate_date(edu.get("startDate"))
            if start_date:
                entry["start_date"] = start_date
            else:
                continue  # Skip entries without valid start date
            
            end_date = self._validate_date(edu.get("endDate"))
            if end_date:
                entry["end_date"] = end_date
            elif edu.get("endDate") and str(edu["endDate"]).lower() == "present":
                entry["end_date"] = "present"
            
            courses = edu.get("courses", [])
            if courses and isinstance(courses, list):
                entry["highlights"] = [str(c) for c in courses if c]
            
        # Clean any extra whitespace from all string fields
            for key, value in entry.items():
                if isinstance(value, str):
                    entry[key] = ' '.join(value.split())
        
            education.append(entry)
        
        return education if education else None
    
    def _build_projects(self, json_resume: Dict) -> Optional[List[Dict]]:
        """Build projects section according to template requirements"""
        projects = json_resume.get("projects", [])
        if not projects:
            return None
            
        formatted_projects = []
        for proj in projects:
            if not isinstance(proj, dict) or not proj.get("name"):
                continue
                
            entry = {"name": str(proj["name"])}
            
            start_date = self._validate_date(proj.get("startDate"))
            if start_date:
                entry["date"] = start_date
                
            if proj.get("description"):
                entry["highlights"] = [str(item) for item in proj["description"]] if isinstance(proj["description"], list) else [str(proj["description"])]
                
            if entry.get("date") or entry.get("highlights"):
                formatted_projects.append(entry)
                
        return formatted_projects if formatted_projects else None
    
    def _build_publications(self, json_resume: Dict) -> Optional[List[Dict]]:
        """
        Build publications section with proper error handling and field validation.
        Ensures all required fields are present and formats are correct.
    
        Args:
            json_resume (Dict): The input JSON resume data
        
        Returns:
            Optional[List[Dict]]: Formatted publications list or None if no valid publications
        """
        publications = json_resume.get("publications", [])
        if not publications:
            return None
        
        formatted_publications = []
        for pub in publications:
            try:
                if not isinstance(pub, dict):
                    continue

                # Skip if no title/name
                if not pub.get("name") and not pub.get("title"):
                    continue
                
                # Create publication entry with required fields
                entry = {
                    "title": str(pub.get("name") or pub.get("title")),
                    # Ensure authors is always a list, default to empty list if missing
                    "authors": [str(author) for author in pub.get("authors", [])] or ["Anonymous"]
                }
            
                #Handle publication date
                if pub.get("releaseDate"):
                    pub_date = self._validate_date(pub["releaseDate"])
                    if pub_date:
                        entry["date"] = pub_date
                elif pub.get("date"):
                    pub_date = self._validate_date(pub["date"])
                    if pub_date:
                        entry["date"] = pub_date
                    
                # Optional fields
                if pub.get("publisher"):
                    entry["journal"] = str(pub["publisher"])
                
                if pub.get("url"):
                    url = self.validate_and_format_website(pub["url"])
                    if url:
                        entry["url"] = url
                    
                if pub.get("doi"):
                    entry["doi"] = str(pub["doi"])
                
                # Only add publications that have all required fields
                if self._validate_publication_entry(entry):
                    formatted_publications.append(entry)
                
            except Exception as e:
                logger.warning(f"Error processing publication: {str(e)}")
                continue
            
        return formatted_publications if formatted_publications else None
    
    def _build_awards(self, json_resume: Dict) -> Optional[List[Dict]]:
        """Build awards section according to template requirements"""
        awards = json_resume.get("awards", [])
        if not awards:
            return None
            
        formatted_awards = []
        for award in awards:
            if not isinstance(award, dict):
                continue
                
            if not award.get("title") or not award.get("awarder"):
                continue
                
            formatted_awards.append({
                "label": str(award["title"]),
                "details": str(award["awarder"])
            })
            
        return formatted_awards if formatted_awards else None
    
    def _build_technologies(self, json_resume: Dict) -> Optional[List[Dict]]:
        """Build technologies section according to template requirements"""
        skills = json_resume.get("skills", [])
        if not skills:
            return None
            
        formatted_skills = []
        for skill in skills:
            if not isinstance(skill, dict):
                continue
                
            name = skill.get("name")
            keywords = skill.get("keywords", [])
            
            if not name or not keywords or not isinstance(keywords, list):
                continue
                
            formatted_skills.append({
                "label": str(name),
                "details": ", ".join(str(k) for k in keywords if k)
            })
            
        return formatted_skills if formatted_skills else None

    def _remove_empty_values(self, data: Union[Dict, List, str, None]) -> Union[Dict, List, str, None]:
        """
        Recursively removes empty values from a dictionary or list
        """
        if isinstance(data, dict):
            return {
                key: self._remove_empty_values(value)
                for key, value in data.items()
                if value not in (None, "", [], {}, OrderedDict())
                and self._remove_empty_values(value) not in (None, "", [], {}, OrderedDict())
            }
        elif isinstance(data, (list, tuple)):
            return [
                self._remove_empty_values(item)
                for item in data
                if item not in (None, "", [], {}, OrderedDict())
                and self._remove_empty_values(item) not in (None, "", [], {}, OrderedDict())
            ]
        return data
    
    def _build_experience(self, json_resume: Dict) -> Optional[List[Dict]]:
        """Build experience section according to template requirements"""
        work = json_resume.get("work", [])
        if not work:
            return None
            
        experience = []
        for job in work:
            if not isinstance(job, dict):
                continue
                
            if not job.get("name") or not job.get("position"):
                continue

            entry = {
                "company": str(job["name"]),
                "position": str(job["position"])
            }
            
            start_date = self._validate_date(job.get("startDate"))
            if not start_date:
                logger.warning(f"Invalid start date for job: {job.get('name')}")
                continue
        
            entry["start_date"] = start_date
                
            if job.get("location"):
                entry["location"] = str(job["location"])
                
            end_date = self._validate_date(job.get("endDate"))
            if end_date:
                entry["end_date"] = end_date
                
            highlights = job.get("highlights", [])
            if highlights and isinstance(highlights, list):
                entry["highlights"] = [str(h) for h in highlights if h]
                
            experience.append(entry)
            
        return experience if experience else None
    def _build_sections(self, json_resume: Dict) -> Optional[Dict]:
        """Build all resume sections according to template requirements"""
        sections = OrderedDict()
    
    # Define section building order
        section_builders = [
            ("summary", self._build_summary),
            ("experience", self._build_experience),
            ("education", self._build_education),
            ("projects", self._build_projects),
            ("publications", self._build_publications),
            ("awards", self._build_awards),
            ("technologies", self._build_technologies)
    ]
    
        for section_name, builder_func in section_builders:
            try:
                result = builder_func(json_resume)
                if result:
                    # Remove any empty values before adding to sections
                    cleaned_result = self._remove_empty_values(result)
                    if cleaned_result:
                        sections[section_name] = cleaned_result
                        logger.debug(f"Added section {section_name}: {json.dumps(cleaned_result, indent=2)}")
            except Exception as e:
                logger.error(f"Error building section {section_name}: {str(e)}")
                # Don't continue here - we want to process other sections
            
        return sections if sections else None

    def _validate_publication_entry(self, entry: Dict) -> bool:
        """
         Validates a publication entry to ensure it has the minimum required fields.
         """
    # Check for required fields
        if not entry.get("title"):
            return False
        
    # Ensure authors is a non-empty list
        authors = entry.get("authors", [])
        if not isinstance(authors, list) or not authors:
            return False
        
    # At least one of date, journal, url, or doi should be present
        optional_fields = ["date", "journal", "url", "doi"]
        if not any(entry.get(field) for field in optional_fields):
            return False
        
        return True

    def convert(self) -> Dict:
        """Convert JSON resume to RenderCV format"""
        try:
            # Clean the entire render_cv dictionary before returning
            cleaned_cv = self._remove_empty_values(self.render_cv)
            logger.debug(f"Final CV data: {json.dumps(cleaned_cv, indent=2)}")
            return cleaned_cv
        except Exception as e:
            logger.error(f"Error in convert method: {str(e)}")
            return self.render_cv  # Return original if cleaning fails