import os
import logging
from configparser import ConfigParser
from typing import Dict, Optional
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_deepseek import ChatDeepSeek
from django.conf import settings
model_name = settings.MODEL_NAME
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()


MODEL_CONFIGS = {
    'gemini': {
        'model_name': 'gemini-1.5-pro',
        'temperature': 0.5,
        'max_tokens': 4096
    },
    'openai': {
        'model_name': 'gpt-4o',
        'temperature': 0.5,
        'max_tokens': 4096
    },
    'deepseek': {
        'model_name': 'deepseek-chat',
        'temperature': 0.6,
        'max_tokens': 4096
    }
}

def get_api_key(model_type: str) -> str:
    """Get API key for the specified model type"""
    api_keys = {
        'gemini': os.getenv('GEMINI_API_KEY'),
        'openai': os.getenv('OPENAI_API_KEY'),
        'deepseek': os.getenv('DEEPSEEK_API_KEY')
    }
    
    api_key = api_keys.get(model_type)
    if not api_key:
        raise ValueError(
            f"Missing API key for {model_type}. "
            f"Please set {model_type.upper()}_API_KEY in your .env file"
        )
    return api_key

def create_gemini_model(config: Optional[Dict] = None):
    """Create and return a Gemini model instance"""
    config = config or MODEL_CONFIGS['gemini']
    api_key = get_api_key('gemini')
    
    import google.generativeai as genai
    genai.configure(api_key=api_key)
    
    return GoogleGenerativeAI(
        model=config['model_name'],
        temperature=config.get('temperature', 1.0)
    )

def create_openai_model(config: Optional[Dict] = None):
    """Create and return an OpenAI model instance"""
    config = config or MODEL_CONFIGS['openai']
    api_key = get_api_key('openai')
    
    return ChatOpenAI(
        api_key=api_key,
        model_name=config['model_name'],
        temperature=config.get('temperature', 1.0),
        max_tokens=config.get('max_tokens', 4000),
        streaming=False
    )

def create_deepseek_model(config: Optional[Dict] = None):
    """Create and return a DeepSeek model instance"""
    config = config or MODEL_CONFIGS['deepseek']
    api_key = get_api_key('deepseek')
    
    return ChatDeepSeek(
        api_key=api_key,
        **config
    )

class SimpleModelManager:
    """A simplified model manager that creates models on demand"""
    
    def __init__(self):
        self.current_model_type = model_name
        self.current_model = None
    
    def get_model(self, model_type: Optional[str] = None):
        """Get a model instance of the specified type"""
        model_type = model_type or self.current_model_type
        model_type = model_type.lower()
        
        if model_type == 'gemini':
            self.current_model = create_gemini_model()
        elif model_type == 'openai':
            self.current_model = create_openai_model()
        elif model_type == 'deepseek':
            self.current_model = create_deepseek_model()
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        self.current_model_type = model_type
        logger.info(f"Using model: {model_type}")
        return self.current_model
    
    def switch_model(self, model_type: str):
        """Switch to a different model type"""
        if model_type not in MODEL_CONFIGS:
            raise ValueError(f"Invalid model type: {model_type}")
        
        get_api_key(model_type)
        
        self.current_model_type = model_type
        self.current_model = None  
        logger.info(f"Switched to model: {model_type}")
        
        return self.get_model()

def init_directories():
    """Initialize necessary directories"""
    base_dir = os.path.abspath(os.path.dirname(__file__))
    output_folder = os.path.join(base_dir, 'temp', 'output')
    upload_folder = os.path.join(base_dir, 'uploads')
    
    for directory in [output_folder, upload_folder]:
        if not os.path.exists(directory):
            os.makedirs(directory)
            logger.info(f"Created directory: {directory}")
    
    return {
        'BASE_DIR': base_dir,
        'OUTPUT_FOLDER': output_folder,
        'UPLOAD_FOLDER': upload_folder,
        'BASE_YAML_PATH': os.path.join(base_dir, 'resume.yaml')
    }

def initialize_app():
    """Initialize the application and return a model manager"""
    try:
        init_directories()
        
        model_manager = SimpleModelManager()
        
        model_manager.get_model()
        
        return model_manager
    except Exception as e:
        logger.error(f"Error during initialization: {str(e)}")
        raise

# # Example usage
# if __name__ == "__main__":
#     # Initialize app and get model manager
#     manager = initialize_app()
    
#     # Get default model
#     model = manager.get_model()
#     print(f"Using model: {manager.current_model_type}")
#     response = model.invoke("Hello, world!")
#     print(response)
#     # Switch to a different model
#     model = manager.switch_model("openai")
#     print(f"Switched to: {manager.current_model_type}")