import logging
import json
import os
from datetime import datetime
from typing import Dict, Any, Optional
import tiktoken

class LLMLogger:
    def __init__(self, log_dir: str = None):
        """Initialize LLM logger with specified log directory"""
        if log_dir is None:
            log_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'logs')
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # Set up file handler
        log_file = os.path.join(log_dir, f"llm_interactions_{datetime.now().strftime('%Y%m%d')}.log")
        
        # Configure logging
        self.logger = logging.getLogger("LLMLogger")
        self.logger.setLevel(logging.INFO)
        
        # Create file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        
        # Add handler to logger
        self.logger.addHandler(file_handler)
        
    def _count_tokens(self, text: str) -> int:
        """Estimate token count using tiktoken"""
        try:
            encoding = tiktoken.get_encoding("cl100k_base")
            return len(encoding.encode(text))
        except Exception:
            # Fallback to approximate token count if tiktoken fails
            return len(text.split()) * 1.3
    
    def _format_log_entry(self, 
                         model_name: str,
                         input_text: str,
                         output_text: str,
                         input_tokens: int,
                         output_tokens: int,
                         metadata: Optional[Dict[str, Any]] = None) -> str:
        """Format log entry as JSON string"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "model_name": model_name,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "input_text": input_text[:1000] + "..." if len(input_text) > 1000 else input_text,
            "output_text": output_text[:1000] + "..." if len(output_text) > 1000 else output_text,
            "metadata": metadata or {}
        }
        return json.dumps(log_entry)
    
    def log_interaction(self,
                       model_name: str,
                       input_text: str,
                       output_text: str,
                       metadata: Optional[Dict[str, Any]] = None) -> None:
        """Log an LLM interaction with token counts"""
        try:
            # Count tokens
            input_tokens = self._count_tokens(input_text)
            output_tokens = self._count_tokens(output_text)
            
            # Format and write log entry
            log_entry = self._format_log_entry(
                model_name=model_name,
                input_text=input_text,
                output_text=output_text,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                metadata=metadata
            )
            
            self.logger.info(log_entry)
            
        except Exception as e:
            self.logger.error(f"Error logging LLM interaction: {str(e)}")

    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics from the log file"""
        stats = {
            "total_interactions": 0,
            "total_tokens": 0,
            "models": {},
            "average_tokens_per_request": 0
        }
        
        try:
            log_file = os.path.join(self.log_dir, f"llm_interactions_{datetime.now().strftime('%Y%m%d')}.log")
            
            if not os.path.exists(log_file):
                return stats
                
            with open(log_file, 'r') as f:
                for line in f:
                    try:
                        # Extract JSON from log line
                        log_data = json.loads(line.split(" | ")[-1])
                        
                        stats["total_interactions"] += 1
                        stats["total_tokens"] += log_data["total_tokens"]
                        
                        if log_data["model_name"] not in stats["models"]:
                            stats["models"][log_data["model_name"]] = {
                                "interactions": 0,
                                "total_tokens": 0
                            }
                            
                        stats["models"][log_data["model_name"]]["interactions"] += 1
                        stats["models"][log_data["model_name"]]["total_tokens"] += log_data["total_tokens"]
                        
                    except json.JSONDecodeError:
                        continue
                        
            if stats["total_interactions"] > 0:
                stats["average_tokens_per_request"] = stats["total_tokens"] / stats["total_interactions"]
                
            return stats
            
        except Exception as e:
            self.logger.error(f"Error getting usage stats: {str(e)}")
            return stats