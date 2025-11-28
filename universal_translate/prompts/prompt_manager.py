"""Prompt management system with external configuration support."""

import json
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any
from string import Template


class PromptTemplate:
    """Template for rendering prompts with variables."""

    def __init__(self, template: str, safe: bool = True):
        """
        Initialize prompt template.

        Args:
            template: Template string with {variables}
            safe: If True, use safe_substitute (missing vars → placeholder)
        """
        self.template = template
        self.safe = safe
        self._compiled = Template(template)

    def render(self, **kwargs) -> str:
        """
        Render template with variables.

        Args:
            **kwargs: Variables to substitute

        Returns:
            Rendered template string
        """
        if self.safe:
            return self._compiled.safe_substitute(**kwargs)
        return self._compiled.substitute(**kwargs)

    def required_variables(self) -> List[str]:
        """
        Extract required template variables.

        Returns:
            List of variable names
        """
        import re
        pattern = r'\$\{?(\w+)\}?'
        return list(set(re.findall(pattern, self.template)))


class PromptManager:
    """
    Manages prompts and examples loaded from external configuration files.

    Supports YAML and JSON formats with dynamic template rendering.
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize prompt manager.

        Args:
            config_path: Path to prompt configuration file (YAML or JSON)
        """
        self.config_path = Path(config_path) if config_path else None
        self.config: Dict[str, Any] = {}
        self.examples: List[Dict[str, str]] = []
        self._templates: Dict[str, PromptTemplate] = {}

        if self.config_path and self.config_path.exists():
            self.load_config(self.config_path)

    def load_config(self, config_path: Path):
        """
        Load prompt configuration from file.

        Args:
            config_path: Path to YAML or JSON file
        """
        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Prompt config not found: {config_path}")

        # Load based on extension
        if config_path.suffix in ['.yaml', '.yml']:
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
        elif config_path.suffix == '.json':
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {config_path.suffix}")

        # Compile templates
        if 'system_prompt' in self.config:
            self._templates['system'] = PromptTemplate(self.config['system_prompt'])

        if 'user_prompt' in self.config:
            self._templates['user'] = PromptTemplate(self.config['user_prompt'])

        # Load examples if specified
        if 'examples_file' in self.config:
            examples_path = config_path.parent / self.config['examples_file']
            if examples_path.exists():
                self.load_examples(examples_path)

    def load_examples(self, examples_path: Path):
        """
        Load translation examples from file.

        Args:
            examples_path: Path to JSON file with examples
        """
        examples_path = Path(examples_path)

        with open(examples_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if 'examples' in data:
            self.examples = data['examples']
        else:
            self.examples = data if isinstance(data, list) else []

    def get_system_prompt(self, **kwargs) -> Optional[str]:
        """
        Get rendered system prompt.

        Args:
            **kwargs: Variables for template rendering

        Returns:
            Rendered system prompt or None
        """
        if 'system' not in self._templates:
            return None

        return self._templates['system'].render(**kwargs)

    def get_user_prompt(self, text: str, **kwargs) -> str:
        """
        Get rendered user prompt.

        Args:
            text: Text to translate
            **kwargs: Additional variables for template rendering

        Returns:
            Rendered user prompt
        """
        if 'user' not in self._templates:
            # Fallback: simple prompt
            return f"Translate the following text to {kwargs.get('target_lang', 'the target language')}:\n\n{text}"

        return self._templates['user'].render(text=text, **kwargs)

    def get_examples(self, max_examples: Optional[int] = None) -> List[Dict[str, str]]:
        """
        Get translation examples.

        Args:
            max_examples: Maximum number of examples to return

        Returns:
            List of example dictionaries
        """
        if max_examples is not None:
            return self.examples[:max_examples]
        return self.examples

    def format_examples_for_prompt(
        self,
        format_template: Optional[str] = None,
        max_examples: Optional[int] = None
    ) -> str:
        """
        Format examples as string for inclusion in prompts.

        Args:
            format_template: Template for each example
                           Default: "Source: {source}\nTarget: {target}\n"
            max_examples: Maximum examples to include

        Returns:
            Formatted examples string
        """
        if not self.examples:
            return ""

        if format_template is None:
            format_template = "Source ({source_lang}): {source}\nTarget ({target_lang}): {target}\n"

        examples = self.get_examples(max_examples)
        template = PromptTemplate(format_template)

        formatted = []
        for i, ex in enumerate(examples, 1):
            # Add example number
            ex_with_num = {**ex, 'num': i}
            formatted.append(template.render(**ex_with_num))

        return "\n".join(formatted)

    def supports_caching(self) -> bool:
        """
        Check if this prompt configuration supports caching.

        Returns:
            True if prompt caching is enabled
        """
        return self.config.get('use_prompt_caching', False)

    def get_parameters(self) -> Dict[str, Any]:
        """
        Get model parameters from configuration.

        Returns:
            Dictionary of model parameters (temperature, max_tokens, etc.)
        """
        return self.config.get('parameters', {})

    def get_metadata(self) -> Dict[str, Any]:
        """
        Get prompt metadata.

        Returns:
            Dictionary with name, description, etc.
        """
        return {
            'name': self.config.get('name', 'unknown'),
            'description': self.config.get('description', ''),
            'num_examples': len(self.examples),
            'supports_caching': self.supports_caching(),
            'parameters': self.get_parameters()
        }

    def __repr__(self) -> str:
        """String representation."""
        name = self.config.get('name', 'unnamed')
        num_ex = len(self.examples)
        return f"PromptManager('{name}', {num_ex} examples)"
