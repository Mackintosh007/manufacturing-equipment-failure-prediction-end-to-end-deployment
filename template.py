import os
from pathlib import Path


PROJECT_STRUCTURE = {
    'src': {
        '__init__.py': '',
        'exception.py': '# content for exception.py',
        'logger.py': '# content for logger.py',
        'utils.py': '# content for utils.py',
        'components': {
            '__init__.py': '',
            'data_ingestion.py': '# content for data_ingestion.py',
            'data_transformation.py': '# content for data_transformation.py',
            'model_trainer.py': '# content for model_trainer.py'
        },
        'pipelines': {
            '__init__.py': '',
            'predict_pipeline.py': '# content for predict_pipeline.py',
            'train_pipeline.py': '# content for train_pipeline.py'
        }
    },
    'data': {},
    '.': {
        '.gitignore': '# content for .gitignore',
        'setup.py': '# content for setup.py',
        'requirements.txt': '# content for requirements.txt',
        'README.md': '# content for README.md'
    }
}

def create_project_structure(base_path, structure):
    """
    Recursively creates folders and files based on the given structure dictionary.
    """
    for name, content in structure.items():
        current_path = os.path.join(base_path, name)

        if isinstance(content, dict):
            # It's a folder, so create it and recurse
            try:
                os.makedirs(current_path, exist_ok=True)
                print(f"Created directory: {current_path}")
                create_project_structure(current_path, content)
            except OSError as e:
                print(f"Error creating directory {current_path}: {e}")
        else:
            # It's a file, so create it and write content
            try:
                with open(current_path, 'w') as f:
                    f.write(content)
                print(f"Created file: {current_path}")
            except OSError as e:
                print(f"Error creating file {current_path}: {e}")

if __name__ == '__main__':
    project_root = Path(__file__).parent
    create_project_structure(project_root, PROJECT_STRUCTURE)