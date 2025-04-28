# CodeBase Visualizer

Turn your codebase into stunning visual representations with AI-powered image generation.

![image](https://github.com/user-attachments/assets/c1688fc7-a7fd-4921-b19d-d6f77244b593)

![Codebase Visualization Example](./generated_images/codebase_ecosystem.png)

## Overview

CodeBase Visualizer analyzes your code and transforms it into beautiful, informative visualizations using Stable Diffusion. This tool parses your codebase structure, extracts meaningful relationships, and generates images that represent your code as various metaphorical ecosystems - from futuristic cityscapes to holographic maps to biodigital forests.

## Features

- **Automatic Code Analysis**: Parses class hierarchies, methods, and relationships in .NET/C# codebases
- **Customizable Visualization Styles**: Multiple visualization themes including:
  - Biodigital Forest Ecosystem
  - Neural Metropolis (Cyberpunk City)
  - Data Crystal Matrix
  - Futuristic Holographic Map
- **Relationship Modeling**: Visualizes code dependencies and interactions
- **GPU-Accelerated**: Utilizes CUDA for high-quality image generation

## Requirements

- Python 3.7+
- PyTorch with CUDA support
- Diffusers library
- NVIDIA GPU with CUDA support (recommended)

## Installation

```bash
# Clone the repository
git clone https://github.com/karthyick/codebase-visualizer.git
cd codebase-visualizer

# Create and activate a virtual environment
python -m venv codebase_visualizer_venv
source codebase_visualizer_venv/bin/activate  # On Windows: codebase_visualizer_venv\Scripts\activate

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install diffusers transformers tqdm
```

## Usage

### Basic Usage

```bash
python main.py --code_dir /path/to/your/codebase
```

### Options

```bash
python main.py --help
```

- `--code_dir`: Path to the code directory to analyze
- `--output_dir`: Directory to save generated images (default: ./generated_images/)
- `--skip_individual`: Skip individual image generation
- `--skip_unified`: Skip unified ecosystem visualization
- `--batch_size`: Batch size for image generation (default: 4)
- `--force_cpu`: Force CPU execution (very slow, not recommended)
- `--fix_cuda`: Run diagnostics to troubleshoot CUDA issues

### CUDA Troubleshooting

If you're experiencing CUDA issues, run:

```bash
python main.py --fix_cuda
```

Follow the recommendations to install the correct PyTorch version for your CUDA installation.

## Visualization Styles

### 1. Biodigital Forest Ecosystem

Represents classes as trees and methods as birds in a vibrant digital forest ecosystem.

### 2. Neural Metropolis (Cyberpunk City)

Visualizes your codebase as a futuristic cyberpunk city with classes as skyscrapers and methods as drones.

### 3. Data Crystal Matrix

Displays code elements as a complex crystalline structure with glowing connections.

### 4. Futuristic Holographic Map

Presents your code as an advanced holographic projection with interactive elements and data flows.

## Customizing Visualizations

To use different visualization styles, modify the `ecosystem_prompt` variable in the `generate_unified_visualization` function in `main.py`. Sample prompts are included in the code.

## Output

The tool generates:

1. `parsed_code/codebase_data.json`: Analysis of your codebase structure
2. `parsed_code/descriptions.json`: Generated descriptions for visualization
3. `generated_images/`: Directory containing generated visualizations
4. `generated_images/codebase_ecosystem.png`: Main unified visualization

## Viewing Results

After generation, you can view the results using the included Streamlit app:

```bash
pip install streamlit pandas pillow matplotlib networkx
streamlit run app.py
```

## License

MIT

## Acknowledgements

This project uses Stable Diffusion for image generation and draws inspiration from various code visualization techniques.
