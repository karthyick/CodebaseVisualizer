import os
import re
import json
import argparse
from tqdm import tqdm
import torch
from diffusers import StableDiffusionPipeline

def parse_codebase(code_dir):
    """
    Parse a .NET codebase directory to extract information about classes and methods.
    
    Args:
        code_dir (str): Path to the directory containing the codebase
        
    Returns:
        list: List of dictionaries containing parsed information about each file
    """
    print("🔍 Analyzing codebase...")
    code_data = []
    
    # Filter for only C# files
    cs_files = []
    for root, dirs, files in os.walk(code_dir):
        for file in files:
            if file.endswith(".cs"):
                cs_files.append(os.path.join(root, file))
    
    # Process each file with a progress bar
    for path in tqdm(cs_files, desc="Parsing files"):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract classes, methods and namespaces
            namespaces = re.findall(r'namespace (\S+)', content)
            classes = re.findall(r'class (\w+)', content)
            # Find public, private, protected, internal methods
            methods = re.findall(r'(public|private|protected|internal) (\w+) (\w+)\(', content)
            # Count lines of code (excluding empty lines)
            non_empty_lines = [line for line in content.split('\n') if line.strip()]
            loc = len(non_empty_lines)
            
            code_data.append({
                "file_path": os.path.relpath(path, code_dir),
                "namespaces": namespaces,
                "classes": classes,
                "methods": [m[2] for m in methods],
                "loc": loc
            })
        except Exception as e:
            print(f"Error parsing file {path}: {e}")
    
    print(f"✅ Parsed {len(code_data)} files.")
    return code_data

def generate_descriptions(parsed_code):
    """
    Generate natural language descriptions for code elements.
    
    Args:
        parsed_code (list): Parsed code data from parse_codebase
        
    Returns:
        list: Tuples of (element_name, description)
    """
    print("📝 Generating descriptions...")
    descriptions = []
    
    # First pass - collect all classes and relationships
    class_info = {}
    for entry in parsed_code:
        file_name = os.path.basename(entry['file_path'])
        
        # Store class info with namespace and file info
        for cls in entry['classes']:
            class_info[cls] = {
                'file': file_name,
                'namespace': entry.get('namespaces', []),
                'methods': [],
                'loc': entry['loc'],
                'related_classes': []
            }
    
    # Second pass - detect relationships
    for cls1 in class_info:
        for cls2 in class_info:
            if cls1 != cls2:
                # Detect inheritance/implementation (crude but effective for visualization)
                if cls1 in cls2 or cls2 in cls1:
                    class_info[cls1]['related_classes'].append(cls2)
                # Controllers and models often have relationships
                if cls1.replace("Controller", "") == cls2 or cls2.replace("Controller", "") == cls1:
                    class_info[cls1]['related_classes'].append(cls2)
    
    # Generate better descriptions
    # Generate updated class descriptions
    for cls in entry['classes']:
        related = class_info.get(cls, {}).get('related_classes', [])

        if len(related) > 0:
            # For classes with relationships
            if "Controller" in cls:
                desc = f"A central Class Monolith titled '{cls}' inside its Namespace Sphere, emitting directed neon energy beams towards related component Monoliths, representing controller coordination."
            elif entry['loc'] > 300:
                desc = f"A towering Class Monolith '{cls}' glowing with intense neon blue light, structurally complex with multiple outgoing Dependency Laser Beams, indicating a highly interconnected core class."
            elif entry['loc'] > 100:
                desc = f"A mid-sized Class Monolith '{cls}' with balanced neon energy, connected through organized Dependency Beams to several neighboring Classes within its Namespace Sphere."
            else:
                desc = f"A compact but important Class Monolith '{cls}' orbiting close to key structures, maintaining soft neon links to related Monoliths."
        else:
            # For standalone classes (no relationships)
            if entry['loc'] > 300:
                desc = f"A massive isolated Class Monolith '{cls}' with deep internal complexity, standing alone inside its Namespace Sphere, radiating a strong stable neon field."
            elif entry['loc'] > 100:
                desc = f"A medium-sized independent Class Monolith '{cls}', softly glowing with faint neon outlines, unconnected to major dependency structures."
            else:
                desc = f"A small solitary Class Monolith '{cls}', minimal in structure, quietly positioned within its Namespace Sphere with no external connections."

        descriptions.append((cls, desc))

    # Generate updated method descriptions
    for method in entry['methods']:
        cls_match = None
        for cls in entry['classes']:
            if cls in class_info:
                class_info[cls]['methods'].append(method)
                cls_match = cls

        if method.startswith("Get") or method.startswith("Fetch"):
            desc = f"A soft neon blue Method Node labeled '{method}', orbiting near its parent Class Monolith, retrieving data from internal class structures."
        elif method.startswith("Set") or method.startswith("Update"):
            desc = f"A pale green Method Node '{method}' modifying and transmitting internal structural data within its parent Class Monolith lattice."
        elif method.startswith("Create") or method.startswith("Add"):
            desc = f"A gentle orange crystalline Method Node '{method}', generating new logical constructs within the Class Monolith ecosystem."
        elif method.startswith("Delete") or method.startswith("Remove"):
            desc = f"A crisp white semi-transparent Method Node '{method}', systematically dismantling obsolete structures inside its parent Class Monolith."
        elif method.startswith("Calculate") or method.startswith("Compute"):
            desc = f"A focused neon white Method Node '{method}', performing high-precision mathematical and algorithmic operations inside the Class structure."
        else:
            desc = f"A minor Method Node '{method}' orbiting silently, handling specialized tasks inside its assigned Class Monolith."

        descriptions.append((method, desc))

    print(f"✅ Generated {len(descriptions)} futuristic-style descriptions matching holographic visualization.")

    return descriptions

def check_cuda_available():
    """
    Check if CUDA is available and raise an error if not.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("❌ CUDA is not available. This script requires CUDA to run.")
    
    print(f"✅ CUDA is available. Using device: cuda")
    print(f"   CUDA Device: {torch.cuda.get_device_name(0)}")
    print(f"   CUDA Version: {torch.version.cuda}")
    return True

def generate_images(descriptions, output_folder="./generated_images/", batch_size=4):
    """
    Generate individual images using Stable Diffusion based on descriptions.
    
    Args:
        descriptions (list): List of (name, description) tuples
        output_folder (str): Folder to save generated images
        batch_size (int): Number of images to generate in parallel
    """
    # Verify CUDA is available
    check_cuda_available()
    
    print("🖼️ Setting up Stable Diffusion...")
    os.makedirs("parsed_code", exist_ok=True)
    os.makedirs(output_folder, exist_ok=True)
    
    # Using CUDA with float16 precision
    device = "cuda"
    torch_dtype = torch.float16
    
    # Load model
    pipe = StableDiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-1",
        torch_dtype=torch_dtype,
        revision="fp16",
    ).to(device)
    
    metadata = []
    
    # Generate images with progress bar
    print("🎨 Generating images...")
    for i, (name, desc) in enumerate(tqdm(descriptions, desc="Generating images")):
        # Clean name for file system
        safe_name = re.sub(r'[^\w\-_.]', '_', name)
        image_path = f"{output_folder}/{safe_name}.png"
        
        # Skip if image already exists
        if os.path.exists(image_path):
            print(f"Image for {name} already exists, skipping...")
            metadata.append({
                "name": name,
                "description": desc,
                "image_path": image_path
            })
            continue
        
        # Generate image with some randomness
        image = pipe(
            desc, 
            num_inference_steps=30,
            guidance_scale=7.5
        ).images[0]
        
        # Save image
        image.save(image_path)
        
        metadata.append({
            "name": name,
            "description": desc,
            "image_path": image_path
        })
        
        # Free up CUDA memory
        if (i + 1) % batch_size == 0:
            torch.cuda.empty_cache()
    
    # Save metadata
    with open("./parsed_code/metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)
        
    print(f"✅ Generated {len(metadata)} images in {output_folder}")
    return metadata

def generate_unified_visualization(parsed_code, descriptions, output_folder="./generated_images/"):
    """
    Generate a unified visualization of the entire codebase ecosystem
    rather than individual images for each component.
    
    Args:
        parsed_code (list): Parsed code data
        descriptions (list): List of (name, description) tuples
        output_folder (str): Folder to save generated image
    """
    # Verify CUDA is available
    check_cuda_available()
    
    print("🖼️ Setting up unified visualization generation...")
    os.makedirs(output_folder, exist_ok=True)
    
    # Prepare the ecosystem description
    class_count = sum(1 for entry in parsed_code for _ in entry.get('classes', []))
    method_count = sum(1 for entry in parsed_code for _ in entry.get('methods', []))
    
    # Get namespaces to represent as different regions
    all_namespaces = set()
    for entry in parsed_code:
        for namespace in entry.get('namespaces', []):
            all_namespaces.add(namespace)
    
    # Create ecosystem description
    ecosystem_prompt = f"""
Create a hyper-realistic, cinematic 8K visualization of a futuristic AI codebase, structured as a fully physical 3D holographic system, emphasizing real named object representations for maximum clarity, realism, and trust.

Global Environment:
- Deep matte black infinite background representing the void.
- Ultra-thin holographic 3D coordinate grid (X, Y, Z) extending infinitely into deep parallax.
- Sparse ambient particle fog at low opacity for atmospheric realism.
- Subtle cold neon blue global ambient lighting tint.

Primary Components:

1. Namespace Spheres ({len(all_namespaces)} total):
   - Large floating transparent spherical zones ("Namespace Spheres") or toroidal rings.
   - Each Namespace Sphere softly glowing with smooth neon volumetric color:
     - Cool Cyan, Pale Teal, Icy Blue, or Soft Violet per Namespace.
   - Namespace Name Label: Thin, high-tech font floating slightly above each sphere center.
   - Internal faint rotating energy filaments connecting internal classes visually.
   - Smooth rotation animation around the Namespace's Y-axis to indicate active domains.

2. Class Monoliths ({class_count} total):
   - Each Class represented by a "Class Panel Monolith" — vertical, transparent glass-like floating panel.
   - Class Monolith Layout:
     - Top Section: Large engraved Class Name illuminated with bright neon blue.
     - Middle Section: Vertical stacked Line of Code (LOC) bars, height representing class size.
     - Side Graphs: Circular dynamic graphs showing Cyclomatic Complexity and Method Density.
     - Internal Layout: Faint neon latticework showing class internal flow.
   - Dimensions:
     - Height = Lines of Code
     - Width = Class complexity
   - Cold neon blue frame with pulsing inner glow to show active/inactive status.

3. Method Nodes ({method_count} total):
   - "Method Nodes" orbiting around their respective Class Monoliths.
   - Each Method Node represented as:
     - Getter Methods → Soft neon blue orbs.
     - Setter Methods → Pale emerald green spheres.
     - Creator Methods → Gentle orange crystalline hex-spheres.
     - Deleter Methods → Crisp white semi-transparent spheres.
   - Floating Method Name Labels beside each Node in thin holographic text.
   - Multi-layer orbital paths sorted by method type and access (public/protected/private).

4. Dependency Laser Beams:
   - Razor-thin "Dependency Beams" connecting Class Monoliths to represent cross-class dependencies.
   - Beam Properties:
     - Brightness proportional to dependency frequency.
     - Beam thickness represents dependency strength.
     - Moving particle streams along the beams representing real-time interactions.
   - Laser Beam Color:
     - Soft neon blue for normal dependencies.
     - Pale violet for cross-namespace dependencies.

5. Data Pulses:
   - Semi-transparent luminous "Data Pulses" moving along dependency beams and internal class lattice.
   - Pulses visually depict live system communication and interactions.

6. Micro HUD Panels:
   - "Floating Info Panels" next to important Class Monoliths:
     - Displaying mini-dashboards:
         - Line of Code (LOC) count
         - Method Count
         - Cyclomatic Complexity
         - Last Modified Timestamp
     - Designed with sharp, minimalistic neon grid layouts.
     - Real-time slight flicker to simulate data updating.

Lighting and Atmosphere:
- Volumetric holographic light scattering through Namespace Spheres and Class Monoliths.
- Subtle holographic bloom around important Method Nodes and HUD Panels.
- Ray-traced reflections on glassy monolith surfaces.
- Lens flares dynamically triggered when camera aligns with beams or central clusters.

Camera and Cinematic Motion:
- Very slow orbital drift around the entire system to reveal 3D structure.
- Smooth cinematic dolly zoom transitions:
    - Namespace Sphere → Inside to Class Monolith → Zoom further to Method Nodes.
- Depth-of-field automatically adjusts to highlight focus area.
- Golden Ratio spatial composition ensuring natural visual balance.

Color Palette (Strict Control):
- Primary Color: Neon Blue
- Secondary Accent: White
- Tertiary Accents: Pale Green, Soft Cyan, Light Violet
- No random saturated colors — strict futuristic scientific control.

Composition Rules:
- Spatial hierarchy strictly enforced:
    - Outer Layer: Namespace Spheres
    - Mid Layer: Class Monoliths clustered by Namespace
    - Inner Layer: Method Nodes orbiting Classes
    - Interconnecting Layer: Dependency Laser Beams
- Sufficient breathing space maintained between objects to ensure maximum clarity.

Goal:
Generate a physically plausible, breathtakingly cinematic, scientifically accurate holographic visualization of a living software ecosystem — where every component (Namespace Sphere, Class Monolith, Method Node, Dependency Beam, HUD Panel, Data Pulse) is clearly identifiable, elegantly organized, and visually stunning, at the level of visualization fidelity demanded by NASA, Tesla, or OpenAI R&D labs.
"""

    # Enhance with specific important classes if there aren't too many
    if class_count <= 10:
        class_section = "Important classes include:\n"
        for name, desc in descriptions:
            if "class" in desc.lower():
                class_section += f"- {name}: {desc}\n"
        ecosystem_prompt += class_section
    
    print("📝 Generated ecosystem prompt:")
    print(ecosystem_prompt)
    
    # Set up the image generation pipeline - CUDA only
    device = "cuda"
    torch_dtype = torch.float16
    
    # Load model
    pipe = StableDiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-1",
        torch_dtype=torch_dtype,
        revision="fp16",
    ).to(device)
    
    print("🎨 Generating unified ecosystem visualization...")
    
    # Generate a higher resolution image for the ecosystem
    image = pipe(
        ecosystem_prompt,
        height=768,  # Higher resolution
        width=1024,  # Wider format
        num_inference_steps=50,  # More steps for quality
        guidance_scale=7.5
    ).images[0]
    
    # Save the ecosystem image
    ecosystem_path = f"{output_folder}/codebase_ecosystem.png"
    image.save(ecosystem_path)
    
    # Save the prompt and metadata
    ecosystem_metadata = {
        "image_path": ecosystem_path,
        "prompt": ecosystem_prompt,
        "class_count": class_count,
        "method_count": method_count,
        "namespaces": list(all_namespaces),
        "parsed_code_summary": {
            "file_count": len(parsed_code),
            "total_loc": sum(entry.get('loc', 0) for entry in parsed_code)
        }
    }
    
    with open(f"{output_folder}/ecosystem_metadata.json", "w") as f:
        json.dump(ecosystem_metadata, f, indent=4)
    
    print(f"✅ Generated unified ecosystem visualization at {ecosystem_path}")
    return ecosystem_metadata

def enhance_with_local_llm(parsed_code, descriptions):
    """
    Use a local LLM to enhance the ecosystem description for better visualization.
    Only runs if both CUDA and transformers are available.
    
    Args:
        parsed_code (list): Parsed code data
        descriptions (list): Initial descriptions
        
    Returns:
        str: Enhanced ecosystem prompt
    """
    # Verify CUDA is available
    check_cuda_available()
    
    try:
        print("🧠 Setting up local LLM for enhanced descriptions...")
        
        # Create a summary of the codebase structure
        summary = {
            "file_count": len(parsed_code),
            "total_loc": sum(entry.get('loc', 0) for entry in parsed_code),
            "class_count": sum(1 for entry in parsed_code for _ in entry.get('classes', [])),
            "method_count": sum(1 for entry in parsed_code for _ in entry.get('methods', [])),
            "namespaces": list(set(ns for entry in parsed_code for ns in entry.get('namespaces', []))),
            "classes": [cls for entry in parsed_code for cls in entry.get('classes', [])],
            "methods": [method for entry in parsed_code for method in entry.get('methods', [])]
        }
        
        # Prepare input for the LLM
        prompt = f"""
        You are a creative visualization expert. Create an artistic description for a single cohesive image that 
        visualizes this codebase as a forest ecosystem:
        
        Codebase summary:
        - {summary['file_count']} files with {summary['total_loc']} total lines of code
        - {summary['class_count']} classes and {summary['method_count']} methods
        - Namespaces: {', '.join(summary['namespaces'])}
        
        Guidelines:
        1. Visualize classes as trees of different sizes and types
        2. Visualize methods as birds performing different activities
        3. Represent namespaces as distinct regions or biomes
        4. Show relationships between classes and their methods
        5. Create a vibrant, detailed landscape that shows the full structure
        6. Include specific details about important classes and their relationships
        
        Your description will be used to generate an actual image, so be specific about visual elements.
        """
        
        # Import transformers for local LLM
        from transformers import pipeline
        
        # Use a local LLM for text generation - CUDA only
        generator = pipeline(
            'text-generation', 
            model='TheBloke/Llama-2-7B-chat-GGUF', 
            device=0  # Force CUDA device
        )
        
        print("Generating enhanced description using local LLM...")
        response = generator(
            prompt,
            max_length=1000,
            num_return_sequences=1,
            temperature=0.7,
        )[0]['generated_text']
        
        # Extract just the response part (not the prompt)
        enhanced_description = response.split(prompt)[-1].strip()
        
        print("✓ Generated enhanced ecosystem description using local LLM")
        return enhanced_description
            
    except ImportError as e:
        print(f"⚠️ Error with transformers package: {str(e)}. Using default description generation.")
        return None
    except Exception as e:
        print(f"⚠️ Error using LLM for enhancement: {str(e)}. Using default description generation.")
        return None

def main():
    """
    Main function to orchestrate the codebase visualization process.
    """
    # First check if CUDA is available
    try:
        check_cuda_available()
    except RuntimeError as e:
        print(str(e))
        print("This script requires CUDA GPU support. Exiting.")
        return
        
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description="Codebase Ecosystem Visualizer (CUDA Only)")
    parser.add_argument("--code_dir", help="Path to the code directory to analyze")
    parser.add_argument("--output_dir", default="./generated_images/", help="Directory to save generated images")
    parser.add_argument("--skip_individual", action="store_true", help="Skip individual image generation")
    parser.add_argument("--skip_unified", action="store_true", help="Skip unified ecosystem visualization")
    parser.add_argument("--use_llm", action="store_true", help="Use local LLM for enhanced descriptions")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for image generation to manage GPU memory")
    args = parser.parse_args()
    
    # Get the code directory
    code_dir = args.code_dir
    
    # If no directory provided via command line, prompt the user
    if not code_dir:
        code_dir = input("📂 Enter the path to your code directory: ").strip()
    
    # Validate directory
    if not os.path.isdir(code_dir):
        print(f"❌ Error: '{code_dir}' is not a valid directory.")
        return
    
    # Create necessary directories
    os.makedirs("parsed_code", exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Step 1: Parse the codebase
    print(f"Analyzing codebase in: {code_dir}")
    parsed_code = parse_codebase(code_dir)
    
    # Save the parsed data
    with open("./parsed_code/codebase_data.json", "w") as f:
        json.dump(parsed_code, f, indent=4)
        print("✓ Saved parsed codebase data to ./parsed_code/codebase_data.json")
    
    # Step 2: Generate descriptions
    descriptions = generate_descriptions(parsed_code)
    
    # Save the descriptions
    with open("./parsed_code/descriptions.json", "w") as f:
        json.dump(descriptions, f, indent=4)
        print("✓ Saved descriptions to ./parsed_code/descriptions.json")
    
    # Step 3: Generate images based on the chosen approach
    if not args.skip_individual:
        # Generate individual images for each component
        print("Generating individual component images...")
        generate_images(descriptions, output_folder=args.output_dir, batch_size=args.batch_size)
    
    if not args.skip_unified:
        # Generate unified ecosystem visualization
        print("Generating unified ecosystem visualization...")
        
        # Use LLM for enhanced description if requested
        enhanced_prompt = None
        if args.use_llm:
            enhanced_prompt = enhance_with_local_llm(parsed_code, descriptions)
        
        # Generate the ecosystem visualization
        generate_unified_visualization(parsed_code, descriptions, output_folder=args.output_dir)
    
    print("\n✅ Codebase visualization complete!")
    print("\nTo view your visualization:")
    print("1. Make sure Streamlit is installed: pip install streamlit pandas pillow matplotlib networkx")
    print("2. Ensure you have app.py in the current directory")
    print("3. Run: streamlit run app.py")

if __name__ == "__main__":
    main()