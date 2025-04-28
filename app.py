import streamlit as st
import json
import os
from PIL import Image
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="Codebase Ecosystem Visualizer", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem;
    }
    .eco-image {
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        margin-bottom: 2rem;
    }
    .stImage {
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    .stImage img {
        border-radius: 10px;
    }
    .card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .info-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .sidebar-info {
        padding: 1rem;
        background-color: #f1f8ff;
        border-radius: 5px;
        margin-bottom: 1rem;
    }
    .component-list {
        max-height: 300px;
        overflow-y: auto;
        padding: 10px;
        background-color: #f9f9f9;
        border-radius: 5px;
        border: 1px solid #eaeaea;
    }
    h1, h2, h3 {
        color: #2c3e50;
    }
    .metric-container {
        display: flex;
        justify-content: space-between;
        gap: 1rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        flex: 1;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #2c3e50;
    }
    .metric-label {
        font-size: 1rem;
        color: #7f8c8d;
    }
    .legend-item {
        display: flex;
        align-items: center;
        margin-bottom: 0.5rem;
    }
    .legend-color {
        width: 20px;
        height: 20px;
        border-radius: 4px;
        margin-right: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Function to load metadata
@st.cache_data
def load_metadata():
    """Load metadata.json from the parsed_code directory"""
    try:
        with open("./parsed_code/metadata.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        st.warning("Component metadata file not found. Component gallery will not be available.")
        return []

# Load codebase data
@st.cache_data
def load_codebase_data():
    """Load codebase_data.json from the parsed_code directory"""
    try:
        with open("./parsed_code/codebase_data.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        st.warning("Codebase data not found. Some analysis features will be limited.")
        return []

# Function to load ecosystem metadata
@st.cache_data
def load_ecosystem_metadata():
    """Load ecosystem_metadata.json if it exists (unified view)"""
    try:
        with open("./generated_images/ecosystem_metadata.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return None

# Load all data
metadata = load_metadata()
codebase_data = load_codebase_data()
ecosystem_metadata = load_ecosystem_metadata()

# Process and enrich metadata if available
if metadata:
    # Extract element types from descriptions
    for item in metadata:
        if "class" in item.get("description", "").lower():
            item["type"] = "Class"
        elif "method" in item.get("description", "").lower():
            item["type"] = "Method"
        else:
            item["type"] = "Other"
    
    # Find relationships between items
    def find_related_items(name, all_items):
        related = []
        # Class-Controller relationship
        if "Controller" not in name:
            controller_name = f"{name}Controller"
            related.extend([item for item in all_items if item["name"] == controller_name])
        else:
            model_name = name.replace("Controller", "")
            related.extend([item for item in all_items if item["name"] == model_name])
        
        # Methods belonging to class (crude approximation)
        current_item = next((item for item in all_items if item["name"] == name), None)
        if current_item and current_item["type"] == "Class":
            related.extend([item for item in all_items if item["type"] == "Method" and name in item["name"]])
        
        return related

    # Add related items to metadata
    for item in metadata:
        item["related"] = find_related_items(item["name"], metadata)

# Header
st.title("🌳 Codebase Ecosystem Visualizer")
st.write("Explore your codebase as a vibrant digital forest where classes are trees and methods are birds")

# Check if we have a unified ecosystem visualization or individual components
has_unified_view = ecosystem_metadata is not None
has_component_view = len(metadata) > 0

# Create tabs based on available data
if has_unified_view and has_component_view:
    tabs = st.tabs(["Ecosystem View", "Component Gallery", "Relationships", "Statistics"])
    ecosystem_tab = tabs[0]
    gallery_tab = tabs[1]
    relationship_tab = tabs[2]
    stats_tab = tabs[3]
elif has_unified_view:
    tabs = st.tabs(["Ecosystem View", "Statistics"])
    ecosystem_tab = tabs[0]
    stats_tab = tabs[1]
    gallery_tab = None
    relationship_tab = None
elif has_component_view:
    tabs = st.tabs(["Component Gallery", "Relationships", "Statistics"])
    ecosystem_tab = None
    gallery_tab = tabs[0]
    relationship_tab = tabs[1]
    stats_tab = tabs[2]
else:
    st.error("No visualization data available. Please run the analysis first.")
    st.stop()

# Tab: Ecosystem View (if available)
if has_unified_view and ecosystem_tab:
    with ecosystem_tab:
        st.subheader("Codebase Ecosystem")
        
        # Display the unified ecosystem image
        try:
            image_path = ecosystem_metadata["image_path"]
            if os.path.exists(image_path):
                st.image(Image.open(image_path), use_container_width=True, caption="Your codebase visualized as a cohesive ecosystem")
            else:
                st.error(f"Ecosystem image not found at: {image_path}")
        except Exception as e:
            st.error(f"Error displaying ecosystem image: {str(e)}")
        
        # Display ecosystem legend
        st.subheader("Ecosystem Legend")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Trees represent classes:**")
            st.markdown("- 🌳 Large oak trees: Large classes (300+ lines)")
            st.markdown("- 🌲 Medium trees: Medium classes (100-300 lines)")
            st.markdown("- 🌱 Small saplings: Small classes (<100 lines)")
        
        with col2:
            st.markdown("**Birds represent methods:**")
            st.markdown("- 🐦 Blue birds: Getter methods (Get*, Fetch*)")
            st.markdown("- 🦜 Yellow-green parrots: Setter methods (Set*, Update*)")
            st.markdown("- 🦢 Red cardinals: Creation methods (Create*, Add*)")
            st.markdown("- 🦅 Woodpeckers: Deletion methods (Delete*, Remove*)")
        
        # Display ecosystem statistics
        st.subheader("Ecosystem Statistics")
        
        class_count = ecosystem_metadata.get("class_count", 0)
        method_count = ecosystem_metadata.get("method_count", 0)
        namespaces = ecosystem_metadata.get("namespaces", [])
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Classes (Trees)", class_count)
        
        with col2:
            st.metric("Methods (Birds)", method_count)
        
        with col3:
            st.metric("Namespaces (Regions)", len(namespaces))
        
        # Show the prompt used to generate the image
        if "prompt" in ecosystem_metadata:
            with st.expander("View Generation Prompt"):
                st.code(ecosystem_metadata["prompt"], language="text")

# Tab: Component Gallery
if has_component_view and gallery_tab:
    with gallery_tab:
        # Sidebar filters
        st.sidebar.header("Gallery Filters")
        
        # Filter by type
        element_types = ["All"] + list(set(item["type"] for item in metadata))
        selected_type = st.sidebar.selectbox("Element Type", element_types)
        
        # Filter by name
        name_filter = st.sidebar.text_input("Filter by name", "")
        
        # Apply filters
        filtered_metadata = metadata
        if selected_type != "All":
            filtered_metadata = [item for item in filtered_metadata if item["type"] == selected_type]
        
        if name_filter:
            filtered_metadata = [item for item in filtered_metadata if name_filter.lower() in item["name"].lower()]
        
        # Sort options
        sort_options = ["Name (A-Z)", "Name (Z-A)", "Type"]
        sort_by = st.sidebar.selectbox("Sort by", sort_options)
        
        if sort_by == "Name (A-Z)":
            filtered_metadata.sort(key=lambda x: x["name"])
        elif sort_by == "Name (Z-A)":
            filtered_metadata.sort(key=lambda x: x["name"], reverse=True)
        elif sort_by == "Type":
            filtered_metadata.sort(key=lambda x: x["type"])
        
        # Display result count
        st.sidebar.markdown(f"**Showing {len(filtered_metadata)} of {len(metadata)} elements**")
        
        # Display gallery
        st.subheader("Component Gallery")
        
        if not filtered_metadata:
            st.info("No elements match your filter criteria.")
        else:
            # Create columns for layout
            cols = st.columns(3)  # Show 3 images per row
            
            for idx, item in enumerate(filtered_metadata):
                col = cols[idx % 3]
                with col:
                    try:
                        # Display image
                        st.image(
                            Image.open(item['image_path']), 
                            caption=item['name'],
                            use_container_width=True
                        )
                        
                        # Display details
                        with st.expander("Details"):
                            st.write(f"**Name:** {item['name']}")
                            st.write(f"**Type:** {item['type']}")
                            st.write(f"**Description:** {item['description']}")
                            
                            # Show related items if any
                            if item.get("related", []):
                                st.write("**Related to:**")
                                for related in item["related"]:
                                    st.write(f"- {related['name']} ({related['type']})")
                    except Exception as e:
                        st.error(f"Error loading image for {item['name']}: {str(e)}")

# Tab: Relationships
if has_component_view and relationship_tab:
    with relationship_tab:
        st.subheader("Codebase Relationships")
        
        # Build a relationship graph
        G = nx.DiGraph()
        
        # Add nodes for all items
        for item in metadata:
            G.add_node(item["name"], type=item["type"])
        
        # Add edges for relationships
        for item in metadata:
            for related in item.get("related", []):
                G.add_edge(item["name"], related["name"])
        
        # Visualize the graph
        if nx.number_of_nodes(G) > 0:
            # Create visualization
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Use a hierarchical layout for better visualization of relationships
            pos = nx.spring_layout(G, seed=42)
            
            # Draw nodes with different colors based on type
            class_nodes = [n for n in G.nodes() if nx.get_node_attributes(G, "type").get(n) == "Class"]
            method_nodes = [n for n in G.nodes() if nx.get_node_attributes(G, "type").get(n) == "Method"]
            
            nx.draw_networkx_nodes(G, pos, 
                                   nodelist=class_nodes, 
                                   node_color="#3498db", 
                                   node_size=800, 
                                   alpha=0.9)
            
            nx.draw_networkx_nodes(G, pos, 
                                   nodelist=method_nodes, 
                                   node_color="#2ecc71", 
                                   node_size=500, 
                                   alpha=0.9)
            
            # Draw edges
            nx.draw_networkx_edges(G, pos, width=1.5, alpha=0.7, edge_color="#7f8c8d")
            
            # Draw labels
            nx.draw_networkx_labels(G, pos, font_size=10, font_family="sans-serif")
            
            # Remove axis
            plt.axis("off")
            
            # Display
            st.pyplot(fig)
            
            # Show relationship details
            st.subheader("Relationship Details")
            
            # Show classes with their related items
            class_items = [item for item in metadata if item["type"] == "Class"]
            for item in class_items:
                if item.get("related", []):
                    with st.expander(f"{item['name']} relationships"):
                        col1, col2 = st.columns([1, 2])
                        with col1:
                            try:
                                st.image(Image.open(item['image_path']), use_container_width=True)
                            except:
                                st.warning("Image not available")
                        
                        with col2:
                            st.write(f"**{item['name']}** is related to:")
                            for related in item["related"]:
                                st.markdown(f"- **{related['name']}** ({related['type']})")
        else:
            st.info("No relationships detected in the codebase.")

# Tab: Statistics
if stats_tab:
    with stats_tab:
        st.subheader("Codebase Statistics")
        
        # Calculate stats
        if has_component_view:
            class_count = sum(1 for item in metadata if item["type"] == "Class")
            method_count = sum(1 for item in metadata if item["type"] == "Method")
        elif has_unified_view:
            class_count = ecosystem_metadata.get("class_count", 0)
            method_count = ecosystem_metadata.get("method_count", 0)
        else:
            class_count = 0
            method_count = 0
        
        # Basic stats
        col1, col2, col3 = st.columns(3)
        
        if has_unified_view:
            file_count = ecosystem_metadata.get("parsed_code_summary", {}).get("file_count", 0)
            col1.metric("Total Elements", class_count + method_count)
            col2.metric("Classes", class_count)
            col3.metric("Methods", method_count)
        else:
            col1.metric("Total Elements", len(metadata))
            col2.metric("Classes", class_count)
            col3.metric("Methods", method_count)
        
        # Create pie chart
        if class_count > 0 or method_count > 0:
            fig, ax = plt.subplots()
            ax.pie([class_count, method_count], 
                labels=["Classes", "Methods"],
                autopct='%1.1f%%',
                colors=["#3498db", "#2ecc71"],
                startangle=90,
                explode=(0.1, 0))
            ax.axis('equal')
            
            st.pyplot(fig)
        
        # More detailed stats if codebase data is available
        if codebase_data:
            # Calculate LOC stats
            if isinstance(codebase_data, list):
                loc_values = [entry.get("loc", 0) for entry in codebase_data]
                
                if loc_values:
                    st.subheader("Lines of Code Distribution")
                    
                    fig, ax = plt.subplots()
                    ax.hist(loc_values, bins=20, color="#3498db", alpha=0.7)
                    ax.set_xlabel("Lines of Code")
                    ax.set_ylabel("Frequency")
                    ax.grid(True, alpha=0.3)
                    
                    st.pyplot(fig)
                    
                    # File statistics
                    st.subheader("File Statistics")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Total Files", len(codebase_data))
                    if len(codebase_data) > 0:
                        col2.metric("Average LOC per File", round(sum(loc_values) / len(loc_values), 1))
                        col3.metric("Largest File (LOC)", max(loc_values))

# Footer
st.markdown("---")
st.markdown("Codebase Ecosystem Visualizer • Connecting Code and Nature")