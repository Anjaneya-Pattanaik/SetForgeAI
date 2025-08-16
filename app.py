import streamlit as st
import pandas as pd
import numpy as np
import io
from datetime import date
from utils.generator import (
    generate_data,
    validate_fields,
    validate_relationships,
    validate_categorical_relationships,
    analyze_data_quality
)

# Page configuration
st.set_page_config(
    page_title="SetForgeAI",
    layout="wide",
    page_icon="🔧"
)

# Initialize session state
if 'generated_df' not in st.session_state:
    st.session_state.generated_df = None
if 'generation_params' not in st.session_state:
    st.session_state.generation_params = None
# if 'dark_theme' not in st.session_state:
#     st.session_state.dark_theme = False

# # Dark theme toggle functionality
# def toggle_theme():
#     st.session_state.dark_theme = not st.session_state.dark_theme
#     st.rerun()

# # Apply dark theme styles with comprehensive coverage
# if st.session_state.dark_theme:
#     st.markdown("""
#     <style>
#     /* Main app background */
#     .stApp {
#         background-color: #2b2b2b;
#         color: #e8e8e8;
#     }
    
#     /* Sidebar styling */
#     .css-1d391kg {
#         background-color: #363636;
#     }
    
#     /* Main content area */
#     .main .block-container {
#         background-color: #2b2b2b;
#         color: #e8e8e8;
#     }
    
#     /* Headers */
#     h1, h2, h3, h4, h5, h6 {
#         color: #ffffff !important;
#     }
    
#     /* Text and labels - comprehensive coverage */
#     .stMarkdown, .stText, label, p, span, div {
#         color: #e8e8e8 !important;
#     }
    
#     /* Specific targeting for checkbox and radio labels */
#     .stCheckbox label, .stRadio label, .stCheckbox > label > div, .stRadio > label > div {
#         color: #e8e8e8 !important;
#     }
    
#     /* Checkbox and radio button text */
#     .stCheckbox > label > div > span, .stRadio > label > div > span {
#         color: #e8e8e8 !important;
#     }
    
#     /* Help text and captions */
#     .stHelp, .caption, small {
#         color: #000000 !important;
#     }
    
#     /* HELP BUTTON AND TOOLTIP FIXES */
#     /* Help button styling */
#     button[title*="help"], button[aria-label*="help"], 
#     div[data-testid*="help"], span[title*="help"],
#     .stTooltipIcon, .stHelpTooltip {
#         color: #ffffff !important;
#         background-color: #555555 !important;
#         border: 1px solid #777777 !important;
#     }
    
#     /* Help button hover */
#     button[title*="help"]:hover, button[aria-label*="help"]:hover,
#     div[data-testid*="help"]:hover, span[title*="help"]:hover {
#         background-color: #666666 !important;
#         color: #ffffff !important;
#     }
    
#     /* Tooltip content */
#     .stTooltip, [data-baseweb="tooltip"], .tooltip-content,
#     div[role="tooltip"], [aria-describedby] {
#         background-color: #ffffff !important;  /* White background */
#         color: #000000 !important;             /* Black text - now readable */
#         border: 1px solid #cccccc !important;
#         box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3) !important;
#     }

#     /* Additional tooltip targeting */
#     div[data-baseweb="tooltip"] div,
#     div[data-baseweb="tooltip"] p,
#     div[data-baseweb="tooltip"] span,
#     .stTooltip div, .stTooltip p, .stTooltip span {
#         color: #000000 !important;
#         background-color: transparent !important;
#     }
    
#     /* Tooltip arrow */
#     .stTooltip::before, [data-baseweb="tooltip"]::before {
#         border-color: #404040 !important;
#     }
    
#     /* Streamlit help icons - target by SVG */
#     svg[data-testid="helpIcon"], svg[data-icon="help"],
#     .stHelp svg, .stTooltipIcon svg {
#         color: #000000 !important;
#         fill: #ffffff !important;
#         stroke: #000000 !important;
#     }
    
#     /* Question mark symbols */
#     .stHelp::before, .stTooltipIcon::before {
#         color: #000000 !important;
#     }
    
#     /* Input field labels */
#     .stTextInput label, .stNumberInput label, .stSelectbox label, .stSlider label, .stDateInput label {
#         color: #e8e8e8 !important;
#     }
    
#     /* Input fields */
#     .stTextInput > div > div > input,
#     .stNumberInput > div > div > input,
#     .stTextArea > div > div > textarea {
#         background-color: #404040 !important;
#         color: #ffffff !important;
#         border: 1px solid #555555 !important;
#     }
    
#     /* Select boxes */
#     .stSelectbox > div > div > div {
#         background-color: #404040 !important;
#         color: #ffffff !important;
#         border: 1px solid #555555 !important;
#     }
    
#     /* Select box options */
#     .stSelectbox > div > div > div > div {
#         background-color: #404040 !important;
#         color: #ffffff !important;
#     }
    
#     /* Slider components */
#     .stSlider > div > div > div {
#         background-color: #404040 !important;
#     }
    
#     .stSlider > div > div > div > div {
#         background-color: #404040 !important;
#     }
    
#     .stSlider .stSlider-value {
#         color: #e8e8e8 !important;
#     }
    
#     /* Buttons */
#     .stButton > button {
#         background-color: #4a4a4a !important;
#         color: #ffffff !important;
#         border: 1px solid #666666 !important;
#     }
    
#     .stButton > button:hover {
#         background-color: #5a5a5a !important;
#         border: 1px solid #777777 !important;
#     }
    
#     /* Primary button */
#     .stButton > button[kind="primary"] {
#         background-color: #0066cc !important;
#         color: #ffffff !important;
#         border: 1px solid #0066cc !important;
#     }
    
#     .stButton > button[kind="primary"]:hover {
#         background-color: #0056b3 !important;
#         border: 1px solid #0056b3 !important;
#     }
    
#     /* Expanders */
#     .streamlit-expanderHeader {
#         background-color: #404040 !important;
#         color: #ffffff !important;
#         border: 1px solid #555555 !important;
#     }
    
#     .streamlit-expanderContent {
#         background-color: #363636 !important;
#         border: 1px solid #555555 !important;
#     }
    
#     /* Expander header text */
#     .streamlit-expanderHeader p {
#         color: #ffffff !important;
#     }
    
#     /* Expander content text */
#     .streamlit-expanderContent p, .streamlit-expanderContent div, .streamlit-expanderContent span {
#         color: #e8e8e8 !important;
#     }
    
#     /* Radio button styling */
#     .stRadio > div {
#         background-color: transparent !important;
#     }
    
#     .stRadio > div > label {
#         color: #e8e8e8 !important;
#     }
    
#     .stRadio > div > label > div {
#         color: #e8e8e8 !important;
#     }
    
#     /* Checkbox styling */
#     .stCheckbox > div {
#         background-color: transparent !important;
#     }
    
#     .stCheckbox > div > label {
#         color: #e8e8e8 !important;
#     }
    
#     .stCheckbox > div > label > div {
#         color: #e8e8e8 !important;
#     }
    
#     /* Dataframes */
#     .dataframe {
#         background-color: #404040 !important;
#         color: #ffffff !important;
#     }
    
#     /* Column headers in dataframes */
#     .dataframe th {
#         background-color: #555555 !important;
#         color: #ffffff !important;
#     }
    
#     /* Dataframe cells */
#     .dataframe td {
#         background-color: #404040 !important;
#         color: #ffffff !important;
#     }
    
#     /* Metrics */
#     .metric-container {
#         background-color: #404040 !important;
#         border: 1px solid #555555 !important;
#         border-radius: 5px;
#         padding: 10px;
#     }
    
#     /* Metric labels and values */
#     .metric-container label, .metric-container div {
#         color: #e8e8e8 !important;
#     }
    
#     /* Tabs */
#     .stTabs [data-baseweb="tab-list"] {
#         background-color: #404040 !important;
#     }
    
#     .stTabs [data-baseweb="tab"] {
#         background-color: #404040 !important;
#         color: #e8e8e8 !important;
#         border: 1px solid #555555 !important;
#     }
    
#     .stTabs [aria-selected="true"] {
#         background-color: #0066cc !important;
#         color: #ffffff !important;
#     }
    
#     /* Tab content */
#     .stTabs > div > div > div > div {
#         color: #e8e8e8 !important;
#     }
    
#     /* Info/Warning/Error boxes */
#     .stAlert {
#         background-color: #404040 !important;
#         color: #ffffff !important;
#         border: 1px solid #555555 !important;
#     }
    
#     /* Success messages */
#     .stSuccess {
#         background-color: #2d5a2d !important;
#         color: #ffffff !important;
#         border: 1px solid #4a7c4a !important;
#     }
    
#     /* Error messages */
#     .stError {
#         background-color: #5a2d2d !important;
#         color: #ffffff !important;
#         border: 1px solid #7c4a4a !important;
#     }
    
#     /* Warning messages */
#     .stWarning {
#         background-color: #5a5a2d !important;
#         color: #ffffff !important;
#         border: 1px solid #7c7c4a !important;
#     }
    
#     /* Info messages */
#     .stInfo {
#         background-color: #2d4a5a !important;
#         color: #ffffff !important;
#         border: 1px solid #4a6b7c !important;
#     }
    
#     /* Download buttons */
#     .stDownloadButton > button {
#         background-color: #4a4a4a !important;
#         color: #ffffff !important;
#         border: 1px solid #666666 !important;
#     }
    
#     .stDownloadButton > button:hover {
#         background-color: #5a5a5a !important;
#     }
    
#     /* Dividers */
#     hr {
#         border-color: #555555 !important;
#     }
    
#     /* Code blocks */
#     .stCode {
#         background-color: #404040 !important;
#         color: #e8e8e8 !important;
#         border: 1px solid #555555 !important;
#     }
    
#     /* Spinner */
#     .stSpinner > div {
#         border-top-color: #0066cc !important;
#     }
    
#     /* Progress bar */
#     .stProgress > div > div > div {
#         background-color: #0066cc !important;
#     }
    
#     /* Date input */
#     .stDateInput > div > div > input {
#         background-color: #404040 !important;
#         color: #ffffff !important;
#         border: 1px solid #555555 !important;
#     }
    
#     /* File uploader */
#     .stFileUploader > div {
#         background-color: #404040 !important;
#         border: 1px solid #555555 !important;
#     }
    
#     /* Specific fixes for form elements */
#     .stForm {
#         background-color: #363636 !important;
#         border: 1px solid #555555 !important;
#     }
    
#     .stForm label, .stForm p, .stForm div, .stForm span {
#         color: #e8e8e8 !important;
#     }
    
#     /* Container backgrounds */
#     .element-container {
#         background-color: transparent !important;
#     }
    
#     /* Widget labels - comprehensive targeting */
#     div[data-testid="stMarkdownContainer"] p {
#         color: #e8e8e8 !important;
#     }
    
#     /* Column container text */
#     div[data-testid="column"] {
#         color: #e8e8e8 !important;
#     }
    
#     div[data-testid="column"] p, div[data-testid="column"] span, div[data-testid="column"] div {
#         color: #e8e8e8 !important;
#     }
    
#     /* Placeholder text in inputs */
#     ::placeholder {
#         color: #888888 !important;
#     }
    
#     /* Streamlit native labels */
#     .stTextInput label, .stNumberInput label, .stSelectbox label, 
#     .stTextArea label, .stDateInput label, .stTimeInput label,
#     .stSlider label, .stCheckbox label, .stRadio label {
#         color: #e8e8e8 !important;
#     }
    
#     /* Widget container labels */
#     [data-testid="stWidgetLabel"] {
#         color: #e8e8e8 !important;
#     }
    
#     /* Caption text */
#     [data-testid="caption"] {
#         color: #b8b8b8 !important;
#     }
    
#     /* Markdown container text */
#     [data-testid="stMarkdownContainer"] {
#         color: #e8e8e8 !important;
#     }
    
#     /* Additional text elements */
#     .stSelectbox div[data-baseweb="select"] {
#         background-color: #404040 !important;
#         border-color: #555555 !important;
#     }
    
#     .stSelectbox div[data-baseweb="select"] div {
#         color: #ffffff !important;
#     }
    
#     /* More specific help button targeting */
#     /* Target the actual help button element */
#     button[data-testid*="help"], 
#     div[data-testid*="help"] button,
#     .stHelp button,
#     [aria-label*="help"] {
#         background-color: #555555 !important;
#         color: #ffffff !important;
#         border: 1px solid #777777 !important;
#         border-radius: 50% !important;
#     }
    
#     /* Number input increment/decrement buttons */
#     .stNumberInput button, 
#     input[type="number"]::-webkit-inner-spin-button,
#     input[type="number"]::-webkit-outer-spin-button {
#         background-color: #555555 !important;
#         color: #ffffff !important;
#         border: 1px solid #777777 !important;
#         opacity: 1 !important;
#     }

#     /* Number input button hover */
#     .stNumberInput button:hover {
#         background-color: #666666 !important;
#         color: #ffffff !important;
#     }

#     /* Specific targeting for Streamlit number input controls */
#     .stNumberInput div[data-baseweb="input"] button {
#         background-color: #555555 !important;
#         color: #ffffff !important;
#         border-left: 1px solid #777777 !important;
#     }

#     .stNumberInput div[data-baseweb="input"] button:hover {
#         background-color: #666666 !important;
#     }

#     /* Plus and minus symbols in buttons */
#     .stNumberInput button span,
#     .stNumberInput button svg {
#         color: #ffffff !important;
#         fill: #ffffff !important;
#     }

    
#     /* Help button content (question mark) */
#     button[data-testid*="help"] span,
#     div[data-testid*="help"] button span,
#     .stHelp button span {
#         color: #ffffff !important;
#     }
    
#     /* Force text color for all elements */
#     * {
#         color: #ffffff;
#     }
    
#     /* Override for specific elements that should be white */
#     h1, h2, h3, h4, h5, h6, button {
#         color: #ffffff !important;
#     }
    
#     /* Input text should be white */
#     input, textarea, select {
#         color: #000000 !important;
#     }
#     </style>
#     """, unsafe_allow_html=True)

#Centred Page Title
st.markdown(
    """
    <div style='text-align: center; padding: 20px 0;'>
        <h1>🔧 SetForgeAI — Synthetic Data Generator</h1>
        <p style='font-size: 18px; color: #FFA500; margin-top: 10px;'>
            Generate realistic synthetic datasets for machine learning, testing, and development.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)
## Header with theme toggle
# col_header1, col_header2 = st.columns([4, 1])
# with col_header1:
#     st.title("🔧 SetForgeAI — Synthetic Data Generator")
#     st.markdown("Generate realistic synthetic datasets for machine learning, testing, and development.")

# with col_header2:
#     theme_icon = "🌙" if not st.session_state.dark_theme else "☀️"
#     theme_text = "Dark Mode" if not st.session_state.dark_theme else "Light Mode"
#     if st.button(f"{theme_icon} {theme_text}", key="theme_toggle"):
#         toggle_theme()

# -----------------------------
# Main Configuration Area
# -----------------------------

# Step 1: Basic Settings
st.header("⚙️ Dataset Configuration")

# Main configuration container
config_container = st.container()
with config_container:
    st.subheader("Dataset Configuration")
    
    # Primary settings
    col1, col2 = st.columns([1, 1])
    with col1:
        dataset_name = st.text_input("Dataset Name", value="SampleDataset")
    with col2:
        num_rows = st.number_input(
            "Number of Records",
            min_value=10,
            max_value=1_000_000,
            value=1000,
            step=10
        )
    
    # Separate container for seed configuration
    seed_container = st.container()
    with seed_container:
        st.write("")  # Small spacing
        
        seed_col1, seed_col2 = st.columns([2, 1])
        with seed_col1:
            use_seed = st.checkbox("Use Random Seed", value=False, help="For reproducible results")
        with seed_col2:
            seed_value = None
            if use_seed:
                seed_value = st.number_input("Seed Value", min_value=0, value=42, step=1)

st.divider()

# Step 2: Field definitions
st.header("📋 Define Fields")

field_count = st.number_input("Number of Fields", min_value=1, max_value=20, value=3)
fields = []

for i in range(field_count):
    with st.expander(f"Field {i+1}", expanded=True):
        col1, col2 = st.columns([1, 2])
        
        with col1:
            name = st.text_input(f"Field Name", key=f"name_{i}", placeholder=f"field_{i+1}")
            dtype = st.selectbox(
                "Data Type",
                ["Integer", "Float", "String", "Date/Time", "Categorical", "Boolean"],
                key=f"type_{i}",
            )
        
        with col2:
            field_config = {"name": name or f"field_{i+1}", "type": dtype.lower()}
            
            # Type-specific configurations
            if dtype in ["Integer", "Float"]:
                col2_1, col2_2 = st.columns(2)
                with col2_1:
                    if st.checkbox("Set Min", key=f"use_min_{i}"):
                        field_config["min"] = st.number_input(
                            "Min", key=f"min_{i}", value=0 if dtype == "Integer" else 0.0
                        )
                    if st.checkbox("Set Mean", key=f"use_mean_{i}"):
                        field_config["mean"] = st.number_input("Mean", key=f"mean_{i}", value=50.0)
                with col2_2:
                    if st.checkbox("Set Max", key=f"use_max_{i}"):
                        field_config["max"] = st.number_input(
                            "Max", key=f"max_{i}", value=100 if dtype == "Integer" else 100.0
                        )
                    if st.checkbox("Set Std Dev", key=f"use_std_{i}"):
                        field_config["std"] = st.number_input(
                            "Std Dev", key=f"std_{i}", value=10.0, min_value=0.000001
                        )
                
                if dtype == "Float":
                    field_config["decimal_places"] = st.number_input(
                        "Decimal Places", key=f"decimal_{i}", value=2, min_value=0, max_value=10
                    )
            
            elif dtype == "Categorical":
                categories_input = st.text_input(
                    "Categories (comma-separated)", key=f"cat_{i}", placeholder="Option1, Option2, Option3"
                )
                cats = [c.strip() for c in categories_input.split(",") if c.strip()]
                if cats:
                    field_config["categories"] = cats
                
                if st.checkbox("Custom Probabilities", key=f"use_probs_{i}"):
                    probs_input = st.text_input(
                        "Probabilities (comma-separated, sum≈1)", key=f"probs_{i}", placeholder="0.5, 0.3, 0.2"
                    )
                    if probs_input.strip():
                        field_config["probs"] = [p.strip() for p in probs_input.split(",")]
            
            elif dtype == "String":
                option = st.radio(
                    "Generation Method",
                    ["Random String", "Template", "Regex Pattern"],
                    key=f"string_option_{i}",
                    horizontal=True
                )
                
                if option == "Template":
                    field_config["template"] = st.selectbox(
                        "Template Type",
                        ["name", "email", "phone", "address", "company", "job", "sentence", "paragraph"],
                        key=f"template_{i}",
                    )
                elif option == "Regex Pattern":
                    field_config["pattern"] = st.text_input(
                        "Regex Pattern", key=f"pattern_{i}", placeholder=r"[A-Z]{2}\d{4}", help="Requires 'rstr' library"
                    )
                else:  # Random String
                    field_config["length"] = st.number_input(
                        "String Length", key=f"length_{i}", value=8, min_value=1, max_value=100
                    )
            
            elif dtype == "Date/Time":
                dt_mode = st.selectbox(
                    "Mode", ["Only Date", "Date + Time", "Only Time"], key=f"dt_mode_{i}"
                )
                field_config["datetime_mode"] = {
                    "Only Date": "date",
                    "Date + Time": "datetime",
                    "Only Time": "time"
                }[dt_mode]
                
                if field_config["datetime_mode"] in ("date", "datetime"):
                    col2_1, col2_2 = st.columns(2)
                    with col2_1:
                        field_config["start_date"] = st.date_input(
                            "Start Date", value=date(2020, 1, 1), key=f"start_{i}"
                        )
                    with col2_2:
                        field_config["end_date"] = st.date_input(
                            "End Date", value=date(2024, 1, 1), key=f"end_{i}"
                        )
            
            elif dtype == "Boolean":
                field_config["true_probability"] = st.slider(
                    "Probability of True", min_value=0.0, max_value=1.0, value=0.5, step=0.01, key=f"bool_prob_{i}"
                )
        
        # Data Imperfections Section
        with st.expander("🔧 Data Imperfections (Optional)", expanded=False):
            st.markdown("Add realistic data quality issues for robust ML model testing")
            
            # Missing Data (all field types)
            col_miss1, col_miss2 = st.columns(2)
            with col_miss1:
                if st.checkbox("Missing Data", key=f"use_missing_{i}"):
                    field_config["missing_pct"] = st.slider(
                        "Missing %", min_value=0.0, max_value=50.0, value=5.0, step=0.5, key=f"missing_pct_{i}"
                    )
            with col_miss2:
                if field_config.get("missing_pct"):
                    field_config["missing_pattern"] = st.selectbox(
                        "Missing Pattern",
                        ["random", "block", "tail"],
                        key=f"missing_pattern_{i}",
                        help="random: scattered throughout, block: contiguous chunk, tail: at the end"
                    )
            
            # Outliers (numeric fields only)
            if dtype in ["Integer", "Float"]:
                col_out1, col_out2 = st.columns(2)
                with col_out1:
                    if st.checkbox("Outliers", key=f"use_outliers_{i}"):
                        field_config["outlier_pct"] = st.slider(
                            "Outlier %", min_value=0.0, max_value=20.0, value=2.0, step=0.1, key=f"outlier_pct_{i}"
                        )
                with col_out2:
                    if field_config.get("outlier_pct"):
                        field_config["outlier_method"] = st.selectbox(
                            "Outlier Type",
                            ["extreme", "mild", "random"],
                            key=f"outlier_method_{i}",
                            help="extreme: 3+ std dev, mild: 2-3 std dev, random: outside range"
                        )
            
            # Duplicates (numeric and string fields only)
            if dtype in ["Integer", "Float", "String"]:
                col_dup1, col_dup2 = st.columns(2)
                with col_dup1:
                    if st.checkbox("Extra Duplicates", key=f"use_duplicates_{i}"):
                        field_config["duplicate_pct"] = st.slider(
                            "Duplicate %", min_value=0.0, max_value=50.0, value=5.0, step=0.5, key=f"duplicate_pct_{i}"
                        )
                with col_dup2:
                    if field_config.get("duplicate_pct"):
                        field_config["duplicate_strategy"] = st.selectbox(
                            "Duplicate Strategy",
                            ["common_values", "random_pick", "single_value"],
                            key=f"duplicate_strategy_{i}",
                            help="common_values: use frequent values, random_pick: random existing, single_value: one repeated value"
                        )
        
        fields.append(field_config)

st.divider()

# Step 3: Mathematical Relationships
st.header("🔗 Mathematical Relationships")

numeric_fields = [f["name"] for f in fields if f["type"] in ["integer", "float"]]
relationships = []

if len(numeric_fields) >= 2:
    rel_count = st.number_input(
        "Number of mathematical relationships",
        min_value=0,
        max_value=min(len(numeric_fields)-1, 10),
        value=0,
        help="Create mathematical relationships between numeric fields"
    )
    
    used_targets = set()
    
    for r in range(rel_count):
        with st.expander(f"Mathematical Relationship {r+1}", expanded=True):
            col1, col2, col3 = st.columns([2, 2, 1])
            
            with col1:
                available_targets = [f for f in numeric_fields if f not in used_targets]
                if not available_targets:
                    st.warning("No more fields available as targets.")
                    break
                
                target = st.selectbox(
                    "Target Field (dependent)", available_targets, key=f"rel_target_{r}"
                )
                
                available_sources = [f for f in numeric_fields if f != target]
                source = st.selectbox(
                    "Source Field (independent)", available_sources, key=f"rel_source_{r}"
                )
            
            with col2:
                func_type = st.selectbox(
                    "Relationship Type",
                    [
                        "Direct Proportion",
                        "Logarithmic", 
                        "Exponential (source^k)",
                        "Exponential (k^source)"
                    ],
                    key=f"rel_func_{r}"
                )
                
                # Parameter input based on function type
                if func_type == "Direct Proportion":
                    param = st.number_input(
                        "Multiplier (k)", value=2.0, key=f"rel_param_{r}",
                        help="target = source * k"
                    )
                elif func_type == "Exponential (source^k)":
                    param = st.number_input(
                        "Exponent (k)", value=2.0, key=f"rel_param_{r}",
                        help="target = source^k"
                    )
                elif func_type in ["Logarithmic", "Exponential (k^source)"]:
                    param = st.number_input(
                        "Base", value=2.0, min_value=0.000001, key=f"rel_param_{r}",
                        help="target = log_base(source)" if func_type == "Logarithmic" else "target = base^source"
                    )
            
            with col3:
                noise = st.slider(
                    "Noise Level (%)", 0, 50, 5, key=f"rel_noise_{r}",
                    help="Add random variation to make relationship more realistic"
                )
            
            relationships.append({
                "target": target,
                "source": source,
                "function": func_type,
                "param": param,
                "noise": noise
            })
            used_targets.add(target)
else:
    st.info("Add at least 2 numeric fields to create mathematical relationships.")

st.divider()

# Step 4: Categorical Relationships
st.header("🏷️ Categorical Relationships")

categorical_fields = [f["name"] for f in fields if f["type"] == "categorical"]
categorical_relationships = []

if categorical_fields and numeric_fields:
    cat_rel_count = st.number_input(
        "Number of categorical relationships",
        min_value=0,
        max_value=min(len(categorical_fields) * len(numeric_fields), 10),
        value=0,
        help="Modify numeric field ranges/means based on categorical field values"
    )
    
    for cr in range(cat_rel_count):
        with st.expander(f"Categorical Relationship {cr+1}", expanded=True):
            st.markdown("**Configure how categorical values affect numeric field distributions**")
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                cat_field = st.selectbox(
                    "Categorical Field (source)", categorical_fields, key=f"cat_rel_cat_{cr}"
                )
                target_field = st.selectbox(
                    "Target Numeric Field", numeric_fields, key=f"cat_rel_target_{cr}"
                )
            
            with col2:
                # Get categories for selected categorical field
                selected_cat_field = next((f for f in fields if f["name"] == cat_field), None)
                if selected_cat_field and "categories" in selected_cat_field:
                    categories = selected_cat_field["categories"]
                    st.markdown(f"**Configure values for each category:**")
                    
                    category_mappings = {}
                    for cat in categories:
                        st.markdown(f"**Category: {cat}**")
                        col_range, col_mean = st.columns(2)
                        
                        with col_range:
                            use_range = st.checkbox(f"Use Range", key=f"cat_rel_{cr}_{cat}_use_range")
                            if use_range:
                                min_val = st.number_input(f"Min", key=f"cat_rel_{cr}_{cat}_min", value=0.0)
                                max_val = st.number_input(f"Max", key=f"cat_rel_{cr}_{cat}_max", value=100.0)
                                category_mappings[cat] = {"range": [min_val, max_val]}
                        
                        with col_mean:
                            if not use_range:
                                use_mean = st.checkbox(f"Use Mean/Std", key=f"cat_rel_{cr}_{cat}_use_mean", value=True)
                                if use_mean:
                                    mean_val = st.number_input(f"Mean", key=f"cat_rel_{cr}_{cat}_mean", value=50.0)
                                    std_val = st.number_input(f"Std Dev", key=f"cat_rel_{cr}_{cat}_std", value=10.0, min_value=0.1)
                                    category_mappings[cat] = {"mean": mean_val, "std": std_val}
                    
                    if category_mappings:
                        categorical_relationships.append({
                            "categorical_field": cat_field,
                            "target_field": target_field,
                            "category_mappings": category_mappings
                        })
                else:
                    st.warning("Selected categorical field has no defined categories.")
else:
    if not categorical_fields:
        st.info("Add at least 1 categorical field to create categorical relationships.")
    elif not numeric_fields:
        st.info("Add at least 1 numeric field to create categorical relationships.")
    else:
        st.info("Add both categorical and numeric fields to create categorical relationships.")

st.divider()

# -----------------------------
# Generation and Results
# -----------------------------

st.header("🚀 Generate Dataset")

# Validation
field_issues = validate_fields(fields)
rel_issues = validate_relationships(relationships, fields) if relationships else []
cat_rel_issues = validate_categorical_relationships(categorical_relationships, fields) if categorical_relationships else []
all_issues = field_issues + rel_issues + cat_rel_issues

col1, col2 = st.columns([3, 1])

with col1:
    if all_issues:
        st.error("⚠️ Configuration Issues:")
        for issue in all_issues:
            st.write(f"• {issue}")
    else:
        st.success("✅ Configuration valid!")

with col2:
    # Generate button
    if st.button("🚀 Generate Dataset", type="primary", disabled=bool(all_issues)):
        with st.spinner("Generating synthetic data..."):
            try:
                df = generate_data(
                    fields, 
                    num_rows, 
                    seed=seed_value, 
                    relationships=relationships,
                    categorical_relationships=categorical_relationships
                )
                st.session_state.generated_df = df
                st.session_state.generation_params = {
                    'fields': len(fields),
                    'rows': num_rows,
                    'relationships': len(relationships),
                    'categorical_relationships': len(categorical_relationships),
                    'seed': seed_value
                }
                st.success(f"✅ Generated {len(df)} records with {len(df.columns)} fields!")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Generation failed: {str(e)}")

# Display generated data
if st.session_state.generated_df is not None:
    df = st.session_state.generated_df
    params = st.session_state.generation_params
    
    st.divider()
    
    # Dataset info
    st.header("📊 Generated Dataset")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Records", f"{len(df):,}")
    with col2:
        st.metric("Fields", len(df.columns))
    with col3:
        rel_total = params.get('relationships', 0) + params.get('categorical_relationships', 0)
        st.metric("Total Relationships", rel_total)
    with col4:
        st.metric("Memory", f"{df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    
    # Data preview
    st.subheader("📋 Data Sample")
    
    # Preview controls
    col1, col2 = st.columns([1, 3])
    with col1:
        preview_rows = st.number_input("Rows to show", min_value=5, max_value=100, value=10)
    
    st.dataframe(df.head(preview_rows), use_container_width=True)
    
    # Enhanced statistics section with data quality analysis
    st.subheader("📈 Data Quality & Statistics")
    
    # Analyze data quality
    quality_stats = analyze_data_quality(df)
    
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["📊 Basic Statistics", "🔍 Data Quality Overview", "📋 Detailed Quality Report"])
    
    with tab1:
        # Basic statistics for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            st.write("**Numeric Field Statistics**")
            st.dataframe(df[numeric_cols].describe(), use_container_width=True)
        
        # Categorical statistics
        categorical_cols = df.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            st.write("**Non-Numeric Field Statistics**")
            cat_stats = []
            for col in categorical_cols:
                value_counts = df[col].value_counts()
                if len(value_counts) > 0:
                    most_freq_count = value_counts.iloc[0]
                    freq_percent = f"{(most_freq_count / len(df) * 100):.1f}%"
                else:
                    most_freq_count = 0
                    freq_percent = '0%'

                cat_stats.append({
                    'Field': col,
                    'Unique Values': len(value_counts),
                    'Most Frequent': value_counts.index[0] if len(value_counts) > 0 else 'N/A',
                    'Most Frequent Count': most_freq_count,
                    'Frequency %': freq_percent
                })

            if cat_stats:
                st.dataframe(pd.DataFrame(cat_stats), use_container_width=True)

    
    with tab2:
        # Data quality overview
        st.write("**Data Quality Summary**")
        
        # Create summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        total_missing = sum(stats['missing_count'] for stats in quality_stats.values())
        total_duplicates = sum(stats['duplicate_count'] for stats in quality_stats.values())
        total_outliers = sum(stats.get('outlier_count', 0) for stats in quality_stats.values())
        total_cells = len(df) * len(df.columns)
        
        with col1:
            st.metric("Missing Values", f"{total_missing:,}", f"{total_missing/total_cells*100:.1f}%")
        with col2:
            st.metric("Duplicate Values", f"{total_duplicates:,}", f"{total_duplicates/total_cells*100:.1f}%")
        with col3:
            st.metric("Outliers (Numeric)", f"{total_outliers:,}")
        with col4:
            completeness = (total_cells - total_missing) / total_cells * 100
            st.metric("Data Completeness", f"{completeness:.1f}%")
    
    with tab3:
        # Detailed quality report
        st.write("**Detailed Field Quality Report**")
        
        quality_report = []
        for field_name, stats in quality_stats.items():
            report_row = {
                'Field': field_name,
                'Total Count': stats['total_count'],
                'Missing': f"{stats['missing_count']} ({stats['missing_pct']:.1f}%)",
                'Duplicates': f"{stats['duplicate_count']} ({stats['duplicate_pct']:.1f}%)",
                'Unique Values': stats['unique_count']
            }
            
            # Add outlier info for numeric fields
            if 'outlier_count' in stats:
                report_row['Outliers'] = f"{stats['outlier_count']} ({stats['outlier_pct']:.1f}%)"
                report_row['Mean'] = f"{stats['mean']:.2f}" if 'mean' in stats else 'N/A'
                report_row['Std Dev'] = f"{stats['std']:.2f}" if 'std' in stats else 'N/A'
                report_row['Range'] = f"{stats['min']:.2f} - {stats['max']:.2f}" if 'min' in stats and 'max' in stats else 'N/A'
            else:
                report_row['Outliers'] = 'N/A (Non-numeric)'
                report_row['Mean'] = 'N/A'
                report_row['Std Dev'] = 'N/A'
                report_row['Range'] = 'N/A'
            
            quality_report.append(report_row)
        
        st.dataframe(pd.DataFrame(quality_report), use_container_width=True)
    
    # Download section
    st.subheader("⬇️ Download Options")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        csv_data = df.to_csv(index=False)
        st.download_button(
            label="📄 Download CSV",
            data=csv_data,
            file_name=f"{dataset_name}.csv",
            mime="text/csv"
        )
    
    with col2:
        json_data = df.to_json(orient='records', indent=2)
        st.download_button(
            label="📋 Download JSON",
            data=json_data,
            file_name=f"{dataset_name}.json",
            mime="application/json"
        )
    
    with col3:
        # Excel download requires openpyxl
        try:
            excel_buffer = io.BytesIO()
            df.to_excel(excel_buffer, index=False, engine='openpyxl')
            excel_data = excel_buffer.getvalue()
            st.download_button(
                label="📊 Download Excel",
                data=excel_data,
                file_name=f"{dataset_name}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        except ImportError:
            st.info("Install openpyxl for Excel export")

# -----------------------------
# Tips section
# -----------------------------

st.divider()
st.header("💡 Quick Tips")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    **Field Configuration:**
    • Use realistic value ranges for your specific domain
    • Integer fields are great for IDs, counts, ages, ratings
    • Float fields work well for prices, measurements, percentages
    • Use templates for realistic names, emails, addresses
    
    **Mathematical Relationships:**
    • Create dependencies between related fields (e.g., price vs quantity)
    • Add 5-15% noise to make relationships more realistic
    • Use logarithmic for diminishing returns scenarios
    • Try exponential for growth patterns
    """)

with col2:
    st.markdown("""
    **Categorical Relationships:**
    • Set different ranges/means for numeric fields based on categories
    • Example: Income based on Job Category (Engineer: 70k-120k, Teacher: 40k-60k)
    • Use ranges for distinct value bands per category
    • Use mean/std for normal distributions per category
    
    **Data Quality Features:**
    • Add missing data to test ML model robustness
    • Include outliers to simulate real-world data issues
    • Control duplicate patterns for data cleaning scenarios
    """)

# Footer
st.divider()
st.markdown(
    f"""
    <div style='text-align: center; color: "gray"; padding: 20px;'>
        <p>SetForgeAI - Advanced Synthetic Data Generator</p>
        <p>Built with ❤️ using Streamlit by Anjaneya</p>
    </div>
    """,
    unsafe_allow_html=True
)




