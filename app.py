"""
Multi-Modal Attribute Extraction & Retrieval for Secondhand E-Commerce
Streamlit Application - Final Version with Enhanced Analytics
"""

import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys
import os
import re

# --- 1. path config (critical for new structure) ---
# add src and scrapers to path for imports
current_dir = Path(__file__).parent
sys.path.append(str(current_dir / "src"))
sys.path.append(str(current_dir / "scrapers"))

# --- 2. updated imports ---
# importing from new folders
from src.model_inference import MultiModalModel
from src.data_utils import load_catalog, get_sample_noisy_description
from scrapers.poshmark_scraper import PoshmarkScraper

# page config
st.set_page_config(
    page_title="Secondhand Fashion AI",
    page_icon="👗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# custom css for styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .confidence-high {
        color: #28a745;
        font-weight: bold;
    }
    .confidence-medium {
        color: #ffc107;
        font-weight: bold;
    }
    .confidence-low {
        color: #dc3545;
        font-weight: bold;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# init session state
if 'processed' not in st.session_state:
    st.session_state.processed = False
if 'uploaded_image' not in st.session_state:
    st.session_state.uploaded_image = None
if 'original_text' not in st.session_state:
    st.session_state.original_text = ""
if 'input_method' not in st.session_state:
    st.session_state.input_method = "upload"
if 'scraped_data' not in st.session_state:
    st.session_state.scraped_data = None
if 'scraped_images' not in st.session_state:
    st.session_state.scraped_images = []
if 'selected_image_idx' not in st.session_state:
    st.session_state.selected_image_idx = 0

# --- sidebar ui (moved up) ---
# define before loading model so it shows at top
st.sidebar.markdown("### 📖 Instructions")
st.sidebar.markdown("""
1. Choose input method (Upload or Poshmark Link)
2. Provide image and description
3. Click "Process Listing" to analyze
4. View extracted attributes and similar items
""")
st.sidebar.markdown("---")

# validation mode toggle
validation_mode = st.sidebar.checkbox(
    "Enable Validation Mode",
    help="Compare predictions against ground truth labels from Fashion-IQ dataset"
)
st.sidebar.markdown("---")
# -----------------------------

# init scraper first (needed for model)
@st.cache_resource
def load_scraper():
    """Load the Poshmark scraper (with caching)"""
    return PoshmarkScraper()

scraper = load_scraper()

# load model with scraper
@st.cache_resource
def load_model():
    """Load the multi-modal model (with caching)"""
    # prints to sidebar on init, so appears below instructions
    return MultiModalModel(scraper=scraper)

@st.cache_data
def load_catalog_data():
    """Load catalog for similarity search (kept for backward compatibility, not used for Module B)"""
    return load_catalog()

model = load_model()
catalog_df = load_catalog_data()

# ==========================================
# main content
# ==========================================
st.markdown('<div class="main-header">👗 Multi-Modal Attribute Extraction & Retrieval</div>', unsafe_allow_html=True)
st.markdown("### Solving the Discovery Problem in Sustainable Fashion")

# ==========================================
# input module -> noisy listing
# ==========================================
st.markdown("---")
st.markdown("## 📥 Input Module - The \"Noisy\" Listing")

# input method selection
input_method = st.radio(
    "Choose Input Method:",
    ["📤 Upload Image & Text", "🔗 Poshmark Link"],
    horizontal=True,
    help="Select how you want to provide the listing information"
)
st.session_state.input_method = "upload" if input_method == "📤 Upload Image & Text" else "poshmark"

# conditional input based on method
if st.session_state.input_method == "upload":
    # traditional upload method
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### Image Upload")
        uploaded_file = st.file_uploader(
            "Upload a clothing item image",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a single image (JPG/PNG) of a clothing item"
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.session_state.uploaded_image = image
            st.image(image, caption="Uploaded Image", width=400)
        elif validation_mode:
            st.info("💡 Validation Mode: Using sample image from Fashion-IQ dataset")

    with col2:
        st.markdown("### Seller Description")
        sample_text = get_sample_noisy_description()
        
        seller_text = st.text_area(
            "Enter seller description (or use sample):",
            value=sample_text if not st.session_state.original_text else st.session_state.original_text,
            height=200,
            help="Enter the noisy, unstructured description from the seller"
        )
        st.session_state.original_text = seller_text
    
    # process button
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        process_button_upload = st.button("🚀 Process Listing", type="primary", use_container_width=True, key="process_upload")
    uploaded_file_check = uploaded_file

else:
    # init for poshmark method
    process_button_upload = False
    uploaded_file_check = None
    # poshmark link scraping
    st.markdown("### 🔗 Poshmark Listing URL")
    poshmark_url = st.text_input(
        "Paste Poshmark listing URL:",
        placeholder="https://poshmark.com/listing/...",
        help="Enter the full URL of a Poshmark listing to automatically extract item information"
    )
    
    scrape_button = st.button("🔍 Scrape Listing", type="primary")
    
    # handle scraping
    if scrape_button:
        if not poshmark_url:
            st.error("❌ Please enter a Poshmark URL")
        else:
            with st.spinner("🕷️ Scraping Poshmark listing... This may take a few seconds."):
                scraped_data = scraper.scrape_listing(poshmark_url)
                st.session_state.scraped_data = scraped_data
                
                if scraped_data['success']:
                    st.success("✅ Successfully scraped listing!")
                    
                    # display scraped info
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        st.markdown("#### 📋 Scraped Information")
                        if scraped_data['title']:
                            st.write(f"**Title:** {scraped_data['title']}")
                        if scraped_data['brand']:
                            st.write(f"**Brand:** {scraped_data['brand']}")
                        if scraped_data['size']:
                            st.write(f"**Size:** {scraped_data['size']}")
                        if scraped_data['price']:
                            st.write(f"**Price:** {scraped_data['price']}")
                    
                    with col2:
                        # display scraped images
                        if scraped_data['images']:
                            st.markdown(f"#### 🖼️ Scraped Images ({len(scraped_data['images'])} found)")
                            
                            # download all images
                            with st.spinner("Downloading images..."):
                                downloaded_images = []
                                for idx, img_url in enumerate(scraped_data['images']):
                                    try:
                                        img = scraper.download_image(img_url)
                                        if img:
                                            downloaded_images.append(img)
                                    except Exception as e:
                                        st.caption(f"Could not download image {idx+1}")
                                
                                st.session_state.scraped_images = downloaded_images
                            
                            # display image gallery with selection
                            if st.session_state.scraped_images:
                                num_images = len(st.session_state.scraped_images)
                                
                                # image selector
                                if num_images > 1:
                                    selected_idx = st.selectbox(
                                        "Select image to use for processing:",
                                        range(num_images),
                                        format_func=lambda x: f"Image {x+1}",
                                        key="image_selector_new",
                                        index=st.session_state.selected_image_idx
                                    )
                                    st.session_state.selected_image_idx = selected_idx
                                else:
                                    selected_idx = 0
                                    st.session_state.selected_image_idx = 0
                                
                                # display selected image
                                selected_image = st.session_state.scraped_images[selected_idx]
                                st.image(selected_image, caption=f"Selected Image {selected_idx+1} of {num_images}", width=300)
                                st.session_state.uploaded_image = selected_image
                                
                                # show gallery of all images in columns
                                if num_images > 1:
                                    st.markdown("**All Images:**")
                                    gallery_cols = st.columns(min(5, num_images))
                                    for idx, img in enumerate(st.session_state.scraped_images):
                                        with gallery_cols[idx % 5]:
                                            # highlight selected image
                                            border = "✅ Selected" if idx == selected_idx else ""
                                            st.image(img, caption=f"{idx+1}", width=100)
                                            if idx == selected_idx:
                                                st.caption("✓")
                            else:
                                st.warning("Could not download any images")
                        
                    # pre-fill description
                    if scraped_data['description']:
                        st.markdown("#### 📝 Scraped Description")
                        edited_description = st.text_area(
                            "Edit description if needed:",
                            value=scraped_data['description'],
                            height=150,
                            key="scraped_description"
                        )
                        st.session_state.original_text = edited_description
                    
                    # process button for scraped data
                    st.markdown("---")
                    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
                    with col_btn2:
                        process_button_scraped = st.button("🚀 Process Scraped Listing", type="primary", use_container_width=True, key="process_scraped")
                else:
                    st.error(f"❌ Scraping failed: {scraped_data.get('error', 'Unknown error')}")
                    process_button_scraped = False
                    uploaded_file_check = None
    else:
        # if we have previously scraped data, show it
        if st.session_state.scraped_data and st.session_state.scraped_data.get('success'):
            scraped_data = st.session_state.scraped_data
            
            col1, col2 = st.columns([1, 1])
            with col1:
                st.markdown("#### 📋 Previously Scraped Information")
                if scraped_data['title']:
                    st.write(f"**Title:** {scraped_data['title']}")
                if scraped_data['size']:
                    st.write(f"**Size:** {scraped_data['size']}")
            
            with col2:
                # display scraped images if available
                if st.session_state.scraped_images:
                    num_images = len(st.session_state.scraped_images)
                    # image selector
                    if num_images > 1:
                        selected_idx = st.selectbox(
                            "Select image to use for processing:",
                            range(num_images),
                            format_func=lambda x: f"Image {x+1}",
                            key="image_selector_saved",
                            index=st.session_state.selected_image_idx
                        )
                        st.session_state.selected_image_idx = selected_idx
                    else:
                        selected_idx = 0
                    
                    if selected_idx < len(st.session_state.scraped_images):
                        selected_image = st.session_state.scraped_images[selected_idx]
                        st.image(selected_image, caption=f"Image {selected_idx+1} of {num_images}", width=300)
                        st.session_state.uploaded_image = selected_image
                    
                    if num_images > 1:
                        st.markdown("**All Images:**")
                        gallery_cols = st.columns(min(5, num_images))
                        for idx, img in enumerate(st.session_state.scraped_images):
                            with gallery_cols[idx % 5]:
                                st.image(img, caption=f"{idx+1}", width=80)
                                if idx == selected_idx:
                                    st.caption("✓ Selected")
                elif st.session_state.uploaded_image:
                    st.image(st.session_state.uploaded_image, caption="Scraped Image", width=300)
            
            if st.session_state.original_text:
                edited_description = st.text_area(
                    "Edit description if needed:",
                    value=st.session_state.original_text,
                    height=150,
                    key="scraped_description_edit"
                )
                st.session_state.original_text = edited_description
            
            col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
            with col_btn2:
                process_button_scraped_saved = st.button("🚀 Process Scraped Listing", type="primary", use_container_width=True, key="process_scraped_saved")
            uploaded_file_check = None
        else:
            process_button_scraped_saved = False
            uploaded_file_check = None

# ==========================================
# processing & output
# ==========================================
try:
    _ = process_button_upload
except NameError:
    process_button_upload = False

try:
    _ = process_button_scraped
except NameError:
    process_button_scraped = False

try:
    _ = process_button_scraped_saved
except NameError:
    process_button_scraped_saved = False

try:
    _ = uploaded_file_check
except NameError:
    uploaded_file_check = None

# combine process button states
process_button = (
    (st.session_state.input_method == "upload" and process_button_upload) or
    (st.session_state.input_method == "poshmark" and process_button_scraped) or
    (st.session_state.input_method == "poshmark" and process_button_scraped_saved)
)

has_valid_input = (
    (st.session_state.input_method == "upload" and (uploaded_file_check is not None or validation_mode)) or
    (st.session_state.input_method == "poshmark" and st.session_state.scraped_data and st.session_state.scraped_data.get('success')) or
    validation_mode
)

if process_button and has_valid_input:
    with st.spinner("🤖 Processing listing... Generating embeddings, extracting attributes, and searching Poshmark..."):
        import time
        # time.sleep(1) # removed sleep for faster performance
        
        text_description = st.session_state.original_text or ""
        
        # 1. run ai model first to get clean data
        if st.session_state.uploaded_image or validation_mode:
            if validation_mode:
                results = model.predict_validation_mode()
            else:
                results = model.predict(
                    image=st.session_state.uploaded_image,
                    text=text_description if text_description else ""
                )
            
            # --- UPDATED LOGIC: Exact Search based on Cleaned Text ---
            
            # 2. Use the clean description directly for search
            search_query = results["clean_description"]
            
            # feedback to user
            st.toast(f"Searching Poshmark for: '{search_query}'")
            print(f"DEBUG: Optimized Search Query: {search_query}")

            # 3. execute search using existing scraper
            try:
                # add gender if extracted, otherwise defaults
                # strict=False allows fuzzy matching
                new_search_results = scraper.search_poshmark(search_query, top_k=5)
                
                # update results with live scraper data if found
                if new_search_results:
                    results["similar_items"] = new_search_results
                
            except Exception as e:
                print(f"Search Error: {e}")
                # fallback to keep existing results/empty list

            # --- end updated logic ---

            st.session_state.processed = True
            st.session_state.results = results
            st.rerun()

# display results if processed
if st.session_state.processed and 'results' in st.session_state:
    results = st.session_state.results
    
    # ==========================================
    # output module a -> structured attribute verification
    # ==========================================
    st.markdown("---")
    st.markdown("## ✅ Output Module A: Structured Attribute Verification")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 📝 Original (Noisy) Description")
        st.info(f'"{st.session_state.original_text}"')
    
    with col2:
        st.markdown("### ✨ Generated (Clean) Description")
        st.success(f'"{results["clean_description"]}"')
    
    # attribute extraction table
    st.markdown("### 🏷️ Extracted Attributes")
    
    attributes_data = results["attributes"]
    attributes_df = pd.DataFrame([
        {
            "Attribute": "Category",
            "Predicted Value": attributes_data["category"]["value"],
            "Confidence": attributes_data["category"]["confidence"],
            "Ground Truth": attributes_data["category"].get("ground_truth", None) if validation_mode else None
        },
        {
            "Attribute": "Color",
            "Predicted Value": attributes_data["color"]["value"],
            "Confidence": attributes_data["color"]["confidence"],
            "Ground Truth": attributes_data["color"].get("ground_truth", None) if validation_mode else None
        },
        {
            "Attribute": "Material",
            "Predicted Value": attributes_data["material"]["value"],
            "Confidence": attributes_data["material"]["confidence"],
            "Ground Truth": attributes_data["material"].get("ground_truth", None) if validation_mode else None
        },
        {
            "Attribute": "Style",
            "Predicted Value": attributes_data["style"]["value"],
            "Confidence": attributes_data["style"]["confidence"],
            "Ground Truth": attributes_data["style"].get("ground_truth", None) if validation_mode else None
        },
        {
            "Attribute": "Condition",
            "Predicted Value": attributes_data["condition"]["value"],
            "Confidence": attributes_data["condition"]["confidence"],
            "Ground Truth": attributes_data["condition"].get("ground_truth", None) if validation_mode else None
        },
    ])
    
    for idx, row in attributes_df.iterrows():
        col1, col2, col3 = st.columns([2, 3, 2])
        with col1:
            st.markdown(f"**{row['Attribute']}**")
        with col2:
            confidence = row['Confidence']
            if confidence >= 80:
                badge_color = "🟢"
                css_class = "confidence-high"
            elif confidence >= 60:
                badge_color = "🟡"
                css_class = "confidence-medium"
            else:
                badge_color = "🔴"
                css_class = "confidence-low"
            
            if validation_mode and row['Ground Truth']:
                ground_truth = row['Ground Truth']
                is_correct = row['Predicted Value'].lower() == ground_truth.lower()
                checkmark = "✅" if is_correct else "❌"
                st.markdown(f"{checkmark} **Predicted:** {row['Predicted Value']} ({confidence}%) | **Actual:** {ground_truth}")
            else:
                st.markdown(f"{badge_color} **{row['Predicted Value']}** ({confidence}%)")
        with col3:
            st.markdown(f'<span class="{css_class}">{confidence}%</span>', unsafe_allow_html=True)
        st.markdown("---")
    
    # ==========================================
    # output module b -> similarity search & retrieval
    # ==========================================
    st.markdown("---")
    st.markdown("## 🔍 Output Module B: Similarity Search & Retrieval")
    
    # ranked results grid
    st.markdown("### 🎯 Top-5 Similar Items Found on Poshmark")
    
    similar_items = results["similar_items"]
    
    if not similar_items:
        st.warning("⚠️ No similar items found. The search may have failed or returned no results.")
    else:
        cols = st.columns(5)
        for idx, item in enumerate(similar_items[:5]):
            with cols[idx]:
                display_image = None
                if item.get("image_url"):
                    try:
                        # attempt to download image directly since we are live searching
                        display_image = scraper.download_image(item["image_url"])
                    except Exception as e:
                        st.caption(f"Could not load image: {str(e)[:50]}")
                
                # fallback for internal images if not live search
                if not display_image and item.get("image_path"):
                    try:
                        display_image = Image.open(item["image_path"])
                    except:
                        pass
                
                if display_image:
                    st.image(display_image, use_container_width=True)
                else:
                    st.image(Image.new('RGB', (200, 200), color='lightgray'), use_container_width=True)
                
                title = item.get("title") or item.get("description", "No title")
                st.markdown(f"**{title[:50]}{'...' if len(title) > 50 else ''}**")
                
                if item.get("brand"):
                    st.caption(f"🏷️ {item['brand']}")
                
                if item.get("price"):
                    st.markdown(f"💰 **{item['price']}**")
                
                if item.get("size"):
                    st.caption(f"📏 Size: {item['size']}")
                
                description = item.get("description", "No description")
                if description != title:
                    st.caption(f"*{description[:60]}{'...' if len(description) > 60 else ''}*")
                
                if item.get("listing_url"):
                    st.markdown(f"[🔗 View on Poshmark]({item['listing_url']})")
                else:
                    st.caption(f"ID: {item.get('item_id', 'N/A')}")
        
        st.markdown("#### 📊 Detailed Search Results")
        similarity_df = pd.DataFrame([
            {
                "Rank": i+1,
                "Title": item.get("title", item.get("description", "N/A"))[:50],
                "Brand": item.get("brand", "N/A"),
                "Price": item.get("price", "N/A"),
                "Size": item.get("size", "N/A"),
                "Listing URL": item.get("listing_url", "N/A")[:40] + "..."
            }
            for i, item in enumerate(similar_items[:5])
        ])
        st.dataframe(similarity_df, use_container_width=True, hide_index=True)

    # ==========================================
    # output module c -> analytics & insights
    # ==========================================
    st.markdown("---")
    st.markdown("## 📈 Output Module C: Analytics & Insights")
    
    tab1, tab2, tab3 = st.tabs(["🧠 Model Confidence", "💰 Price Analysis", "🗺️ Embedding Space"])
    
    # tab 1 -> confidence radar
    with tab1:
        st.markdown("##### Visualizing model certainty across attributes")
        try:
            attr_keys = list(results['attributes'].keys())
            attr_vals = [results['attributes'][k]['confidence'] for k in attr_keys]
            # close the loop
            attr_keys += [attr_keys[0]]
            attr_vals += [attr_vals[0]]
            
            fig = go.Figure(data=go.Scatterpolar(
                r=attr_vals,
                theta=[k.title() for k in attr_keys],
                fill='toself',
                name='Confidence',
                line_color='#1f77b4'
            ))
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                showlegend=False,
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Could not generate radar chart: {e}")

    # tab 2 -> price analysis (improved w/ box plot)
    with tab2:
        st.markdown("##### Price distribution of similar items found on Poshmark")
        price_data = []
        for item in similar_items:
            try:
                p_str = str(item.get('price', ''))
                # extract number from string (e.g. "$25")
                p_val = float(re.sub(r'[^\d.]', '', p_str))
                price_data.append({
                    "Price": p_val,
                    "Title": item.get('title', 'Item'),
                    "Brand": item.get('brand', 'N/A')
                })
            except:
                continue
        
        if price_data:
            df_prices = pd.DataFrame(price_data)
            col_metrics1, col_metrics2 = st.columns(2)
            avg_price = df_prices["Price"].mean()
            min_price = df_prices["Price"].min()
            max_price = df_prices["Price"].max()
            
            with col_metrics1:
                st.metric("Average Market Price", f"${avg_price:.2f}")
            with col_metrics2:
                st.metric("Price Range", f"${min_price:.0f} - ${max_price:.0f}")
                
            # visualization -> box plot w/ points (jitter)
            fig_price = px.box(
                df_prices,
                x="Price",  # horizontal for better readability of price range
                points="all", # show individual points
                hover_data=["Title", "Brand"],
                title="Market Price Distribution",
                labels={"Price": "Price ($)"},
                color_discrete_sequence=['#2ca02c']
            )
            fig_price.add_vline(x=avg_price, line_dash="dash", line_color="red", annotation_text="Avg")
            st.plotly_chart(fig_price, use_container_width=True)
        else:
            st.info("Not enough price data collected from search results.")

    # tab 3 -> embedding visualization (t-sne)
    with tab3:
        st.markdown("##### 2D Projection of Semantic Similarity")
        if results.get('embeddings_vis') is not None:
            st.plotly_chart(results['embeddings_vis'], use_container_width=True)
        else:
            st.info("ℹ️ Visualization not available. (Ensure 'scikit-learn' is installed and you have enough inventory data).")

elif process_button:
    if st.session_state.input_method == "upload":
        if st.session_state.uploaded_image is None and not validation_mode:
            st.error("❌ Please upload an image before processing!")
    else:
        if not st.session_state.scraped_data or not st.session_state.scraped_data.get('success'):
            st.error("❌ Please scrape a Poshmark listing first!")

# footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray; padding: 20px;'>
    Multi-Modal Attribute Extraction & Retrieval System | Intro to AI
    </div>
    """,
    unsafe_allow_html=True
)