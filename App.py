import streamlit as st
import pandas as pd
import os
import requests
from PIL import Image, ImageDraw, ImageFont
from deep_translator import GoogleTranslator
import json
import time
from io import BytesIO

# Embedded API Keys
GROQ_API_KEY = "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"
HUGGINGFACE_API_KEY = "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"
PEXELS_API_KEY = "fmQ9Eyr6rn8WGagZwhePDSRKHbd2FGELJDuGI19ChGAeU7eAXCrABJ5B"

# Set environment variables
os.environ['GROQ_API_KEY'] = GROQ_API_KEY
os.environ['HUGGINGFACE_API_KEY'] = HUGGINGFACE_API_KEY
os.environ['PEXELS_API_KEY'] = PEXELS_API_KEY

# Page configuration
st.set_page_config(
    page_title="HS Code Finder",
    page_icon="🌾",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .product-card {
        background-color: white;
        padding: 0px;
        margin: 0px;
    }
    .info-label {
        font-weight: bold;
        color: #2c3e50;
        font-size: 16px;
    }
    .info-value {
        color: #34495e;
        font-size: 15px;
        margin-bottom: 12px;
    }
    .stButton>button {
        background-color: #27ae60;
        color: white;
        font-size: 18px;
        padding: 10px 30px;
        border-radius: 8px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #229954;
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 15px;
        margin: 10px 0;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize directories and files
def initialize_app():
    """Create necessary directories and files on first run"""
    if not os.path.exists('images'):
        os.makedirs('images')
    
    if not os.path.exists('hs_data.csv'):
        create_sample_data()
    else:
        try:
            df = pd.read_csv('hs_data.csv')
            required_columns = ['common_name', 'hs_code', 'scientific_name', 'family', 
                              'sub_family', 'order', 'genus', 'kingdom', 'description']
            if not all(col in df.columns for col in required_columns):
                create_sample_data()
        except:
            create_sample_data()

def create_sample_data():
    """Create empty database structure"""
    sample_data = {
        'common_name': [],
        'hs_code': [],
        'scientific_name': [],
        'family': [],
        'sub_family': [],
        'order': [],
        'genus': [],
        'kingdom': [],
        'description': []
    }
    df = pd.DataFrame(sample_data)
    df.to_csv('hs_data.csv', index=False)

def generate_placeholder_image(product_name):
    """Generate a placeholder image for products without images"""
    img = Image.new('RGB', (400, 400), color=(230, 240, 250))
    draw = ImageDraw.Draw(img)
    
    draw.ellipse([100, 100, 300, 300], fill=(100, 180, 100), outline=(50, 150, 50), width=3)
    
    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except:
        font = ImageFont.load_default()
    
    text = product_name.upper()[:20]
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    position = ((400 - text_width) // 2, (400 - text_height) // 2)
    draw.text(position, text, fill=(255, 255, 255), font=font)
    
    return img

def fetch_image_from_unsplash(product_name):
    """Fetch product image from Unsplash (free API)"""
    try:
        # Clean product name for search
        search_term = product_name.lower().replace(' ', '+')
        
        # Try direct Unsplash source (simple method)
        url = f"https://source.unsplash.com/featured/400x400/?{search_term},food,fruit,vegetable,agriculture"
        
        response = requests.get(url, timeout=15, allow_redirects=True)
        
        if response.status_code == 200 and len(response.content) > 1000:
            img = Image.open(BytesIO(response.content))
            # Verify it's not a tiny placeholder
            if img.size[0] > 100 and img.size[1] > 100:
                return img
        
        # Fallback: try alternative search
        url2 = f"https://source.unsplash.com/400x400/?{search_term}"
        response2 = requests.get(url2, timeout=15, allow_redirects=True)
        
        if response2.status_code == 200 and len(response2.content) > 1000:
            img = Image.open(BytesIO(response2.content))
            if img.size[0] > 100 and img.size[1] > 100:
                return img
        
        return None
    except Exception as e:
        return None

def fetch_image_from_pexels(product_name):
    """Fetch product image from Pexels (free API with key)"""
    try:
        api_key = PEXELS_API_KEY
        
        if not api_key:
            return None
        
        url = f"https://api.pexels.com/v1/search?query={product_name}&per_page=1&orientation=square"
        headers = {"Authorization": api_key}
        
        response = requests.get(url, headers=headers, timeout=15)
        
        if response.status_code == 200:
            data = response.json()
            if data.get('photos') and len(data['photos']) > 0:
                image_url = data['photos'][0]['src']['medium']
                img_response = requests.get(image_url, timeout=15)
                
                if img_response.status_code == 200:
                    img = Image.open(BytesIO(img_response.content))
                    # Resize to standard size
                    img = img.resize((400, 400), Image.Resampling.LANCZOS)
                    return img
        
        return None
    except Exception as e:
        return None

def get_product_image(product_name, img_path):
    """Get product image from multiple sources"""
    
    # Try Unsplash first (no API key needed)
    img = fetch_image_from_unsplash(product_name)
    
    if img:
        img.save(img_path)
        return img
    
    # Try Pexels as fallback (if API key is set)
    img = fetch_image_from_pexels(product_name)
    
    if img:
        img.save(img_path)
        return img
    
    # Generate placeholder if all fail
    img = generate_placeholder_image(product_name)
    img.save(img_path)
    return img

def get_info_from_groq(product_name, status_container):
    """
    Get product information using Groq's free LLM API
    """
    try:
        api_key = GROQ_API_KEY
        
        if not api_key:
            return None
        
        url = "https://api.groq.com/openai/v1/chat/completions"
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        prompt = f"""You are an expert in agricultural products and international trade classification. 
Provide ACCURATE information for the product: "{product_name}"

Return ONLY a valid JSON object with this exact structure (no additional text):
{{
    "common_name": "exact common name",
    "hs_code": "4 or 6 digit HS code",
    "scientific_name": "scientific binomial name",
    "family": "taxonomic family",
    "sub_family": "taxonomic sub-family",
    "order": "taxonomic order",
    "genus": "taxonomic genus",
    "kingdom": "Plantae or appropriate kingdom",
    "description": "2-3 sentence description"
}}

IMPORTANT: 
- Use REAL HS codes (Harmonized System codes used in international trade)
- Provide accurate botanical/scientific names
- If you're not certain about any field, use "Not available" as the value
- Return ONLY the JSON object, no markdown formatting, no explanations"""

        data = {
            "model": "llama-3.3-70b-versatile",
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.3,
            "max_tokens": 1000
        }
        
        response = requests.post(url, headers=headers, json=data, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            content = result['choices'][0]['message']['content'].strip()
            
            # Clean up the response - remove markdown code blocks if present
            if content.startswith('```'):
                content = content.split('```')[1]
                if content.startswith('json'):
                    content = content[4:]
                content = content.strip()
            
            # Parse JSON
            product_info = json.loads(content)
            status_container.success("✅ Information retrieved from Groq AI")
            return product_info
        else:
            return None
            
    except json.JSONDecodeError as e:
        return None
    except Exception as e:
        return None

def get_info_from_huggingface(product_name, status_container):
    """
    Fallback: Get product information using Hugging Face Inference API (free)
    """
    try:
        api_key = HUGGINGFACE_API_KEY
        
        if not api_key:
            return None
        
        url = "https://api-inference.huggingface.co/models/mistralai/Mixtral-8x7B-Instruct-v0.1"
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        prompt = f"""<s>[INST] You are an expert in agricultural products and trade classification.
Provide accurate information for: "{product_name}"

Return only valid JSON:
{{
    "common_name": "name",
    "hs_code": "code",
    "scientific_name": "name",
    "family": "family",
    "sub_family": "sub-family",
    "order": "order",
    "genus": "genus",
    "kingdom": "kingdom",
    "description": "description"
}}
[/INST]"""

        data = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 500,
                "temperature": 0.3,
                "return_full_text": False
            }
        }
        
        response = requests.post(url, headers=headers, json=data, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            content = result[0]['generated_text'].strip()
            
            # Extract JSON from response
            start_idx = content.find('{')
            end_idx = content.rfind('}') + 1
            if start_idx != -1 and end_idx > start_idx:
                json_str = content[start_idx:end_idx]
                product_info = json.loads(json_str)
                status_container.success("✅ Information retrieved from Hugging Face")
                return product_info
        
        return None
            
    except Exception as e:
        return None

def get_info_from_ollama(product_name, status_container):
    """
    Get information from local Ollama instance (if running)
    """
    try:
        url = "http://localhost:11434/api/generate"
        
        prompt = f"""Provide accurate agricultural and trade information for: "{product_name}"

Return ONLY valid JSON (no markdown, no extra text):
{{
    "common_name": "exact name",
    "hs_code": "harmonized system code",
    "scientific_name": "scientific name",
    "family": "family",
    "sub_family": "sub-family",
    "order": "order",
    "genus": "genus",
    "kingdom": "kingdom",
    "description": "2-3 sentences"
}}"""

        data = {
            "model": "llama2",
            "prompt": prompt,
            "stream": False,
            "format": "json"
        }
        
        response = requests.post(url, json=data, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            product_info = json.loads(result['response'])
            status_container.success("✅ Information retrieved from Ollama")
            return product_info
        
        return None
            
    except Exception as e:
        return None

def get_product_info_with_ai(product_name, status_container):
    """Try multiple AI sources to get accurate information"""
    
    # Try Groq first (best quality, free tier available)
    status_container.info("🤖 Using AI to fetch accurate product information...")
    product_info = get_info_from_groq(product_name, status_container)
    
    if product_info:
        return product_info
    
    # Try Hugging Face as fallback
    status_container.info("🔄 Trying Hugging Face API...")
    product_info = get_info_from_huggingface(product_name, status_container)
    
    if product_info:
        return product_info
    
    # Try local Ollama as final fallback
    status_container.info("🔄 Checking for local Ollama instance...")
    product_info = get_info_from_ollama(product_name, status_container)
    
    if product_info:
        return product_info
    
    # If all fail, return None
    return None

def save_to_csv(product_info, status_container):
    """Save new product information to CSV"""
    try:
        df = pd.read_csv('hs_data.csv')
        
        # Check if product already exists
        existing = df[df['common_name'].str.lower() == product_info['common_name'].lower()]
        
        if not existing.empty:
            # Update existing entry
            df.loc[df['common_name'].str.lower() == product_info['common_name'].lower()] = list(product_info.values())
        else:
            # Add new entry
            new_df = pd.DataFrame([product_info])
            df = pd.concat([df, new_df], ignore_index=True)
        
        df.to_csv('hs_data.csv', index=False)
        status_container.success("💾 Product information saved to database!")
        return True
    except Exception as e:
        return False

def translate_text(text, target_lang):
    """Translate text to target language"""
    if target_lang == 'en' or not text or text in ['Not available', 'Not found']:
        return text
    
    try:
        translator = GoogleTranslator(source='en', target=target_lang)
        translated = translator.translate(text)
        return translated
    except Exception as e:
        return f"{text}"

def search_product(product_name, language_code, status_container):
    """Search for product in database or use AI"""
    try:
        # Load existing data
        df = pd.read_csv('hs_data.csv')
        
        if len(df) > 0:
            # Search in existing data (case-insensitive)
            result = df[df['common_name'].str.lower() == product_name.lower()]
            
            if not result.empty:
                status_container.info("✅ Found in local database")
                product_info = result.iloc[0].to_dict()
                
                # Translate if needed
                if language_code != 'en':
                    product_info['description'] = translate_text(product_info['description'], language_code)
                    product_info['family'] = translate_text(product_info['family'], language_code)
                    product_info['sub_family'] = translate_text(product_info['sub_family'], language_code)
                    product_info['order'] = translate_text(product_info['order'], language_code)
                    product_info['genus'] = translate_text(product_info['genus'], language_code)
                    product_info['kingdom'] = translate_text(product_info['kingdom'], language_code)
                
                return product_info
        
        # Not in database - use AI
        status_container.info("🔍 Product not found in local database. Using AI to fetch information...")
        product_info = get_product_info_with_ai(product_name, status_container)
        
        if product_info:
            # Validate the structure
            required_fields = ['common_name', 'hs_code', 'scientific_name', 'family', 
                             'sub_family', 'order', 'genus', 'kingdom', 'description']
            
            # Ensure all fields exist
            for field in required_fields:
                if field not in product_info:
                    product_info[field] = 'Not available'
            
            # Save to CSV for future use
            save_to_csv(product_info, status_container)
            
            # Translate if needed
            if language_code != 'en':
                product_info['description'] = translate_text(product_info['description'], language_code)
                product_info['family'] = translate_text(product_info['family'], language_code)
                product_info['sub_family'] = translate_text(product_info['sub_family'], language_code)
                product_info['order'] = translate_text(product_info['order'], language_code)
                product_info['genus'] = translate_text(product_info['genus'], language_code)
                product_info['kingdom'] = translate_text(product_info['kingdom'], language_code)
            
            return product_info
        else:
            return None
    
    except Exception as e:
        status_container.error(f"❌ Error searching product: {str(e)}")
        return None

# Initialize app
initialize_app()

# Header
st.title("🌾 HS Code Finder for Agriculture & Products")
st.markdown("### AI-Powered Search for HS codes, scientific names, and classifications")

# Language selection
languages = {
    'English': 'en',
    'Tamil': 'ta',
    'Hindi': 'hi',
    'French': 'fr',
    'Spanish': 'es',
    'German': 'de',
    'Chinese': 'zh-CN',
    'Japanese': 'ja',
    'Arabic': 'ar'
}

col1, col2 = st.columns([3, 1])
with col1:
    product_input = st.text_input("🔍 Enter product name:", placeholder="e.g., Apple, Banana, Rice, Coconut Oil, Turmeric...", key="product_search")
with col2:
    selected_language = st.selectbox("🌐 Language:", list(languages.keys()))

search_button = st.button("🔎 Search", use_container_width=True)

# Trigger search on Enter key press
if product_input and product_input != st.session_state.get('last_search', ''):
    search_button = True
    st.session_state['last_search'] = product_input

# Example products
st.markdown("**💡 Try searching for:** Apple, Banana, Rice, Wheat, Turmeric, Cardamom, Coffee, Tea, Coconut Oil, etc.")

# Search functionality
if search_button and product_input:
    with st.spinner("🔍 Searching for product information..."):
        language_code = languages[selected_language]
        
        # Create placeholder for results
        result_placeholder = st.empty()
        
        # Create status container BELOW the search
        status_container = st.container()
        
        product_info = search_product(product_input, language_code, status_container)
        
        if product_info:
            with result_placeholder.container():
                # Display results
                col_img, col_info = st.columns([1, 2], vertical_alignment="top")
                
                with col_img:
                    # Check for existing image or fetch/generate new one
                    img_path = f"images/{product_info['common_name'].lower().replace(' ', '_')}.png"
                    
                    if os.path.exists(img_path):
                        st.image(img_path, use_container_width=True)
                    else:
                        # Fetch or generate image
                        with st.spinner("📷 Fetching product image..."):
                            img = get_product_image(product_info['common_name'], img_path)
                            st.image(img, use_container_width=True)
                
                with col_info:
                    st.markdown('<div class="product-card">', unsafe_allow_html=True)
                    
                    st.markdown(f"<p class='info-label'>Common Name:</p><p class='info-value'>{product_info['common_name']}</p>", unsafe_allow_html=True)
                    st.markdown(f"<p class='info-label'>HS Code:</p><p class='info-value'>{product_info['hs_code']}</p>", unsafe_allow_html=True)
                    st.markdown(f"<p class='info-label'>Category (Family):</p><p class='info-value'>{product_info['family']}</p>", unsafe_allow_html=True)
                    st.markdown(f"<p class='info-label'>Sub Category (Sub Family):</p><p class='info-value'>{product_info['sub_family']}</p>", unsafe_allow_html=True)
                    st.markdown(f"<p class='info-label'>Scientific Name:</p><p class='info-value'><em>{product_info['scientific_name']}</em></p>", unsafe_allow_html=True)
                    st.markdown(f"<p class='info-label'>Order:</p><p class='info-value'>{product_info['order']}</p>", unsafe_allow_html=True)
                    st.markdown(f"<p class='info-label'>Genus:</p><p class='info-value'>{product_info['genus']}</p>", unsafe_allow_html=True)
                    st.markdown(f"<p class='info-label'>Kingdom:</p><p class='info-value'>{product_info['kingdom']}</p>", unsafe_allow_html=True)
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Description section
                st.markdown('<div class="product-card">', unsafe_allow_html=True)
                st.markdown(f"<p class='info-label'>Description:</p>", unsafe_allow_html=True)
                st.write(product_info['description'])
                st.markdown('</div>', unsafe_allow_html=True)
            
        else:
            status_container.error("❌ Unable to fetch product information. Please try again later.")

elif search_button and not product_input:
    st.warning("⚠️ Please enter a product name to search.")

# Footer
st.markdown("---")
st.markdown("💡 **Powered by AI:** This app uses advanced language models to provide accurate, real-time product information")
st.markdown("📊 **Data Source:** AI-generated information with local caching for faster subsequent searches")
