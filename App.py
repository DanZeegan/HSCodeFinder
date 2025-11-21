import streamlit as st
import pandas as pd
import os
import requests
from PIL import Image, ImageDraw, ImageFont
from deep_translator import GoogleTranslator
import json
import time
from io import BytesIO

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
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 20px 0;
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

def fetch_image_from_wikipedia(product_name):
    """Fetch product image from Wikipedia (most accurate)"""
    try:
        # Clean product name
        search_term = product_name.replace(' ', '_')
        
        # Search Wikipedia for the article
        search_url = f"https://en.wikipedia.org/w/api.php"
        search_params = {
            'action': 'query',
            'format': 'json',
            'titles': search_term,
            'prop': 'pageimages',
            'pithumbsize': 400
        }
        
        response = requests.get(search_url, params=search_params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            pages = data.get('query', {}).get('pages', {})
            
            for page_id, page_data in pages.items():
                if 'thumbnail' in page_data:
                    image_url = page_data['thumbnail']['source']
                    
                    # Download the image
                    img_response = requests.get(image_url, timeout=10)
                    if img_response.status_code == 200:
                        img = Image.open(BytesIO(img_response.content))
                        # Resize to standard size
                        img = img.resize((400, 400), Image.Resampling.LANCZOS)
                        return img
        
        return None
    except Exception as e:
        return None

def fetch_image_from_unsplash(product_name):
    """Fetch product image from Unsplash (free API)"""
    try:
        # Clean product name for search
        search_term = product_name.lower().replace(' ', '+')
        
        # List of search strategies to try
        search_urls = [
            f"https://source.unsplash.com/featured/400x400/?{search_term},plant",
            f"https://source.unsplash.com/featured/400x400/?{search_term},flower",
            f"https://source.unsplash.com/featured/400x400/?{search_term},fruit",
            f"https://source.unsplash.com/featured/400x400/?{search_term},food",
            f"https://source.unsplash.com/featured/400x400/?{search_term},agriculture",
            f"https://source.unsplash.com/400x400/?{search_term}",
        ]
        
        for url in search_urls:
            try:
                response = requests.get(url, timeout=10, allow_redirects=True)
                
                if response.status_code == 200 and len(response.content) > 5000:
                    img = Image.open(BytesIO(response.content))
                    # Verify it's a valid image
                    if img.size[0] >= 200 and img.size[1] >= 200:
                        # Resize to standard size
                        img = img.resize((400, 400), Image.Resampling.LANCZOS)
                        return img
            except:
                continue
        
        return None
    except Exception as e:
        return None

def fetch_image_from_pexels(product_name):
    """Fetch product image from Pexels (free API with key)"""
    try:
        api_key = os.environ.get('PEXELS_API_KEY', '')
        
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
        st.warning(f"Pexels fetch failed: {str(e)}")
        return None

def get_product_image(product_name, img_path):
    """Get product image from multiple sources"""
    
    # Try Pexels first if API key is available (better quality)
    if os.environ.get('PEXELS_API_KEY'):
        img = fetch_image_from_pexels(product_name)
        
        if img:
            img.save(img_path)
            st.success("✅ Image fetched from Pexels")
            return img
    
    # Try Unsplash (no API key needed)
    img = fetch_image_from_unsplash(product_name)
    
    if img:
        img.save(img_path)
        st.success("✅ Image fetched from Unsplash")
        return img
    
    # Generate placeholder if all fail
    st.info("ℹ️ Using placeholder image (couldn't find online)")
    img = generate_placeholder_image(product_name)
    img.save(img_path)
    return img

def get_info_from_groq(product_name):
    """
    Get product information using Groq's free LLM API
    You can get a free API key from: https://console.groq.com/
    """
    try:
        # Check if API key is set
        api_key = os.environ.get('GROQ_API_KEY', '')
        
        if not api_key:
            st.warning("⚠️ GROQ_API_KEY not found in environment variables.")
            st.info("To get accurate real-time data, please set your Groq API key as an environment variable.")
            st.code("export GROQ_API_KEY='your_api_key_here'", language='bash')
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
            "model": "llama-3.3-70b-versatile",  # Free tier model
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
            return product_info
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return None
            
    except json.JSONDecodeError as e:
        st.error(f"Failed to parse API response: {e}")
        st.code(content)
        return None
    except Exception as e:
        st.error(f"Error calling Groq API: {str(e)}")
        return None

def get_info_from_huggingface(product_name):
    """
    Fallback: Get product information using Hugging Face Inference API (free)
    You can get a free API key from: https://huggingface.co/settings/tokens
    """
    try:
        api_key = os.environ.get('HUGGINGFACE_API_KEY', '')
        
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
                return product_info
        
        return None
            
    except Exception as e:
        return None

def get_info_from_ollama(product_name):
    """
    Get information from local Ollama instance (if running)
    Install Ollama from: https://ollama.ai
    Run: ollama pull llama2
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
            return product_info
        
        return None
            
    except Exception as e:
        return None

def get_product_info_with_ai(product_name):
    """Try multiple AI sources to get accurate information"""
    
    # Try Groq first (best quality, free tier available)
    st.info("🤖 Using AI to fetch accurate product information...")
    product_info = get_info_from_groq(product_name)
    
    if product_info:
        st.success("✅ Information retrieved from Groq AI")
        return product_info
    
    # Try Hugging Face as fallback
    st.info("🔄 Trying Hugging Face API...")
    product_info = get_info_from_huggingface(product_name)
    
    if product_info:
        st.success("✅ Information retrieved from Hugging Face")
        return product_info
    
    # Try local Ollama as final fallback
    st.info("🔄 Checking for local Ollama instance...")
    product_info = get_info_from_ollama(product_name)
    
    if product_info:
        st.success("✅ Information retrieved from Ollama")
        return product_info
    
    # If all fail, return None
    return None

def save_to_csv(product_info):
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
        return True
    except Exception as e:
        st.error(f"Error saving to CSV: {str(e)}")
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

def search_product(product_name, language_code):
    """Search for product in database or use AI"""
    try:
        # Load existing data
        df = pd.read_csv('hs_data.csv')
        
        if len(df) > 0:
            # Search in existing data (case-insensitive)
            result = df[df['common_name'].str.lower() == product_name.lower()]
            
            if not result.empty:
                st.info("✅ Found in local database")
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
        st.info("🔍 Product not found in local database. Using AI to fetch information...")
        product_info = get_product_info_with_ai(product_name)
        
        if product_info:
            # Validate the structure
            required_fields = ['common_name', 'hs_code', 'scientific_name', 'family', 
                             'sub_family', 'order', 'genus', 'kingdom', 'description']
            
            # Ensure all fields exist
            for field in required_fields:
                if field not in product_info:
                    product_info[field] = 'Not available'
            
            # Save to CSV for future use
            if save_to_csv(product_info):
                st.success("💾 Product information saved to database!")
            
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
        st.error(f"❌ Error searching product: {str(e)}")
        return None

# Initialize app
initialize_app()

# Header
st.title("🌾 HS Code Finder for Agriculture & Products")
st.markdown("### AI-Powered Search for HS codes, scientific names, and classifications")

# API Setup Instructions in Sidebar
with st.sidebar:
    st.header("⚙️ API Setup")
    
    # Option to enter API keys directly
    st.subheader("🔑 Enter API Keys")
    
    groq_key = st.text_input("Groq API Key:", type="password", value=os.environ.get('GROQ_API_KEY', ''))
    if groq_key:
        os.environ['GROQ_API_KEY'] = groq_key
    
    hf_key = st.text_input("Hugging Face API Key:", type="password", value=os.environ.get('HUGGINGFACE_API_KEY', ''))
    if hf_key:
        os.environ['HUGGINGFACE_API_KEY'] = hf_key
    
    pexels_key = st.text_input("Pexels API Key (Optional - for better images):", type="password", value=os.environ.get('PEXELS_API_KEY', ''))
    if pexels_key:
        os.environ['PEXELS_API_KEY'] = pexels_key
    
    st.markdown("---")
    
    st.markdown("""
    **How to get API keys:**
    
    **Option 1: Groq API (Recommended)**
    1. Visit [console.groq.com](https://console.groq.com/)
    2. Sign up for free
    3. Get your API key
    4. Paste it above
    
    **Option 2: Hugging Face**
    1. Visit [huggingface.co](https://huggingface.co/settings/tokens)
    2. Create free account
    3. Generate token
    4. Paste it above
    
    **Option 3: Pexels (For better images)**
    1. Visit [pexels.com/api](https://www.pexels.com/api/)
    2. Sign up for free
    3. Get API key
    4. Paste it above (optional)
    
    **Option 4: Local Ollama**
    1. Install [Ollama](https://ollama.ai)
    2. Run: `ollama pull llama2`
    3. Start Ollama service
    
    **Note:** Images are automatically fetched from Unsplash (no key needed)
    """)
    
    # Check API status
    st.subheader("📊 API Status")
    if os.environ.get('GROQ_API_KEY'):
        st.success("✅ Groq API configured")
    else:
        st.warning("⚠️ Groq API not configured")
    
    if os.environ.get('HUGGINGFACE_API_KEY'):
        st.success("✅ Hugging Face API configured")
    else:
        st.warning("⚠️ Hugging Face API not configured")
    
    if os.environ.get('PEXELS_API_KEY'):
        st.success("✅ Pexels API configured (Better images)")
    else:
        st.info("ℹ️ Using Unsplash for images (no key needed)")

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
    product_input = st.text_input("🔍 Enter product name:", placeholder="e.g., Apple, Banana, Rice, Coconut Oil, Turmeric...")
with col2:
    selected_language = st.selectbox("🌐 Language:", list(languages.keys()))

search_button = st.button("🔎 Search", use_container_width=True)

# Example products
st.markdown("**💡 Try searching for:** Apple, Banana, Rice, Wheat, Turmeric, Cardamom, Coffee, Tea, Coconut Oil, etc.")

# Search functionality
if search_button and product_input:
    with st.spinner("🔍 Searching for product information..."):
        language_code = languages[selected_language]
        product_info = search_product(product_input, language_code)
        
        if product_info:
            # Display results
            col_img, col_info = st.columns([1, 2])
            
            with col_img:
                # Check for existing image or fetch/generate new one
                img_path = f"images/{product_info['common_name'].lower().replace(' ', '_')}.png"
                
                # Add a refresh button
                col_a, col_b = st.columns([3, 1])
                with col_b:
                    refresh_image = st.button("🔄 Refresh", key="refresh_img")
                
                # Delete image if refresh is clicked
                if refresh_image and os.path.exists(img_path):
                    os.remove(img_path)
                    st.rerun()
                
                if os.path.exists(img_path) and not refresh_image:
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
            st.error("❌ Unable to fetch product information.")
            st.markdown("""
            <div class="warning-box">
                <strong>⚠️ No API configured!</strong><br>
                Please set up at least one API (Groq, Hugging Face, or Ollama) to get accurate product information.
                Check the sidebar for setup instructions.
            </div>
            """, unsafe_allow_html=True)

elif search_button and not product_input:
    st.warning("⚠️ Please enter a product name to search.")

# Footer
st.markdown("---")
st.markdown("💡 **Powered by AI:** This app uses advanced language models to provide accurate, real-time product information")
st.markdown("📊 **Data Source:** AI-generated information with local caching for faster subsequent searches")
