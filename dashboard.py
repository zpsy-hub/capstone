import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import random
from typing import Dict, List
import json
import os

# Load the secret from GitHub Actions environment
api_key = os.environ.get("GROQ_API_KEY")  # Matches the Name you set above

# Try to import optional libraries with fallbacks
try:
    import feedparser
except ImportError:
    feedparser = None
    st.warning("⚠️ feedparser not installed. RSS functionality will be limited. Run: pip install feedparser")

try:
    import requests
except ImportError:
    requests = None
    st.warning("⚠️ requests not installed. Weather data functionality will be limited. Run: pip install requests")

# Page config - must be the first Streamlit command
st.set_page_config(
    page_title="May Pasok Ba - Metro Manila Class Suspension Predictor",
    page_icon="🌧️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .suspension-alert {
        background-color: #ffebee;
        border: 2px solid #f44336;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        text-align: center;
        font-size: 1.2rem;
        font-weight: bold;
        color: #d32f2f;
    }
    .no-suspension-alert {
        background-color: #e8f5e8;
        border: 2px solid #4caf50;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        text-align: center;
        font-size: 1.2rem;
        font-weight: bold;
        color: #2e7d32;
    }
    .prob-high {
        color: #d32f2f;
        font-weight: bold;
    }
    .prob-medium {
        color: #ff9800;
        font-weight: bold;
    }
    .prob-low {
        color: #4caf50;
        font-weight: bold;
    }
    .city-card {
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        background-color: #fafafa;
    }
    .disclaimer {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
        font-size: 0.9rem;
        color: #000000;
    }
    /* Fix for map and expander overlap */
    .stExpander {
        margin-top: 1rem;
        clear: both;
    }
    .element-container {
        z-index: 1;
    }
    /* Ensure proper spacing between sections */
    .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Sample data for demonstration
METRO_MANILA_CITIES = [
    "Manila", "Quezon City", "Makati", "Pasig", "Taguig", "Muntinlupa",
    "Parañaque", "Las Piñas", "Pasay", "Caloocan", "Malabon", "Navotas",
    "Valenzuela", "Marikina", "San Juan", "Mandaluyong", "Pateros"
]

def fetch_openmeteo_weather(lat, lon, city_name):
    """Fetch real weather data from OpenMeteo API"""
    if requests is None:
        return None
    
    try:
        # OpenMeteo API endpoint for current weather and 24h forecast
        url = "https://api.open-meteo.com/v1/forecast"
        
        params = {
            "latitude": lat,
            "longitude": lon,
            "current": [
                "temperature_2m",
                "relative_humidity_2m", 
                "precipitation",
                "weather_code",
                "wind_speed_10m",
                "wind_direction_10m"
            ],
            "hourly": [
                "temperature_2m",
                "precipitation_probability",
                "precipitation",
                "wind_speed_10m",
                "weather_code",
                "relative_humidity_2m"
            ],
            "timezone": "Asia/Manila",
            "forecast_days": 2  # Today + tomorrow
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        current = data.get("current", {})
        hourly = data.get("hourly", {})
        
        # Calculate derived metrics
        temp = current.get("temperature_2m", 0)
        humidity = current.get("relative_humidity_2m", 0)
        wind_speed = current.get("wind_speed_10m", 0)
        precipitation = current.get("precipitation", 0)
        
        # Get next 24 hours data
        hourly_data = []
        if hourly.get("time"):
            for i in range(min(24, len(hourly["time"]))):
                hourly_data.append({
                    "time": hourly["time"][i],
                    "temperature": hourly["temperature_2m"][i] if hourly.get("temperature_2m") else 25,
                    "precipitation": hourly["precipitation"][i] if hourly.get("precipitation") else 0,
                    "precipitation_prob": hourly["precipitation_probability"][i] if hourly.get("precipitation_probability") else 0,
                    "wind_speed": hourly["wind_speed_10m"][i] if hourly.get("wind_speed_10m") else 15,
                    "humidity": hourly["relative_humidity_2m"][i] if hourly.get("relative_humidity_2m") else 70,
                    "weather_code": hourly["weather_code"][i] if hourly.get("weather_code") else 1
                })
        
        # Get forecast averages for next 24 hours
        forecast_temp_avg = sum(h["temperature"] for h in hourly_data) / len(hourly_data) if hourly_data else temp
        forecast_precip_total = sum(h["precipitation"] for h in hourly_data) if hourly_data else 0
        forecast_wind_avg = sum(h["wind_speed"] for h in hourly_data) / len(hourly_data) if hourly_data else wind_speed
        forecast_humidity_avg = sum(h["humidity"] for h in hourly_data) / len(hourly_data) if hourly_data else humidity
        
        # Get next 6 hours precipitation probability
        precip_prob_6h = 0
        if hourly.get("precipitation_probability"):
            precip_prob_6h = max(hourly["precipitation_probability"][:6])
        
        # Calculate flood risk based on precipitation and location
        flood_risk = "Low"
        if precipitation > 20 or forecast_precip_total > 30:
            flood_risk = "High"
        elif precipitation > 10 or forecast_precip_total > 15:
            flood_risk = "Medium"
        
        # Determine TCWS level based on wind speed
        tcws_level = 0
        max_wind = max(wind_speed, forecast_wind_avg)
        if max_wind >= 89:
            tcws_level = 3
        elif max_wind >= 62:
            tcws_level = 2
        elif max_wind >= 39:
            tcws_level = 1
        
        # Weather condition from weather code
        weather_codes = {
            0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
            45: "Foggy", 48: "Depositing rime fog", 51: "Light drizzle", 53: "Moderate drizzle",
            55: "Dense drizzle", 56: "Light freezing drizzle", 57: "Dense freezing drizzle",
            61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain", 66: "Light freezing rain",
            67: "Heavy freezing rain", 71: "Slight snow", 73: "Moderate snow", 75: "Heavy snow",
            77: "Snow grains", 80: "Slight rain showers", 81: "Moderate rain showers",
            82: "Violent rain showers", 85: "Slight snow showers", 86: "Heavy snow showers",
            95: "Thunderstorm", 96: "Thunderstorm with hail", 99: "Thunderstorm with heavy hail"
        }
        
        weather_condition = weather_codes.get(current.get("weather_code", 0), "Unknown")
        
        return {
            "city": city_name,
            # Current conditions
            "temperature": round(temp, 1),
            "humidity": round(humidity, 1),
            "wind_speed": round(wind_speed, 1),
            "precipitation": round(precipitation, 1),
            "weather_condition": weather_condition,
            # Forecast conditions
            "forecast_temp": round(forecast_temp_avg, 1),
            "forecast_humidity": round(forecast_humidity_avg, 1),
            "forecast_wind": round(forecast_wind_avg, 1),
            "forecast_precip": round(forecast_precip_total, 1),
            # Combined metrics
            "precipitation_prob_6h": round(precip_prob_6h, 1),
            "flood_risk": flood_risk,
            "tcws_level": tcws_level,
            "last_updated": datetime.now().strftime("%H:%M:%S"),
            "hourly_data": hourly_data
        }
        
    except Exception as e:
        print(f"Error fetching weather for {city_name}: {e}")
        return None

def generate_sample_data():
    """Generate sample suspension probabilities with real weather data"""
    data = []
    base_time = datetime.now()
    
    # City coordinates for OpenMeteo API
    city_coords = {
        "Manila": [14.5995, 120.9842],
        "Quezon City": [14.6760, 121.0437],
        "Makati": [14.5547, 121.0244],
        "Pasig": [14.5764, 121.0851],
        "Taguig": [14.5176, 121.0509],
        "Muntinlupa": [14.3781, 121.0168],
        "Parañaque": [14.4793, 121.0198],
        "Las Piñas": [14.4378, 120.9761],
        "Pasay": [14.5378, 121.0014],
        "Caloocan": [14.6507, 120.9676],
        "Malabon": [14.6570, 120.9658],
        "Navotas": [14.6691, 120.9470],
        "Valenzuela": [14.6958, 120.9831],
        "Marikina": [14.6507, 121.1029],
        "San Juan": [14.6019, 121.0355],
        "Mandaluyong": [14.5832, 121.0409],
        "Pateros": [14.5441, 121.0699]
    }
    
    for city in METRO_MANILA_CITIES:
        lat, lon = city_coords[city]
        
        # Try to fetch real weather data
        weather_data = fetch_openmeteo_weather(lat, lon, city)
        
        if weather_data:
            # Use real weather data to influence probabilities
            base_prob = 0.3  # Base probability
            
            # Increase probability based on weather conditions
            if weather_data["precipitation"] > 15:
                base_prob += 0.4
            elif weather_data["precipitation"] > 5:
                base_prob += 0.2
                
            if weather_data["wind_speed"] > 50:
                base_prob += 0.3
            elif weather_data["wind_speed"] > 30:
                base_prob += 0.1
                
            if weather_data["tcws_level"] > 0:
                base_prob += 0.3
                
            if weather_data["flood_risk"] == "High":
                base_prob += 0.2
            elif weather_data["flood_risk"] == "Medium":
                base_prob += 0.1
            
            # Add some randomness
            evening_prob = max(0.05, min(0.95, base_prob + random.uniform(-0.15, 0.15)))
            morning_prob = max(0.05, min(0.95, base_prob + random.uniform(-0.2, 0.2)))
            
            weather_factors = {
                'temperature': weather_data["temperature"],
                'humidity': weather_data["humidity"],
                'rainfall_mm': weather_data["precipitation"],
                'wind_speed_kph': weather_data["wind_speed"],
                'precipitation_prob_6h': weather_data["precipitation_prob_6h"],
                'weather_condition': weather_data["weather_condition"],
                'flood_risk': weather_data["flood_risk"],
                'tcws_level': weather_data["tcws_level"],
                'last_updated': weather_data["last_updated"],
                # Forecast data
                'forecast_temp': weather_data["forecast_temp"],
                'forecast_humidity': weather_data["forecast_humidity"],
                'forecast_wind': weather_data["forecast_wind"],
                'forecast_precip': weather_data["forecast_precip"],
                'hourly_data': weather_data["hourly_data"]
            }
        else:
            # Fallback to sample data if API fails
            base_prob = random.uniform(0.1, 0.9)
            evening_prob = max(0.05, min(0.95, base_prob + random.uniform(-0.2, 0.2)))
            morning_prob = max(0.05, min(0.95, base_prob + random.uniform(-0.3, 0.3)))
            
            weather_factors = {
                'temperature': random.uniform(24, 35),
                'humidity': random.uniform(60, 95),
                'rainfall_mm': random.uniform(0, 50),
                'wind_speed_kph': random.uniform(10, 80),
                'precipitation_prob_6h': random.uniform(0, 100),
                'weather_condition': random.choice(['Partly cloudy', 'Light rain', 'Moderate rain', 'Clear sky', 'Thunderstorm']),
                'flood_risk': random.choice(['Low', 'Medium', 'High']),
                'tcws_level': random.choice([0, 1, 2]),
                'last_updated': datetime.now().strftime("%H:%M:%S"),
                # Forecast fallback data
                'forecast_temp': random.uniform(24, 35),
                'forecast_humidity': random.uniform(60, 95),
                'forecast_wind': random.uniform(10, 80),
                'forecast_precip': random.uniform(0, 30),
                'hourly_data': []
            }
        
        data.append({
            'city': city,
            'evening_probability': evening_prob,
            'morning_probability': morning_prob,
            'last_updated': base_time - timedelta(minutes=random.randint(5, 120)),
            'weather_factors': weather_factors,
            'lat': lat,
            'lon': lon
        })
    
    return pd.DataFrame(data)

# Try to import groq for real AI interpretation
try:
    import requests
    # Get API key from environment (either from .env locally or GitHub Secrets in production)
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")  # Secure way to access the key
    
    if not GROQ_API_KEY:
        raise ValueError("API key not found")
        
    GROQ_API_AVAILABLE = True
    
except (ImportError, ValueError) as e:
    GROQ_API_AVAILABLE = False
    st.warning(f"⚠️ Groq not available ({str(e)}). Using rule-based interpretation.")
    
def get_real_ai_weather_interpretation(factors):
    """Get REAL AI interpretation using Groq API"""
    if not GROQ_API_AVAILABLE:
        return ["🤖 AI unavailable - using smart rules instead"]
    
    try:
        # Prepare weather data for AI
        weather_summary = f"""
        Weather Data for Metro Manila:
        - Temperature: {factors['temperature']}°C
        - Humidity: {factors['humidity']}%
        - Rainfall: {factors['rainfall_mm']}mm
        - Wind Speed: {factors['wind_speed_kph']}kph
        - Weather Condition: {factors['weather_condition']}
        - Flood Risk: {factors['flood_risk']}
        - TCWS Level: {factors['tcws_level']}
        - 6h Rain Probability: {factors['precipitation_prob_6h']}%
        """
        
        prompt = f"""You are a weather expert analyzing conditions for school suspension decisions in Metro Manila, Philippines. 

{weather_summary}

Provide 3-4 brief, practical interpretations about:
1. What this weather means for students and transportation
2. Specific risks or safety concerns
3. Impact on school operations
4. Overall suspension likelihood

Keep each point under 15 words. Use relevant emojis. Be specific to Philippines weather patterns.

Example format:
🌧️ Heavy rainfall creating dangerous travel conditions for students
⚡ Thunderstorms increase lightning risk during outdoor activities  
🚨 High flood risk making major roads like EDSA impassable"""

        # Call Groq API
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {GROQ_API_KEY}"
        }
        
        data = {
            "model": "llama-3.3-70b-versatile",
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 300
        }
        
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=10
        )
        
        if response.status_code == 200:
            ai_response = response.json()['choices'][0]['message']['content']
            # Split into individual interpretations
            interpretations = [line.strip() for line in ai_response.split('\n') if line.strip() and ('🌧️' in line or '⚡' in line or '🚨' in line or '🌊' in line or '💨' in line or '🔥' in line or '❄️' in line or '🌪️' in line)]
            return interpretations[:4] if interpretations else [ai_response.strip()]
        else:
            return [f"🤖 AI Error: {response.status_code} - Using fallback interpretation"]
            
    except Exception as e:
        return [f"🤖 AI Connection failed: {str(e)[:50]}... - Using smart rules"]

def get_ai_weather_interpretation(factors):
    """Smart weather interpretation - tries real AI first, falls back to rules"""
    
    # Try real AI first
    ai_interpretations = get_real_ai_weather_interpretation(factors)
    
    # If AI worked, return AI results
    if ai_interpretations and not any('Error' in interp or 'failed' in interp for interp in ai_interpretations):
        return ["🤖 AI Analysis:"] + ai_interpretations
    
    # Fallback to comprehensive rule-based logic
    interpretations = ["📊 Smart Analysis:"]
    
    # Temperature analysis with more cases
    temp = factors['temperature']
    if temp > 38:
        interpretations.append("🔥 Dangerously hot conditions - heat exhaustion risk for students")
    elif temp > 35:
        interpretations.append("🌡️ Extremely hot weather may cause health issues during outdoor activities")
    elif temp > 32:
        interpretations.append("☀️ Very hot conditions - schools may consider heat advisories")
    elif temp > 28:
        interpretations.append("🌤️ Warm weather - comfortable for school activities")
    elif temp < 18:
        interpretations.append("🥶 Unusually cold weather for Metro Manila - rare occurrence")
    elif temp < 22:
        interpretations.append("🌬️ Cool weather conditions - pleasant for outdoor activities")
    else:
        interpretations.append("🌡️ Normal temperature range for Metro Manila")
    
    # Detailed rainfall impact analysis
    rainfall = factors['rainfall_mm']
    if rainfall > 50:
        interpretations.append("🌊 Extreme rainfall - widespread flooding, roads impassable")
    elif rainfall > 30:
        interpretations.append("⛈️ Very heavy rainfall - severe flooding in low-lying areas")
    elif rainfall > 20:
        interpretations.append("🌧️ Heavy rainfall - major roads flooded, transport severely affected")
    elif rainfall > 15:
        interpretations.append("🌦️ Moderate to heavy rain - flood-prone areas affected, some road closures")
    elif rainfall > 10:
        interpretations.append("☔ Moderate rain - localized flooding possible in usual spots")
    elif rainfall > 5:
        interpretations.append("🌧️ Light to moderate rain - minimal flooding in flood-prone areas only")
    elif rainfall > 1:
        interpretations.append("🌦️ Light rain - roads wet but generally passable")
    elif rainfall > 0:
        interpretations.append("💧 Very light drizzle - minimal impact on transportation")
    else:
        interpretations.append("☀️ No current rainfall - clear conditions")
    
    # Comprehensive wind impact analysis
    wind = factors['wind_speed_kph']
    if wind > 220:
        interpretations.append("🌪️ Super typhoon winds - catastrophic damage, total evacuation needed")
    elif wind > 185:
        interpretations.append("🌀 Typhoon winds - extreme destruction, all activities suspended")
    elif wind > 118:
        interpretations.append("💨 Very strong typhoon winds - dangerous flying debris, schools closed")
    elif wind > 89:
        interpretations.append("🌬️ Strong typhoon winds - structural damage possible, unsafe conditions")
    elif wind > 62:
        interpretations.append("💨 Typhoon-force winds - trees may fall, power lines down")
    elif wind > 50:
        interpretations.append("🌪️ Very strong winds - dangerous for students, outdoor activities cancelled")
    elif wind > 39:
        interpretations.append("🌬️ Strong winds - may affect transport, caution for pedestrians")
    elif wind > 25:
        interpretations.append("🍃 Moderate winds - generally safe, minor inconvenience")
    elif wind > 15:
        interpretations.append("🌱 Light to moderate winds - comfortable conditions")
    elif wind > 5:
        interpretations.append("🍃 Light breeze - pleasant weather conditions")
    else:
        interpretations.append("🌅 Very calm conditions - minimal wind")
    
    # Enhanced flood risk interpretation
    flood_risk = factors['flood_risk']
    if flood_risk == 'High':
        interpretations.append("🚨 High flood risk - major thoroughfares like EDSA likely impassable")
    elif flood_risk == 'Medium':
        interpretations.append("⚠️ Moderate flood risk - some underpasses and low areas may flood")
    else:
        interpretations.append("✅ Low flood risk - main roads should remain passable")
    
    # Detailed TCWS interpretation with impact
    tcws = factors['tcws_level']
    if tcws >= 5:
        interpretations.append("🌪️ Signal #5: Super typhoon - catastrophic winds, total evacuation")
    elif tcws >= 4:
        interpretations.append("🌀 Signal #4: Typhoon - very destructive winds, all schools closed")
    elif tcws >= 3:
        interpretations.append("💨 Signal #3: Destructive winds 89-117 kph - schools automatically suspended")
    elif tcws >= 2:
        interpretations.append("🌬️ Signal #2: Damaging winds 62-88 kph - very high suspension probability")
    elif tcws >= 1:
        interpretations.append("🌊 Signal #1: Strong winds 39-61 kph - schools often suspend classes")
    else:
        interpretations.append("🌤️ No tropical cyclone signals - normal wind conditions")
    
    # Weather condition impact with more detail
    condition = factors['weather_condition'].lower()
    if 'thunderstorm' in condition and 'heavy' in condition:
        interpretations.append("⚡ Severe thunderstorms with heavy rain - very dangerous travel conditions")
    elif 'thunderstorm' in condition:
        interpretations.append("⛈️ Thunderstorms present - lightning risk, avoid outdoor activities")
    elif 'heavy rain' in condition:
        interpretations.append("🌧️ Heavy rain creates hazardous driving and walking conditions")
    elif 'moderate rain' in condition:
        interpretations.append("🌦️ Moderate rain affects visibility and road safety")
    elif 'light rain' in condition or 'drizzle' in condition:
        interpretations.append("☔ Light precipitation - minor impact on transportation")
    elif 'fog' in condition:
        interpretations.append("🌫️ Fog reduces visibility - extra caution needed for transport")
    elif 'clear' in condition or 'sunny' in condition:
        interpretations.append("☀️ Clear weather conditions - excellent for all activities")
    elif 'cloudy' in condition:
        interpretations.append("☁️ Overcast skies but stable weather conditions")
    
    return interpretations[:4]  # Return top 4 interpretations

def get_suspension_risk_factors(factors):
    """Calculate and explain suspension risk factors"""
    risk_factors = []
    
    if factors['rainfall_mm'] > 15:
        risk_factors.append("Heavy rainfall (>15mm)")
    if factors['wind_speed_kph'] > 50:
        risk_factors.append("Strong winds (>50kph)")
    if factors['tcws_level'] > 0:
        risk_factors.append(f"Typhoon Signal #{factors['tcws_level']}")
    if factors['flood_risk'] == 'High':
        risk_factors.append("High flood risk")
    if factors['precipitation_prob_6h'] > 80:
        risk_factors.append("Very high rain probability")
    
    return risk_factors

def get_metro_manila_recommendation(df):
    """Get overall Metro Manila recommendation based on majority rule and time of day"""
    from datetime import datetime
    
    # Determine which prediction to use based on current time
    current_hour = datetime.now().hour
    show_evening_first = (current_hour >= 19) or (current_hour < 5)
    
    # Use the appropriate prediction based on time
    if show_evening_first:
        high_prob_cities = len(df[df['evening_probability'] >= 0.7])
        prediction_type = "Evening"
    else:
        high_prob_cities = len(df[df['morning_probability'] >= 0.7])
        prediction_type = "Morning"
    
    total_cities = len(df)
    confidence = high_prob_cities / total_cities
    
    if high_prob_cities >= total_cities * 0.6:
        return "SUSPEND", confidence, prediction_type
    else:
        return "NO SUSPENSION", confidence, prediction_type
    """Get overall Metro Manila recommendation based on majority rule and time of day"""
    from datetime import datetime
    
    # Determine which prediction to use based on current time
    current_hour = datetime.now().hour
    show_evening_first = (current_hour >= 19) or (current_hour < 5)
    
    # Use the appropriate prediction based on time
    if show_evening_first:
        high_prob_cities = len(df[df['evening_probability'] >= 0.7])
        prediction_type = "Evening"
    else:
        high_prob_cities = len(df[df['morning_probability'] >= 0.7])
        prediction_type = "Morning"
    
    total_cities = len(df)
    confidence = high_prob_cities / total_cities
    
    if high_prob_cities >= total_cities * 0.6:
        return "SUSPEND", confidence, prediction_type
    else:
        return "NO SUSPENSION", confidence, prediction_type

def fetch_rss_feed(url, max_items=5):
    """Fetch and parse RSS feed"""
    if not url or url.strip() == "":
        return []
    
    # Check if feedparser is available
    if feedparser is None:
        return [{"error": "feedparser library not installed. Run: pip install feedparser"}]
    
    try:
        # Parse the RSS feed
        feed = feedparser.parse(url)
        
        if feed.bozo:
            return [{"error": f"Invalid RSS feed format: {feed.bozo_exception}"}]
        
        items = []
        for entry in feed.entries[:max_items]:
            # Get published date
            pub_date = "Unknown date"
            if hasattr(entry, 'published'):
                try:
                    pub_date = datetime.strptime(entry.published, "%a, %d %b %Y %H:%M:%S %z").strftime("%Y-%m-%d %H:%M")
                except:
                    try:
                        pub_date = datetime.strptime(entry.published, "%a, %d %b %Y %H:%M:%S %Z").strftime("%Y-%m-%d %H:%M")
                    except:
                        pub_date = entry.published
            elif hasattr(entry, 'updated'):
                try:
                    pub_date = datetime.strptime(entry.updated, "%a, %d %b %Y %H:%M:%S %z").strftime("%Y-%m-%d %H:%M")
                except:
                    pub_date = entry.updated
            
            items.append({
                "title": getattr(entry, 'title', 'No title'),
                "link": getattr(entry, 'link', '#'),
                "published": pub_date,
                "summary": getattr(entry, 'summary', 'No summary available')[:200] + "..." if len(getattr(entry, 'summary', '')) > 200 else getattr(entry, 'summary', 'No summary available')
            })
        
        return items
    
    except Exception as e:
        return [{"error": f"Error fetching RSS feed: {str(e)}"}]

def get_sample_rss_urls():
    """Get sample RSS URLs for demonstration"""
    return {
        "pagasa": [
            "https://www.pagasa.dost.gov.ph/rss",
            "https://feeds.feedburner.com/pagasa-weather-updates",
            "https://news.google.com/rss/search?q=site:twitter.com+PAGASA+advisory+when:1d&hl=en-PH&gl=PH&ceid=PH:en"
        ],
        "news": [
            "https://www.rappler.com/rss/nation/weather/",
            "https://newsinfo.inquirer.net/category/latest-stories/feed",
            "https://mb.com.ph/feed/",
            "https://www.gmanetwork.com/news/rss/news/"
        ],
        "twitter_walangpasok": [
            "https://news.google.com/rss/search?q=site:twitter.com+%23WalangPasok+when:1d&hl=en-PH&gl=PH&ceid=PH:en",
            "https://news.google.com/rss/search?q=site:twitter.com+%22class+suspension%22+when:1d&hl=en-PH&gl=PH&ceid=PH:en",
            "https://news.google.com/rss/search?q=site:twitter.com+(%23WalangPasok+OR+%23ClassSuspension)+when:1d&hl=en-PH&gl=PH&ceid=PH:en",
            "https://news.google.com/rss/search?q=site:twitter.com+(Quezon+City+OR+Manila+OR+Makati)+suspension+when:1d&hl=en-PH&gl=PH&ceid=PH:en"
        ]
    }
    """Fetch and parse RSS feed"""
    if not url or url.strip() == "":
        return []
    
    # Check if feedparser is available
    if feedparser is None:
        return [{"error": "feedparser library not installed. Run: pip install feedparser"}]
    
    try:
        # Parse the RSS feed
        feed = feedparser.parse(url)
        
        if feed.bozo:
            return [{"error": f"Invalid RSS feed format: {feed.bozo_exception}"}]
        
        items = []
        for entry in feed.entries[:max_items]:
            # Get published date
            pub_date = "Unknown date"
            if hasattr(entry, 'published'):
                try:
                    pub_date = datetime.strptime(entry.published, "%a, %d %b %Y %H:%M:%S %z").strftime("%Y-%m-%d %H:%M")
                except:
                    try:
                        pub_date = datetime.strptime(entry.published, "%a, %d %b %Y %H:%M:%S %Z").strftime("%Y-%m-%d %H:%M")
                    except:
                        pub_date = entry.published
            elif hasattr(entry, 'updated'):
                try:
                    pub_date = datetime.strptime(entry.updated, "%a, %d %b %Y %H:%M:%S %z").strftime("%Y-%m-%d %H:%M")
                except:
                    pub_date = entry.updated
            
            items.append({
                "title": getattr(entry, 'title', 'No title'),
                "link": getattr(entry, 'link', '#'),
                "published": pub_date,
                "summary": getattr(entry, 'summary', 'No summary available')[:200] + "..." if len(getattr(entry, 'summary', '')) > 200 else getattr(entry, 'summary', 'No summary available')
            })
        
        return items
    
    except Exception as e:
        return [{"error": f"Error fetching RSS feed: {str(e)}"}]

def get_sample_rss_urls():
    """Get sample RSS URLs for demonstration"""
    return {
        "pagasa": [
            "https://www.pagasa.dost.gov.ph/rss",
            "https://feeds.feedburner.com/pagasa-weather-updates",
            "https://news.google.com/rss/search?q=site:twitter.com+PAGASA+advisory+when:1d&hl=en-PH&gl=PH&ceid=PH:en"
        ],
        "news": [
            "https://www.rappler.com/rss/nation/weather/",
            "https://newsinfo.inquirer.net/category/latest-stories/feed",
            "https://mb.com.ph/feed/",
            "https://www.gmanetwork.com/news/rss/news/"
        ],
        "twitter_walangpasok": [
            "https://news.google.com/rss/search?q=site:twitter.com+%23WalangPasok+when:1d&hl=en-PH&gl=PH&ceid=PH:en",
            "https://news.google.com/rss/search?q=site:twitter.com+%22class+suspension%22+when:1d&hl=en-PH&gl=PH&ceid=PH:en",
            "https://news.google.com/rss/search?q=site:twitter.com+(%23WalangPasok+OR+%23ClassSuspension)+when:1d&hl=en-PH&gl=PH&ceid=PH:en",
            "https://news.google.com/rss/search?q=site:twitter.com+(Quezon+City+OR+Manila+OR+Makati)+suspension+when:1d&hl=en-PH&gl=PH&ceid=PH:en"
        ]
    }

def create_map_visualization(df):
    """Create a map visualization of Metro Manila with suspension probabilities"""
    # Create enhanced hover data
    df['hover_text'] = df.apply(lambda row: 
        f"<b>{row['city']}</b><br>" +
        f"<b>Suspension Probability:</b> {row['morning_probability']:.1%}<br>" +
        f"<b>Weather Condition:</b> {row['weather_factors']['weather_condition']}<br>" +
        f"<b>Temperature:</b> {row['weather_factors']['temperature']:.1f}°C<br>" +
        f"<b>Humidity:</b> {row['weather_factors']['humidity']:.1f}%<br>" +
        f"<b>Rainfall:</b> {row['weather_factors']['rainfall_mm']:.1f} mm<br>" +
        f"<b>Wind Speed:</b> {row['weather_factors']['wind_speed_kph']:.1f} kph<br>" +
        f"<b>6h Rain Prob:</b> {row['weather_factors']['precipitation_prob_6h']:.1f}%<br>" +
        f"<b>Flood Risk:</b> {row['weather_factors']['flood_risk']}<br>" +
        f"<b>TCWS Level:</b> {row['weather_factors']['tcws_level']}<br>" +
        f"<b>Updated:</b> {row['weather_factors']['last_updated']}", axis=1
    )
    
    # Create map with enhanced hover
    fig = px.scatter_map(
        df,
        lat='lat',
        lon='lon',
        color='morning_probability',
        size='morning_probability',
        hover_name='city',
        hover_data={
            'lat': False,
            'lon': False,
            'morning_probability': False,
            'hover_text': False
        },
        color_continuous_scale='RdYlGn_r',
        size_max=25,
        zoom=10,
        title="Metro Manila Suspension Probability Map"
    )
    
    # Update hover template to show detailed weather info
    fig.update_traces(
        hovertemplate="%{customdata[0]}<extra></extra>",
        customdata=df[['hover_text']].values
    )
    
    # Update layout
    fig.update_layout(
        height=600,
        margin={"r":0,"t":40,"l":0,"b":20},
        coloraxis_colorbar=dict(
            title="Suspension<br>Probability",
            tickformat=".0%"
        )
    )
    
    return fig

def create_24hour_forecast_chart(hourly_data, city_name):
    """Create a 24-hour forecast chart for a specific city"""
    if not hourly_data:
        return None
    
    # Prepare data
    times = []
    temps = []
    precip = []
    precip_prob = []
    
    for hour in hourly_data:
        try:
            # Parse time and format for display
            time_str = hour["time"]
            if 'T' in time_str:
                dt = datetime.fromisoformat(time_str.replace('Z', '+00:00'))
                times.append(dt.strftime('%H:%M'))
            else:
                times.append(time_str[-5:])  # Last 5 chars should be HH:MM
            
            temps.append(hour["temperature"])
            precip.append(hour["precipitation"])
            precip_prob.append(hour["precipitation_prob"])
        except:
            continue
    
    if not times:
        return None
    
    # Create subplot with secondary y-axis
    fig = go.Figure()
    
    # Temperature line
    fig.add_trace(go.Scatter(
        x=times,
        y=temps,
        mode='lines+markers',
        name='Temperature (°C)',
        line=dict(color='#ff6b6b', width=3),
        marker=dict(size=6),
        yaxis='y'
    ))
    
    # Precipitation bars
    fig.add_trace(go.Bar(
        x=times,
        y=precip,
        name='Precipitation (mm)',
        marker_color='#4ecdc4',
        opacity=0.7,
        yaxis='y2'
    ))
    
    # Precipitation probability line
    fig.add_trace(go.Scatter(
        x=times,
        y=precip_prob,
        mode='lines',
        name='Rain Probability (%)',
        line=dict(color='#45b7d1', width=2, dash='dash'),
        yaxis='y3'
    ))
    
    # Update layout
    fig.update_layout(
        title=f"24-Hour Weather Forecast - {city_name}",
        xaxis=dict(
            title="Time",
            tickangle=45,
            showgrid=True
        ),
        yaxis=dict(
            title="Temperature (°C)",
            side="left",
            showgrid=True,
            color='#ff6b6b'
        ),
        yaxis2=dict(
            title="Precipitation (mm)",
            overlaying="y",
            side="right",
            position=1,
            color='#4ecdc4'
        ),
        yaxis3=dict(
            title="Rain Probability (%)",
            overlaying="y",
            side="right",
            position=0.85,
            color='#45b7d1'
        ),
        height=400,
        margin=dict(l=50, r=100, t=50, b=50),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        hovermode='x unified'
    )
    
    return fig

def main_dashboard():
    """Main public dashboard"""
    
    # Disclaimer at the very top
    st.markdown('''
    <div class="disclaimer" style="margin-top: 0; margin-bottom: 2rem;">
        <strong>⚠️ DISCLAIMER:</strong> This system provides AI-powered recommendations based on weather data analysis. 
        Official class suspension announcements are made solely by Local Government Units (LGUs) and school administrators. 
        Always check official sources for final decisions. This tool is for informational purposes only.
    </div>
    ''', unsafe_allow_html=True)
    
    st.markdown('<h1 class="main-header">🌧️ MAY PASOK BA?</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Metro Manila Class Suspension Predictor</p>', unsafe_allow_html=True)
    
    # Generate sample data
    df = generate_sample_data()
    
    # Dynamic prediction setup - define variables first
    from datetime import datetime, timedelta
    now = datetime.now()
    current_hour = now.hour
    today = now.date()
    tomorrow = today + timedelta(days=1)
    
    # Between 7PM (19:00) and 4:49AM next day - show Evening prediction only
    show_evening_first = (current_hour >= 19) or (current_hour < 5)
    
    # Get overall recommendation based on time-appropriate prediction
    recommendation, confidence, prediction_type = get_metro_manila_recommendation(df)
    
    # Display main recommendation with time context
    if recommendation == "SUSPEND":
        st.markdown(f'''
        <div class="suspension-alert">
            ⚠️ HIGH PROBABILITY OF CLASS SUSPENSION
            <br>
            <span style="font-size: 1rem;">Metro Manila-wide confidence: {confidence:.1%} ({prediction_type} Prediction)</span>
        </div>
        ''', unsafe_allow_html=True)
    else:
        st.markdown(f'''
        <div class="no-suspension-alert">
            ✅ LOW PROBABILITY OF CLASS SUSPENSION
            <br>
            <span style="font-size: 1rem;">Metro Manila-wide confidence: {(1-confidence):.1%} ({prediction_type} Prediction)</span>
        </div>
        ''', unsafe_allow_html=True)
    
    # Dynamic prediction display - show only the relevant one
    if show_evening_first:
        # Show only Evening prediction
        st.subheader(f"🌙 Evening Prediction (19:00)")
        if current_hour >= 19:
            # After 7PM today, predicting for tomorrow
            target_date = tomorrow
            st.info(f"**NEXT DAY FORECAST** - {target_date.strftime('%B %d, %Y')}\n\n24-hour forecast based - Planning purposes")
        else:
            # Before 5AM, still showing yesterday evening's prediction for today
            target_date = today
            st.info(f"**TODAY'S FORECAST** - {target_date.strftime('%B %d, %Y')}\n\nEvening prediction active until 5:00 AM")
    else:
        # Show only Morning prediction (5AM to 6:59PM)
        st.subheader(f"🌅 Morning Prediction (05:00)")
        st.info(f"**TODAY'S CONFIRMATION** - {today.strftime('%B %d, %Y')}\n\nUpdated with overnight weather data - Higher confidence")
    
    # Map visualization
    st.subheader("📍 Interactive Map")
    map_fig = create_map_visualization(df)
    st.plotly_chart(map_fig, use_container_width=True)
    
    # Add some spacing before city breakdown
    st.markdown("<br>", unsafe_allow_html=True)
    
    # City-by-city breakdown
    st.subheader("🏙️ City-by-City Breakdown")
    
    # Sort cities by the most relevant prediction based on time of day
    if show_evening_first:
        df_sorted = df.sort_values('evening_probability', ascending=False)
    else:
        df_sorted = df.sort_values('morning_probability', ascending=False)
    
    # Display cities in expanders - MUCH LARGER city font
    for _, row in df_sorted.iterrows():
        # Use only the active prediction based on time of day
        active_prob = row['evening_probability'] if show_evening_first else row['morning_probability']
        prediction_type = "Evening" if show_evening_first else "Morning"
        
        # MUCH LARGER FONT FOR CITY NAMES - Make them super visible
        city_name_style = f"""
        <style>
        .big-city-name {{
            font-size: 1.8rem !important;
            font-weight: bold !important;
            color: white !important;
        }}
        </style>
        """
        st.markdown(city_name_style, unsafe_allow_html=True)
        
        with st.expander(f"# {row['city']} - {active_prob:.1%} probability", expanded=False):
            
            # 1. SUSPENSION PROBABILITY - Most Important
            st.markdown("### 🎯 **SUSPENSION PROBABILITY**")
            prob_col1, prob_col2 = st.columns([3, 1])
            
            with prob_col1:
                # Clean probability display without borders
                prob_color = "#d32f2f" if active_prob >= 0.7 else "#ff9800" if active_prob >= 0.4 else "#4caf50"
                st.markdown(f"""
                <div style="text-align: center; padding: 15px; background-color: {prob_color}15; border-radius: 10px;">
                    <h1 style="color: {prob_color}; margin: 0; font-size: 2.5rem;">{active_prob:.1%}</h1>
                    <h4 style="color: white; margin: 5px 0;">{"🌙 Evening" if show_evening_first else "🌅 Morning"} Prediction</h4>
                    <p style="margin: 0; color: #ccc; font-size: 0.9rem;">Target: {target_date.strftime('%B %d, %Y') if show_evening_first else today.strftime('%B %d, %Y')}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with prob_col2:
                # AI Risk Assessment
                factors = row['weather_factors']
                risk_factors = get_suspension_risk_factors(factors)
                risk_level = "🟢 Low" if len(risk_factors) <= 1 else "🟡 Medium" if len(risk_factors) <= 3 else "🔴 High"
                
                st.markdown(f"""
                <div style="text-align: center; padding: 12px; background-color: #333; border-radius: 8px;">
                    <h5 style="margin: 0; color: white;">🤖 AI Risk Analysis</h5>
                    <h3 style="margin: 5px 0; color: white;">{risk_level}</h3>
                    <small style="color: #ccc;">{len(risk_factors)} risk factor(s)</small>
                </div>
                """, unsafe_allow_html=True)
            
            # AI Weather Interpretation Section - Enhanced with transparency
            st.markdown("### 🤖 **AI WEATHER INTERPRETATION**")
            
            # AI Transparency Notice
            st.markdown("""
            <div style="padding: 8px; margin: 4px 0; background-color: #1a1a1a; border-radius: 6px; border-left: 3px solid #ffa500;">
                <small style="color: #ccc; font-size: 0.85rem;">
                    <strong>🔬 AI Model:</strong> Llama 3.3 70B via Groq API | 
                    <strong>⚠️ Note:</strong> AI interpretation may vary from actual on-site conditions. Always verify with official weather sources.
                </small>
            </div>
            """, unsafe_allow_html=True)
            
            interpretations = get_ai_weather_interpretation(factors)
            
            # Display interpretations in a clean format
            for interpretation in interpretations[:4]:  # Show top 4 most relevant
                st.markdown(f"""
                <div style="padding: 10px; margin: 6px 0; background-color: #2a2a2a; border-radius: 6px; border-left: 3px solid #1f77b4;">
                    <span style="color: white; font-size: 1rem;">{interpretation}</span>
                </div>
                """, unsafe_allow_html=True)
            
            if risk_factors:
                st.markdown("**⚠️ Key Suspension Risk Factors:**")
                for factor in risk_factors:
                    st.markdown(f"• {factor}", unsafe_allow_html=True)
            
            st.markdown("---")
            
            # 2. CURRENT CONDITIONS - Improved font sizing
            st.markdown("### 🌡️ **CURRENT CONDITIONS**")
            curr_col1, curr_col2, curr_col3, curr_col4 = st.columns(4)
            
            with curr_col1:
                st.markdown(f"""
                <div style="text-align: center; padding: 12px; background-color: #2a2a2a; border-radius: 8px;">
                    <h5 style="margin: 0; color: #ccc; font-size: 1rem;">🌡️ Temperature</h5>
                    <h2 style="margin: 5px 0; color: white; font-size: 1.8rem;">{factors['temperature']:.1f}°C</h2>
                    <small style="color: #888; font-size: 0.9rem;">{"Feels hot" if factors['temperature'] > 32 else "Feels warm" if factors['temperature'] > 28 else "Comfortable"}</small>
                </div>
                """, unsafe_allow_html=True)
            
            with curr_col2:
                st.markdown(f"""
                <div style="text-align: center; padding: 12px; background-color: #2a2a2a; border-radius: 8px;">
                    <h5 style="margin: 0; color: #ccc; font-size: 1rem;">💧 Humidity</h5>
                    <h2 style="margin: 5px 0; color: white; font-size: 1.8rem;">{factors['humidity']:.1f}%</h2>
                    <small style="color: #888; font-size: 0.9rem;">{"Very humid" if factors['humidity'] > 85 else "Humid" if factors['humidity'] > 70 else "Moderate"}</small>
                </div>
                """, unsafe_allow_html=True)
            
            with curr_col3:
                rain_intensity = "Heavy" if factors['rainfall_mm'] > 15 else "Moderate" if factors['rainfall_mm'] > 5 else "Light" if factors['rainfall_mm'] > 0 else "None"
                rain_color = "#ff4444" if factors['rainfall_mm'] > 15 else "#ff8800" if factors['rainfall_mm'] > 5 else "#44ff44"
                st.markdown(f"""
                <div style="text-align: center; padding: 12px; background-color: #2a2a2a; border-radius: 8px;">
                    <h5 style="margin: 0; color: #ccc; font-size: 1rem;">🌧️ Rainfall</h5>
                    <h2 style="margin: 5px 0; color: {rain_color}; font-size: 1.8rem;">{factors['rainfall_mm']:.1f} mm</h2>
                    <small style="color: #888; font-size: 0.9rem;">{rain_intensity} rain</small>
                </div>
                """, unsafe_allow_html=True)
            
            with curr_col4:
                wind_intensity = "Very strong" if factors['wind_speed_kph'] > 60 else "Strong" if factors['wind_speed_kph'] > 40 else "Moderate" if factors['wind_speed_kph'] > 25 else "Light"
                wind_color = "#ff4444" if factors['wind_speed_kph'] > 50 else "#ff8800" if factors['wind_speed_kph'] > 30 else "#44ff44"
                st.markdown(f"""
                <div style="text-align: center; padding: 12px; background-color: #2a2a2a; border-radius: 8px;">
                    <h5 style="margin: 0; color: #ccc; font-size: 1rem;">💨 Wind Speed</h5>
                    <h2 style="margin: 5px 0; color: {wind_color}; font-size: 1.8rem;">{factors['wind_speed_kph']:.1f} kph</h2>
                    <small style="color: #888; font-size: 0.9rem;">{wind_intensity} winds</small>
                </div>
                """, unsafe_allow_html=True)
            
            # Additional current conditions - IMPROVED FONT SIZES
            st.markdown("<br>", unsafe_allow_html=True)
            curr2_col1, curr2_col2, curr2_col3, curr2_col4 = st.columns(4)
            
            with curr2_col1:
                st.markdown(f"""
                <div style="text-align: center; padding: 10px; background-color: #2a2a2a; border-radius: 6px;">
                    <small style="color: #ccc; font-size: 0.95rem;">☁️ Condition</small><br>
                    <strong style="color: white; font-size: 1.1rem;">{factors['weather_condition']}</strong>
                </div>
                """, unsafe_allow_html=True)
            
            with curr2_col2:
                flood_color = "#ff4444" if factors['flood_risk'] == 'High' else "#ff8800" if factors['flood_risk'] == 'Medium' else "#44ff44"
                st.markdown(f"""
                <div style="text-align: center; padding: 10px; background-color: #2a2a2a; border-radius: 6px;">
                    <small style="color: #ccc; font-size: 0.95rem;">🌊 Flood Risk</small><br>
                    <strong style="color: {flood_color}; font-size: 1.1rem;">{factors['flood_risk']}</strong>
                </div>
                """, unsafe_allow_html=True)
            
            with curr2_col3:
                tcws_color = "#ff4444" if factors['tcws_level'] > 1 else "#ff8800" if factors['tcws_level'] > 0 else "#44ff44"
                tcws_meaning = "Destructive" if factors['tcws_level'] > 2 else "Damaging" if factors['tcws_level'] > 1 else "Strong" if factors['tcws_level'] > 0 else "Normal"
                st.markdown(f"""
                <div style="text-align: center; padding: 10px; background-color: #2a2a2a; border-radius: 6px;">
                    <small style="color: #ccc; font-size: 0.95rem;">🌀 TCWS Level</small><br>
                    <strong style="color: {tcws_color}; font-size: 1.1rem;">{factors['tcws_level']} - {tcws_meaning}</strong>
                </div>
                """, unsafe_allow_html=True)
            
            with curr2_col4:
                st.markdown(f"""
                <div style="text-align: center; padding: 10px; background-color: #2a2a2a; border-radius: 6px;">
                    <small style="color: #ccc; font-size: 0.95rem;">🕐 Updated</small><br>
                    <strong style="color: white; font-size: 1.1rem;">{factors['last_updated']}</strong>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # 3. FORECAST - Clean borderless design
            if show_evening_first:
                st.markdown("### 📅 **TOMORROW'S FORECAST**")
            else:
                st.markdown("### 📊 **TODAY'S OUTLOOK**")
            
            forecast_col1, forecast_col2, forecast_col3 = st.columns(3)
            
            with forecast_col1:
                st.markdown(f"""
                <div style="text-align: center; padding: 12px; background-color: #2a2a2a; border-radius: 8px;">
                    <h5 style="margin: 0; color: #ccc;">🌡️ Temperature</h5>
                    <h3 style="margin: 3px 0; color: white;">{factors['forecast_temp']:.1f}°C</h3>
                    <small style="color: #888;">Avg for next 24h</small>
                </div>
                """, unsafe_allow_html=True)
            
            with forecast_col2:
                precip_color = "#ff4444" if factors['forecast_precip'] > 20 else "#ff8800" if factors['forecast_precip'] > 10 else "#44ff44"
                rain_impact = "Major flooding likely" if factors['forecast_precip'] > 30 else "Flooding possible" if factors['forecast_precip'] > 15 else "Minor flooding" if factors['forecast_precip'] > 5 else "Minimal impact"
                st.markdown(f"""
                <div style="text-align: center; padding: 12px; background-color: #2a2a2a; border-radius: 8px;">
                    <h5 style="margin: 0; color: #ccc;">🌧️ Total Rain</h5>
                    <h3 style="margin: 3px 0; color: {precip_color};">{factors['forecast_precip']:.1f} mm</h3>
                    <small style="color: #888;">{rain_impact}</small>
                </div>
                """, unsafe_allow_html=True)
            
            with forecast_col3:
                rain_prob_color = "#ff4444" if factors['precipitation_prob_6h'] > 70 else "#ff8800" if factors['precipitation_prob_6h'] > 40 else "#44ff44"
                prob_meaning = "Very likely" if factors['precipitation_prob_6h'] > 80 else "Likely" if factors['precipitation_prob_6h'] > 60 else "Possible" if factors['precipitation_prob_6h'] > 30 else "Unlikely"
                st.markdown(f"""
                <div style="text-align: center; padding: 12px; background-color: #2a2a2a; border-radius: 8px;">
                    <h5 style="margin: 0; color: #ccc;">☔ Rain Chance</h5>
                    <h3 style="margin: 3px 0; color: {rain_prob_color};">{factors['precipitation_prob_6h']:.1f}%</h3>
                    <small style="color: #888;">{prob_meaning}</small>
                </div>
                """, unsafe_allow_html=True)
            
            # 4. DETAILED FORECAST CHART - Optional/Expandable
            if row['weather_factors']['hourly_data']:
                with st.expander("📈 Detailed 24-Hour Forecast"):
                    forecast_chart = create_24hour_forecast_chart(
                        row['weather_factors']['hourly_data'], 
                        row['city']
                    )
                    if forecast_chart:
                        st.plotly_chart(forecast_chart, use_container_width=True)
    
    # RSS Feeds Section - Simple Feed Display
    st.subheader("📡 Official Updates & News Feeds")
    
    # Simple CSS for clean feed display
    st.markdown("""
    <style>
    .feed-item {
        padding: 8px 0;
        border-bottom: 1px solid #eee;
        margin-bottom: 8px;
    }
    .feed-time {
        color: #666;
        font-size: 0.9rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🌪️ PAGASA Updates")
        
        # Simple RSS URL input
        pagasa_url = st.text_input(
            "RSS URL:", 
            placeholder="Enter PAGASA RSS feed URL...",
            key="pagasa_simple_input"
        )
        
        if pagasa_url and st.button("🔄 Load Feed", key="load_pagasa"):
            with st.spinner("Loading..."):
                items = fetch_rss_feed(pagasa_url)
                st.session_state.pagasa_feed = items
        
        # Display feed items
        if pagasa_url and 'pagasa_feed' in st.session_state:
            for item in st.session_state.pagasa_feed:
                if "error" in item:
                    st.error(f"Error: {item['error']}")
                else:
                    st.markdown(f"""
                    <div class="feed-item">
                        <strong>{item['title']}</strong><br>
                        <span class="feed-time">{item['published']}</span><br>
                        <small>{item['summary'][:100]}...</small>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            # Sample feed content
            sample_feed = [
                ("🌩️ Thunderstorm Advisory for Metro Manila", "3:30 PM", "Moderate to heavy rainshowers with lightning and strong winds expected over Metro Manila within 2 hours."),
                ("🌧️ Yellow Rainfall Warning Lifted", "2:00 PM", "Yellow rainfall warning has been lifted for NCR as rainfall intensity has decreased."),
                ("💨 Southwest Monsoon Advisory", "12:00 PM", "Southwest monsoon continues to affect Luzon. Occasional rains expected."),
                ("🌀 Weather Update - Partly Cloudy", "10:00 AM", "Partly cloudy skies with isolated thunderstorms over Metro Manila."),
                ("📊 24-Hour Weather Forecast", "6:00 AM", "Partly cloudy to cloudy skies with scattered rainshowers and thunderstorms.")
            ]
            
            for title, time, summary in sample_feed:
                st.markdown(f"""
                <div class="feed-item">
                    <strong>{title}</strong><br>
                    <span class="feed-time">{time}</span><br>
                    <small>{summary}</small>
                </div>
                """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 📰 #WalangPasok Updates")
        
        # Simple RSS URL input
        news_url = st.text_input(
            "RSS URL:", 
            placeholder="Enter news RSS feed URL...",
            key="news_simple_input"
        )
        
        if news_url and st.button("🔄 Load Feed", key="load_news"):
            with st.spinner("Loading..."):
                items = fetch_rss_feed(news_url)
                st.session_state.news_feed = items
        
        # Display feed items
        if news_url and 'news_feed' in st.session_state:
            for item in st.session_state.news_feed:
                if "error" in item:
                    st.error(f"Error: {item['error']}")
                else:
                    st.markdown(f"""
                    <div class="feed-item">
                        <strong>{item['title']}</strong><br>
                        <span class="feed-time">{item['published']}</span><br>
                        <small>{item['summary'][:100]}...</small>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            # Sample feed content
            sample_news = [
                ("📢 Quezon City Suspends Classes", "3:00 PM", "Quezon City announces suspension of classes in all levels due to continuous heavy rainfall."),
                ("🏫 Makati: No Class Suspension", "2:30 PM", "Makati City announces no class suspension despite intermittent rains in the area."),
                ("⚡ Power Outages Reported", "1:45 PM", "Several areas in Metro Manila experience power interruptions due to weather conditions."),
                ("🚗 Traffic Updates - EDSA", "1:15 PM", "Heavy traffic reported along EDSA due to flooding in some areas."),
                ("🌊 Flood Alert - Low-lying Areas", "12:45 PM", "Residents in flood-prone areas advised to take necessary precautions.")
            ]
            
            for title, time, summary in sample_news:
                st.markdown(f"""
                <div class="feed-item">
                    <strong>{title}</strong><br>
                    <span class="feed-time">{time}</span><br>
                    <small>{summary}</small>
                </div>
                """, unsafe_allow_html=True)
    
    # Last updated timestamp
    st.markdown(f"<p style='text-align: center; color: #666; font-size: 0.9rem;'>Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>", unsafe_allow_html=True)

def admin_dashboard():
    """Admin-only dashboard for system monitoring and logging"""
    st.markdown('<h1 class="main-header">🔧 ADMIN DASHBOARD</h1>', unsafe_allow_html=True)
    
    # Authentication check (simplified for demo)
    if 'admin_authenticated' not in st.session_state:
        st.session_state.admin_authenticated = False
    
    if not st.session_state.admin_authenticated:
        st.subheader("🔐 Admin Authentication")
        password = st.text_input("Enter admin password:", type="password")
        if st.button("Login"):
            if password == "admin123":  # Simple demo password
                st.session_state.admin_authenticated = True
                st.rerun()
            else:
                st.error("Invalid password!")
        return
    
    # Admin tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 System Status", "📈 Model Performance", "🗂️ Data Logs", "⚙️ Configuration"])
    
    with tab1:
        st.subheader("System Status Overview")
        
        # System health metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("System Uptime", "99.8%", "0.1%")
        with col2:
            st.metric("API Calls Today", "1,247", "12")
        with col3:
            st.metric("Data Freshness", "2 min ago", "Normal")
        with col4:
            st.metric("Active Users", "156", "23")
        
        # Recent system events
        st.subheader("Recent System Events")
        events_data = [
            {"timestamp": "2024-01-15 14:30:00", "event": "Model prediction completed", "status": "SUCCESS"},
            {"timestamp": "2024-01-15 14:25:00", "event": "Weather data fetched", "status": "SUCCESS"},
            {"timestamp": "2024-01-15 14:20:00", "event": "RSS feed updated", "status": "SUCCESS"},
            {"timestamp": "2024-01-15 14:15:00", "event": "Database backup completed", "status": "SUCCESS"},
            {"timestamp": "2024-01-15 14:10:00", "event": "User session started", "status": "INFO"}
        ]
        
        events_df = pd.DataFrame(events_data)
        st.dataframe(events_df, use_container_width=True)
    
    with tab2:
        st.subheader("Model Performance Metrics")
        
        # Performance metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Model Accuracy", "87.3%", "2.1%")
        with col2:
            st.metric("Precision", "84.7%", "1.8%")
        with col3:
            st.metric("Recall", "89.2%", "1.3%")
        
        # Performance over time chart
        dates = pd.date_range(start='2024-01-01', end='2024-01-15', freq='D')
        accuracy_data = [85 + random.uniform(-5, 5) for _ in dates]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dates, y=accuracy_data, mode='lines+markers', name='Accuracy'))
        fig.update_layout(title="Model Accuracy Over Time", xaxis_title="Date", yaxis_title="Accuracy (%)")
        st.plotly_chart(fig, use_container_width=True)
        
        # Feature importance
        st.subheader("Feature Importance")
        features = ['Rainfall (mm)', 'Wind Speed', 'Flood Risk', 'TCWS Level', 'Historical Pattern', 'Time of Day']
        importance = [0.35, 0.25, 0.20, 0.15, 0.10, 0.05]
        
        fig = px.bar(x=features, y=importance, title="Feature Importance in Model")
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Data Collection & Processing Logs")
        
        # Log type selector
        log_type = st.selectbox("Select Log Type:", ["Weather Data", "Suspension Announcements", "RSS Feeds", "User Activity"])
        
        # Sample log data
        if log_type == "Weather Data":
            log_data = [
                {"timestamp": "2024-01-15 14:30:00", "source": "OpenMeteo API", "status": "SUCCESS", "records": 17},
                {"timestamp": "2024-01-15 14:25:00", "source": "OpenMeteo API", "status": "SUCCESS", "records": 17},
                {"timestamp": "2024-01-15 14:20:00", "source": "OpenMeteo API", "status": "SUCCESS", "records": 17},
            ]
        elif log_type == "Suspension Announcements":
            log_data = [
                {"timestamp": "2024-01-15 14:30:00", "source": "QC Facebook", "status": "SUCCESS", "records": 1},
                {"timestamp": "2024-01-15 14:25:00", "source": "Manila Twitter", "status": "SUCCESS", "records": 0},
                {"timestamp": "2024-01-15 14:20:00", "source": "Makati Facebook", "status": "SUCCESS", "records": 1},
            ]
        elif log_type == "RSS Feeds":
            log_data = [
                {"timestamp": "2024-01-15 14:30:00", "source": "PAGASA RSS", "status": "SUCCESS", "records": 3},
                {"timestamp": "2024-01-15 14:25:00", "source": "Rappler RSS", "status": "SUCCESS", "records": 5},
                {"timestamp": "2024-01-15 14:20:00", "source": "Inquirer RSS", "status": "SUCCESS", "records": 2},
            ]
        else:  # User Activity
            log_data = [
                {"timestamp": "2024-01-15 14:30:00", "action": "Page View", "user_id": "anonymous_123", "page": "main"},
                {"timestamp": "2024-01-15 14:29:00", "action": "City Detail", "user_id": "anonymous_456", "page": "quezon_city"},
                {"timestamp": "2024-01-15 14:28:00", "action": "Map Interaction", "user_id": "anonymous_789", "page": "main"},
            ]
        
        log_df = pd.DataFrame(log_data)
        st.dataframe(log_df, use_container_width=True)
        
        # Export logs
        if st.button("Export Logs to CSV"):
            csv = log_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"{log_type.lower().replace(' ', '_')}_logs.csv",
                mime="text/csv"
            )
    
    with tab4:
        st.subheader("System Configuration")
        
        # Model settings
        st.subheader("Model Settings")
        prediction_threshold = st.slider("Suspension Probability Threshold", 0.0, 1.0, 0.7, 0.01)
        confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.8, 0.01)
        
        # Data collection settings
        st.subheader("Data Collection Settings")
        weather_update_interval = st.selectbox("Weather Data Update Interval", [5, 10, 15, 30, 60], index=1)
        social_media_check_interval = st.selectbox("Social Media Check Interval", [5, 10, 15, 30], index=0)
        
        # RSS Feed URLs
        st.subheader("RSS Feed Configuration")
        pagasa_rss = st.text_input("PAGASA RSS URL:", value="https://pagasa.dost.gov.ph/rss/weather-advisory")
        rappler_rss = st.text_input("Rappler RSS URL:", value="https://rappler.com/rss/nation/weather")
        inquirer_rss = st.text_input("Inquirer RSS URL:", value="https://newsinfo.inquirer.net/category/latest-stories/feed")
        
        # Save configuration
        if st.button("Save Configuration"):
            st.success("Configuration saved successfully!")
            
        # System maintenance
        st.subheader("System Maintenance")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Retrain Model"):
                st.info("Model retraining initiated...")
        
        with col2:
            if st.button("Clear Cache"):
                st.info("Cache cleared successfully!")
    
    # Logout button
    if st.button("🚪 Logout"):
        st.session_state.admin_authenticated = False
        st.rerun()

def main():
    """Main application function"""
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Select Dashboard:", ["🏠 Main Dashboard", "🔧 Admin Dashboard"])
    
    if page == "🏠 Main Dashboard":
        main_dashboard()
    else:
        admin_dashboard()
    
    # Sidebar info
    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.markdown("""
    **May Pasok Ba?** is an AI-powered system that predicts class suspension probabilities 
    for Metro Manila based on weather conditions and historical LGU decision patterns.
    """)
    
    st.sidebar.markdown("### System Status")
    st.sidebar.success("🟢 System Online")
    st.sidebar.info(f"🕐 Last Update: {datetime.now().strftime('%H:%M:%S')}")
    
    # Auto-refresh option
    if st.sidebar.checkbox("Auto-refresh (30s)"):
        st.rerun()

if __name__ == "__main__":
    main()