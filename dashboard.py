import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import requests
import json
import time
import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# Page config
st.set_page_config(
    page_title="May Pasok Ba - Metro Manila Class Suspension Predictor",
    page_icon="🌧️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern design
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    .main {
        font-family: 'Inter', sans-serif;
    }
    
    .sidebar .sidebar-content {
        background-color: #ffffff;
        border-right: 1px solid #e5e7eb;
    }
    
    .main-header {
        font-size: 2.5rem;
        font-weight: 800;
        color: #1f2937;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        font-size: 1.2rem;
        color: #6b7280;
        margin-bottom: 2rem;
    }
    
    .lgu-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 0.5rem 0;
        color: white;
        cursor: pointer;
        transition: transform 0.2s, box-shadow 0.2s;
        border: none;
    }
    
    .lgu-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    
    .recommendation-card {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        border-radius: 12px;
        padding: 2rem;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .recommendation-card.safe {
        background: linear-gradient(135deg, #26de81 0%, #20bf6b 100%);
    }
    
    .stat-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        border: 1px solid #e5e7eb;
        margin-bottom: 1rem;
    }
    
    .probability-display {
        font-size: 4rem;
        font-weight: 800;
        line-height: 1;
    }
    
    .prob-high { color: #dc2626; }
    .prob-medium { color: #f59e0b; }
    .prob-low { color: #10b981; }
    
    .ai-interpretation {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px;
        padding: 1.5rem;
        color: white;
        margin: 1rem 0;
    }
    
    .interpretation-item {
        background: rgba(255,255,255,0.15);
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid rgba(255,255,255,0.4);
        color: white;
        font-size: 0.95rem;
        line-height: 1.5;
    }
    
    .weather-metric {
        text-align: center;
        padding: 1rem;
        background: #f9fafb;
        border-radius: 8px;
        margin: 0.5rem;
    }
    
    .weather-metric h3 {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0.5rem 0;
    }
    
    .disclaimer {
        background: #fef3c7;
        border: 1px solid #f59e0b;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        color: #92400e;
    }
</style>
""", unsafe_allow_html=True)

# Metro Manila LGUs with coordinates
METRO_MANILA_LGUS = {
    "Manila": {"lat": 14.5995, "lon": 120.9842},
    "Quezon City": {"lat": 14.6760, "lon": 121.0437},
    "Makati": {"lat": 14.5547, "lon": 121.0244},
    "Pasig": {"lat": 14.5764, "lon": 121.0851},
    "Taguig": {"lat": 14.5176, "lon": 121.0509},
    "Muntinlupa": {"lat": 14.3781, "lon": 121.0168},
    "Parañaque": {"lat": 14.4793, "lon": 121.0198},
    "Las Piñas": {"lat": 14.4378, "lon": 120.9761},
    "Pasay": {"lat": 14.5378, "lon": 121.0014},
    "Caloocan": {"lat": 14.6507, "lon": 120.9676},
    "Malabon": {"lat": 14.6570, "lon": 120.9658},
    "Navotas": {"lat": 14.6691, "lon": 120.9470},
    "Valenzuela": {"lat": 14.6958, "lon": 120.9831},
    "Marikina": {"lat": 14.6507, "lon": 121.1029},
    "San Juan": {"lat": 14.6019, "lon": 121.0355},
    "Mandaluyong": {"lat": 14.5832, "lon": 121.0409},
    "Pateros": {"lat": 14.5441, "lon": 121.0699}
}

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_weather_data(lat, lon, city_name):
    """Fetch weather data from OpenMeteo API"""
    try:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat,
            "longitude": lon,
            "current": [
                "temperature_2m", "relative_humidity_2m", "precipitation",
                "weather_code", "wind_speed_10m", "wind_direction_10m"
            ],
            "hourly": [
                "temperature_2m", "precipitation_probability", "precipitation",
                "wind_speed_10m", "weather_code", "relative_humidity_2m"
            ],
            "timezone": "Asia/Manila",
            "forecast_days": 2
        }
        
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        
        current = data.get("current", {})
        hourly = data.get("hourly", {})
        
        # Calculate 24-hour rainfall forecast
        rainfall_24h = []
        if hourly.get("precipitation"):
            rainfall_24h = hourly["precipitation"][:24]
        else:
            rainfall_24h = [0] * 24  # Fallback
        
        # Weather interpretation
        weather_code = current.get("weather_code", 0)
        weather_conditions = {
            0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
            45: "Foggy", 48: "Depositing rime fog", 51: "Light drizzle", 53: "Moderate drizzle",
            55: "Dense drizzle", 61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
            71: "Slight snow", 73: "Moderate snow", 75: "Heavy snow",
            80: "Slight rain showers", 81: "Moderate rain showers", 82: "Violent rain showers",
            95: "Thunderstorm", 96: "Thunderstorm with hail", 99: "Thunderstorm with heavy hail"
        }
        
        return {
            "city": city_name,
            "temperature": float(current.get("temperature_2m", 28)),
            "humidity": float(current.get("relative_humidity_2m", 70)),
            "wind_speed": float(current.get("wind_speed_10m", 15)),
            "precipitation": float(current.get("precipitation", 0)),
            "weather_condition": weather_conditions.get(weather_code, "Unknown"),
            "weather_code": weather_code,
            "rainfall_24h": rainfall_24h,
            "hourly_data": hourly,
            "last_updated": datetime.now().strftime("%H:%M:%S"),
            "data_source": "OpenMeteo API"
        }
        
    except requests.exceptions.RequestException as e:
        st.warning(f"Network error fetching weather for {city_name}: {str(e)[:50]}...")
        return None
    except Exception as e:
        st.warning(f"Error processing weather data for {city_name}: {str(e)[:50]}...")
        return None

def calculate_suspension_probability(weather_data):
    """Calculate suspension probability based on weather conditions"""
    if not weather_data or not isinstance(weather_data, dict):
        return 30  # Default probability for invalid data
    
    try:
        base_prob = 20
        
        # Heavy rainfall
        precipitation = float(weather_data.get("precipitation", 0))
        if precipitation > 20:
            base_prob += 40
        elif precipitation > 10:
            base_prob += 25
        elif precipitation > 5:
            base_prob += 15
        
        # High winds
        wind_speed = float(weather_data.get("wind_speed", 0))
        if wind_speed > 60:
            base_prob += 35
        elif wind_speed > 40:
            base_prob += 20
        elif wind_speed > 25:
            base_prob += 10
        
        # Dangerous weather conditions
        dangerous_codes = [82, 95, 96, 99]  # Violent rain, thunderstorms
        weather_code = weather_data.get("weather_code", 0)
        if weather_code in dangerous_codes:
            base_prob += 30
        
        # 24-hour rainfall forecast
        rainfall_24h = weather_data.get("rainfall_24h", [])
        if rainfall_24h and len(rainfall_24h) > 0:
            try:
                max_rainfall = max(float(r) for r in rainfall_24h if r is not None)
                if max_rainfall > 15:
                    base_prob += 20
                elif max_rainfall > 10:
                    base_prob += 10
            except (ValueError, TypeError):
                pass  # Skip if rainfall data is invalid
        
        return min(95, max(5, int(base_prob)))
        
    except Exception as e:
        st.warning(f"Error calculating probability: {str(e)[:50]}...")
        return 30  # Default fallback

def calculate_suspension_probability(weather_data):
    """Calculate suspension probability based on weather conditions"""
    if not weather_data:
        return 30
    
    base_prob = 20
    
    # Heavy rainfall
    if weather_data["precipitation"] > 20:
        base_prob += 40
    elif weather_data["precipitation"] > 10:
        base_prob += 25
    elif weather_data["precipitation"] > 5:
        base_prob += 15
    
    # High winds
    if weather_data["wind_speed"] > 60:
        base_prob += 35
    elif weather_data["wind_speed"] > 40:
        base_prob += 20
    elif weather_data["wind_speed"] > 25:
        base_prob += 10
    
    # Dangerous weather conditions
    dangerous_codes = [82, 95, 96, 99]  # Violent rain, thunderstorms
    if weather_data["weather_code"] in dangerous_codes:
        base_prob += 30
    
    # 24-hour rainfall forecast
    if weather_data["rainfall_24h"]:
        max_rainfall = max(weather_data["rainfall_24h"])
        if max_rainfall > 15:
            base_prob += 20
        elif max_rainfall > 10:
            base_prob += 10
    
    return min(95, max(5, base_prob))

def get_groq_weather_interpretation(weather_data):
    """Get AI weather interpretation using Groq API"""
    if not weather_data or not isinstance(weather_data, dict):
        return ["⚠️ Weather data unavailable - using basic analysis"]
    
    try:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {GROQ_API_KEY}"
        }
        
        # Safely get weather data with fallbacks
        temperature = weather_data.get('temperature', 28)
        humidity = weather_data.get('humidity', 70)
        precipitation = weather_data.get('precipitation', 0)
        wind_speed = weather_data.get('wind_speed', 15)
        weather_condition = weather_data.get('weather_condition', 'Unknown')
        city = weather_data.get('city', 'Metro Manila')
        
        # Get first 6 hours of rainfall forecast safely
        rainfall_forecast = []
        rainfall_24h = weather_data.get('rainfall_24h', [])
        if rainfall_24h and len(rainfall_24h) >= 6:
            try:
                rainfall_forecast = [f'{float(r):.1f}' for r in rainfall_24h[:6]]
            except (ValueError, TypeError):
                rainfall_forecast = ['0.0'] * 6
        else:
            rainfall_forecast = ['0.0'] * 6
        
        weather_summary = f"""
        Current Weather Data for {city}:
        - Temperature: {temperature:.1f}°C
        - Humidity: {humidity:.1f}%
        - Current Rainfall: {precipitation:.1f}mm
        - Wind Speed: {wind_speed:.1f}kph
        - Weather Condition: {weather_condition}
        - Next 6 Hours Rainfall: {', '.join(rainfall_forecast)}mm
        """
        
        prompt = f"""You are a professional weather forecaster in the Philippines, specifically analyzing conditions for school suspension decisions in Metro Manila.

{weather_summary}

As a weather expert, provide 3-4 practical interpretations focusing on:
1. Impact on students' safety and transportation
2. Specific risks during school hours (7 AM - 5 PM)
3. Travel conditions for parents and school buses
4. Overall recommendation for school operations

Write in a conversational, easy-to-understand tone that Filipino families would appreciate. Include relevant emojis and be specific about timing and impacts.

Format each interpretation as a short paragraph (2-3 sentences max). Think about real concerns parents have: Will my child get soaked? Are roads safe? Should I pick them up early?

IMPORTANT: Do not use asterisks (*) or any markdown formatting. Write in plain text with emojis only."""

        data = {
            "model": "llama-3.3-70b-versatile",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_tokens": 400
        }
        
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=20
        )
        
        if response.status_code == 200:
            ai_response = response.json()['choices'][0]['message']['content']
            
            # Clean up any remaining markdown formatting
            def clean_markdown(text):
                import re
                # Remove bold formatting
                text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
                text = re.sub(r'\*(.*?)\*', r'\1', text)
                # Remove other markdown
                text = re.sub(r'__(.*?)__', r'\1', text)
                text = re.sub(r'_(.*?)_', r'\1', text)
                # Clean up extra spaces
                text = re.sub(r'\s+', ' ', text)
                return text.strip()
            
            # Split into interpretations and clean each one
            interpretations = [clean_markdown(line.strip()) for line in ai_response.split('\n') if line.strip()]
            
            return interpretations[:4] if interpretations else ["🤖 AI analysis completed successfully"]
        else:
            return [f"🤖 AI service temporarily unavailable (Status: {response.status_code})"]
            
    except requests.exceptions.RequestException as e:
        return [f"🌐 Network error connecting to AI service: {str(e)[:30]}..."]
    except Exception as e:
        return [f"🤖 AI analysis error: {str(e)[:50]}..."]

def create_rainfall_chart(rainfall_data, city_name):
    """Create 24-hour rainfall forecast chart"""
    if not rainfall_data or len(rainfall_data) == 0:
        # Create empty chart
        fig = go.Figure()
        fig.add_annotation(
            text="No rainfall forecast data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="#374151")
        )
        fig.update_layout(
            title=f"24-Hour Rainfall Forecast - {city_name}",
            height=400,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        return fig
    
    try:
        # Ensure we have 24 hours of data
        if len(rainfall_data) < 24:
            rainfall_data = list(rainfall_data) + [0] * (24 - len(rainfall_data))
        else:
            rainfall_data = rainfall_data[:24]
        
        # Convert to float and handle None values
        processed_data = []
        for value in rainfall_data:
            try:
                processed_data.append(float(value) if value is not None else 0.0)
            except (ValueError, TypeError):
                processed_data.append(0.0)
        
        hours = [f"{i:02d}:00" for i in range(24)]
        
        # Helper function to get rainfall interpretation
        def get_rainfall_interpretation(mm):
            if mm >= 20:
                return {
                    'intensity': 'Extreme Rain ⚠️',
                    'impact': 'Severe flooding likely, dangerous travel conditions',
                    'advice': 'Stay indoors, avoid all travel if possible'
                }
            elif mm >= 15:
                return {
                    'intensity': 'Very Heavy Rain 🌊',
                    'impact': 'Major flooding possible, very dangerous for commuting',
                    'advice': 'Avoid non-essential travel, school suspension likely'
                }
            elif mm >= 7.5:
                return {
                    'intensity': 'Heavy Rain ⛈️',
                    'impact': 'Localized flooding, difficult travel conditions',
                    'advice': 'Use caution when traveling, bring umbrella'
                }
            elif mm >= 2.5:
                return {
                    'intensity': 'Moderate Rain 🌧️',
                    'impact': 'Wet roads, reduced visibility',
                    'advice': 'Normal precautions needed, light rain gear'
                }
            elif mm > 0:
                return {
                    'intensity': 'Light Rain 💧',
                    'impact': 'Minimal impact on travel',
                    'advice': 'Light drizzle, umbrella recommended'
                }
            else:
                return {
                    'intensity': 'No Rain ☀️',
                    'impact': 'Clear conditions',
                    'advice': 'Good weather for outdoor activities'
                }
        
        fig = go.Figure()
        
        # Color coding based on rainfall intensity with better contrast
        colors = []
        hover_texts = []
        
        for i, mm in enumerate(processed_data):
            # Color coding
            if mm >= 20:
                colors.append('#991b1b')  # Dark red for extreme rain
            elif mm >= 15:
                colors.append('#dc2626')  # Red for very heavy rain
            elif mm >= 7.5:
                colors.append('#f59e0b')  # Orange for heavy rain
            elif mm >= 2.5:
                colors.append('#3b82f6')  # Blue for moderate rain
            elif mm > 0:
                colors.append('#60a5fa')  # Light blue for light rain
            else:
                colors.append('#e5e7eb')  # Gray for no rain
            
            # Create detailed hover text
            interpretation = get_rainfall_interpretation(mm)
            hover_text = f"""
<b>Time: {hours[i]}</b><br>
<b>Rainfall: {mm:.1f}mm/hr</b><br>
<b>{interpretation['intensity']}</b><br>
<i>Impact:</i> {interpretation['impact']}<br>
<i>Recommendation:</i> {interpretation['advice']}
            """.strip()
            hover_texts.append(hover_text)
        
        fig.add_trace(go.Bar(
            x=hours,
            y=processed_data,
            marker_color=colors,
            name='Rainfall (mm)',
            text=[f'{mm:.1f}mm' if mm > 0.1 else '' for mm in processed_data],
            textposition='outside',
            textfont=dict(size=10, color='#374151'),
            hovertemplate='%{customdata}<extra></extra>',
            customdata=hover_texts
        ))
        
        # Add rainfall intensity reference lines
        fig.add_hline(y=20, line_dash="dash", line_color="#991b1b", 
                     annotation_text="Extreme Rain (20mm)", annotation_position="top right")
        fig.add_hline(y=15, line_dash="dash", line_color="#dc2626", 
                     annotation_text="Very Heavy Rain (15mm)", annotation_position="top right")
        fig.add_hline(y=7.5, line_dash="dash", line_color="#f59e0b", 
                     annotation_text="Heavy Rain (7.5mm)", annotation_position="top right")
        
        fig.update_layout(
            title=dict(
                text=f"24-Hour Rainfall Forecast - {city_name}",
                font=dict(size=18, color='#1f2937', family='Inter')
            ),
            xaxis=dict(
                title="Hour of Day",
                tickangle=45,
                title_font=dict(size=14, color='#374151'),
                tickfont=dict(size=12, color='#374151')
            ),
            yaxis=dict(
                title="Rainfall (mm/hr)",
                title_font=dict(size=14, color='#374151'),
                tickfont=dict(size=12, color='#374151')
            ),
            height=450,
            showlegend=False,
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(family='Inter', color='#374151'),
            hoverlabel=dict(
                bgcolor="white",
                font_size=12,
                font_family="Inter",
                font_color="#374151",
                bordercolor="#e5e7eb"
            )
        )
        
        # Add grid for better readability
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#f3f4f6')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#f3f4f6')
        
        return fig
        
    except Exception as e:
        # Return error chart
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error creating chart: {str(e)[:50]}...",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color="#dc2626")
        )
        fig.update_layout(
            title=f"24-Hour Rainfall Forecast - {city_name}",
            height=400,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        return fig

def render_sidebar():
    """Render sidebar with LGU list"""
    st.sidebar.markdown("""
    <div style='text-align: center; padding: 1rem; border-bottom: 1px solid #e5e7eb;'>
        <h1 style='font-size: 1.5rem; font-weight: 800; color: #1f2937; margin: 0;'>May Pasok Ba?</h1>
        <p style='color: #6b7280; margin: 0; font-size: 0.9rem;'>Class Suspension Advisory</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Get weather data for all LGUs
    if 'weather_data' not in st.session_state or not st.session_state.weather_data:
        st.session_state.weather_data = {}
        
        progress_bar = st.sidebar.progress(0)
        status_text = st.sidebar.empty()
        
        total_lgus = len(METRO_MANILA_LGUS)
        
        for i, (lgu, coords) in enumerate(METRO_MANILA_LGUS.items()):
            status_text.text(f"Loading {lgu}...")
            progress_bar.progress((i + 1) / total_lgus)
            
            try:
                weather = fetch_weather_data(coords["lat"], coords["lon"], lgu)
                if weather:
                    st.session_state.weather_data[lgu] = weather
                else:
                    # Create fallback data if API fails
                    st.session_state.weather_data[lgu] = {
                        "city": lgu,
                        "temperature": 28.0,
                        "humidity": 75.0,
                        "wind_speed": 15.0,
                        "precipitation": 0.0,
                        "weather_condition": "Partly cloudy",
                        "weather_code": 2,
                        "rainfall_24h": [0] * 24,
                        "hourly_data": {},
                        "last_updated": "API Error - Using fallback data"
                    }
                time.sleep(0.2)  # Rate limiting
            except Exception as e:
                st.sidebar.error(f"Error loading {lgu}: {str(e)[:50]}...")
                # Create fallback data
                st.session_state.weather_data[lgu] = {
                    "city": lgu,
                    "temperature": 28.0,
                    "humidity": 75.0,
                    "wind_speed": 15.0,
                    "precipitation": 0.0,
                    "weather_condition": "Data unavailable",
                    "weather_code": 1,
                    "rainfall_24h": [0] * 24,
                    "hourly_data": {},
                    "last_updated": "Error - Using fallback"
                }
        
        progress_bar.empty()
        status_text.empty()
    
    # Calculate probabilities and sort
    lgu_probabilities = {}
    for lgu, weather in st.session_state.weather_data.items():
        try:
            prob = calculate_suspension_probability(weather)
            lgu_probabilities[lgu] = prob
        except Exception as e:
            st.sidebar.warning(f"Error calculating probability for {lgu}")
            lgu_probabilities[lgu] = 30  # Default probability
    
    # Ensure we have data before proceeding
    if not lgu_probabilities:
        st.sidebar.error("No weather data available. Please refresh the page.")
        return "Metro Manila Overview", {}
    
    sorted_lgus = sorted(lgu_probabilities.items(), key=lambda x: x[1], reverse=True)
    
    # Metro Manila recommendation
    high_risk_count = sum(1 for _, prob in lgu_probabilities.items() if prob >= 70)
    total_lgus = len(lgu_probabilities)
    
    if total_lgus > 0:
        if high_risk_count >= total_lgus * 0.6:
            recommendation = "SUSPENSION RECOMMENDED"
            rec_class = "recommendation-card"
        else:
            recommendation = "LOW SUSPENSION PROBABILITY"
            rec_class = "recommendation-card safe"
        
        st.sidebar.markdown(f"""
        <div class="{rec_class}">
            <div style='font-size: 1.2rem; font-weight: 700;'>{recommendation}</div>
            <div style='font-size: 0.9rem; opacity: 0.9;'>{high_risk_count} out of {total_lgus} LGUs</div>
        </div>
        """, unsafe_allow_html=True)
    
    # LGU selection
    st.sidebar.markdown("### Select LGU for Detailed Analysis")
    
    lgu_options = ["Metro Manila Overview"]
    if sorted_lgus:
        lgu_options.extend([lgu for lgu, _ in sorted_lgus])
    
    selected_lgu = st.sidebar.selectbox(
        "Choose an LGU:",
        lgu_options,
        index=0
    )
    
    # Display LGU list with probabilities
    if sorted_lgus:
        st.sidebar.markdown("### All LGUs (Ranked by Risk)")
        for lgu, prob in sorted_lgus:
            prob_class = "prob-high" if prob >= 70 else "prob-medium" if prob >= 40 else "prob-low"
            st.sidebar.markdown(f"""
            <div class="lgu-card" style='margin: 0.3rem 0; padding: 0.8rem;'>
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-weight: 500;">{lgu}</span>
                    <span style="font-weight: 700; font-size: 1.1rem;">{prob}%</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    st.sidebar.markdown("""
    <div style='padding: 1rem; border-top: 1px solid #e5e7eb; margin-top: 1rem;'>
        <small style='color: #6b7280;'>
            <i>⚠️ This is a recommendation system. Always check official LGU announcements.</i>
        </small>
    </div>
    """, unsafe_allow_html=True)
    
    return selected_lgu, lgu_probabilities

def render_lgu_details(lgu_name, weather_data, probability):
    """Render detailed view for specific LGU"""
    if not weather_data:
        st.error(f"No weather data available for {lgu_name}. Please refresh or try again later.")
        return
    
    st.markdown(f"""
    <h1 class="main-header">{lgu_name}</h1>
    <p class="subtitle">Detailed Weather Analysis & Class Suspension Assessment</p>
    """, unsafe_allow_html=True)
    
    # Main metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        prob_class = "prob-high" if probability >= 70 else "prob-medium" if probability >= 40 else "prob-low"
        st.markdown(f"""
        <div class="stat-card">
            <div style="text-align: center;">
                <h4 style="color: #6b7280; margin: 0;">Suspension Probability</h4>
                <div class="probability-display {prob_class}">{probability}%</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        temperature = weather_data.get('temperature', 28)
        temp_desc = "Hot" if temperature > 32 else "Warm" if temperature > 28 else "Comfortable"
        st.markdown(f"""
        <div class="weather-metric">
            <div style="color: #6b7280; font-size: 0.9rem;">🌡️ Temperature</div>
            <h3 style="color: #1f2937;">{temperature:.1f}°C</h3>
            <small style="color: #9ca3af;">{temp_desc}</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        precipitation = weather_data.get('precipitation', 0)
        rain_color = "#dc2626" if precipitation > 15 else "#f59e0b" if precipitation > 5 else "#10b981"
        rain_desc = "Heavy" if precipitation > 15 else "Moderate" if precipitation > 5 else "Light" if precipitation > 0 else "None"
        st.markdown(f"""
        <div class="weather-metric">
            <div style="color: #6b7280; font-size: 0.9rem;">🌧️ Current Rain</div>
            <h3 style="color: {rain_color};">{precipitation:.1f}mm</h3>
            <small style="color: #9ca3af;">{rain_desc}</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        wind_speed = weather_data.get('wind_speed', 15)
        wind_color = "#dc2626" if wind_speed > 50 else "#f59e0b" if wind_speed > 30 else "#10b981"
        wind_desc = "Strong" if wind_speed > 30 else "Moderate" if wind_speed > 15 else "Light"
        st.markdown(f"""
        <div class="weather-metric">
            <div style="color: #6b7280; font-size: 0.9rem;">💨 Wind Speed</div>
            <h3 style="color: {wind_color};">{wind_speed:.1f} kph</h3>
            <small style="color: #9ca3af;">{wind_desc}</small>
        </div>
        """, unsafe_allow_html=True)
    
    # Rainfall Chart - Full Width
    st.markdown('<div class="stat-card">', unsafe_allow_html=True)
    try:
        rainfall_data = weather_data.get('rainfall_24h', [])
        if rainfall_data:
            chart = create_rainfall_chart(rainfall_data, lgu_name)
            st.plotly_chart(chart, use_container_width=True)
        else:
            st.info("📊 24-hour rainfall forecast data is being loaded...")
    except Exception as e:
        st.warning(f"Chart unavailable: {str(e)[:50]}...")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # AI Weather Forecaster Section - After Chart
    st.markdown("### 🧠 AI Weather Forecaster")
    
    with st.spinner("Getting AI weather interpretation..."):
        try:
            interpretations = get_groq_weather_interpretation(weather_data)
            
            if interpretations and any(i.strip() for i in interpretations):
                for interpretation in interpretations:
                    if interpretation and interpretation.strip():
                        st.markdown(f"""
                        <div class="stat-card" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; margin: 0.5rem 0;">
                            <div style="padding: 1rem;">
                                {interpretation}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="stat-card" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white;">
                    <div style="padding: 1rem;">
                        🤖 AI analysis is processing weather data for detailed insights...
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
        except Exception as e:
            st.markdown(f"""
            <div class="stat-card" style="background: linear-gradient(135deg, #f59e0b 0%, #f97316 100%); color: white;">
                <div style="padding: 1rem;">
                    ⚠️ AI temporarily unavailable. Weather data shows {weather_data.get('weather_condition', 'current conditions')} with {precipitation:.1f}mm rainfall.
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Additional weather details
    st.markdown("### Additional Weather Information")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        weather_condition = weather_data.get('weather_condition', 'Unknown')
        st.markdown(f"""
        <div class="stat-card">
            <h4 style="color: #1f2937; font-weight: 600; margin-bottom: 0.5rem;">Weather Condition</h4>
            <p style="font-size: 1.4rem; font-weight: 700; color: #374151; margin: 0.5rem 0;">{weather_condition}</p>
            <small style="color: #6b7280;">Current conditions</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        humidity = weather_data.get('humidity', 70)
        humidity_desc = "Very humid" if humidity > 80 else "Humid" if humidity > 60 else "Comfortable"
        st.markdown(f"""
        <div class="stat-card">
            <h4 style="color: #1f2937; font-weight: 600; margin-bottom: 0.5rem;">Humidity</h4>
            <p style="font-size: 1.4rem; font-weight: 700; color: #374151; margin: 0.5rem 0;">{humidity:.1f}%</p>
            <small style="color: #6b7280;">{humidity_desc}</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        last_updated = weather_data.get('last_updated', 'Unknown')
        data_source = weather_data.get('data_source', 'Weather API')
        st.markdown(f"""
        <div class="stat-card">
            <h4 style="color: #1f2937; font-weight: 600; margin-bottom: 0.5rem;">Last Updated</h4>
            <p style="font-size: 1.4rem; font-weight: 700; color: #374151; margin: 0.5rem 0;">{last_updated}</p>
            <small style="color: #6b7280;">{data_source}</small>
        </div>
        """, unsafe_allow_html=True)

def render_overview(lgu_probabilities):
    """Render Metro Manila overview"""
    st.markdown("""
    <h1 class="main-header">Metro Manila Overview</h1>
    <p class="subtitle">Real-time Weather Monitoring & Regional Assessment</p>
    """, unsafe_allow_html=True)
    
    # Check if we have valid data
    if not lgu_probabilities:
        st.error("⚠️ No weather data available. Please refresh the page or check your internet connection.")
        if st.button("🔄 Retry Loading Data"):
            st.session_state.weather_data = {}
            st.rerun()
        return
    
    # Calculate Metro Manila-wide binary recommendation
    try:
        high_risk_count = sum(1 for prob in lgu_probabilities.values() if prob >= 70)
        total_lgus = len(lgu_probabilities)
        avg_prob = sum(lgu_probabilities.values()) / total_lgus if total_lgus > 0 else 0
        
        # Binary recommendation logic
        suspension_threshold = 0.6  # 60% of LGUs need high risk
        if high_risk_count >= total_lgus * suspension_threshold:
            metro_answer = "WALANG PASOK"
            metro_subtitle = "Classes Suspended"
            recommendation_color = "#dc2626"
            recommendation_bg = "#fef2f2"
            recommendation_icon = "❌"
            status_emoji = "🚫"
        else:
            metro_answer = "MAY PASOK"
            metro_subtitle = "Classes Continue"
            recommendation_color = "#16a34a"
            recommendation_bg = "#f0fdf4"
            recommendation_icon = "✅"
            status_emoji = "🎒"
        
    except Exception as e:
        st.error(f"Error calculating statistics: {str(e)}")
        return
    
    # Main Metro Manila Recommendation Card - IMPROVED BINARY VERSION
    st.markdown(f"""
    <div style="background: {recommendation_bg}; border: 3px solid {recommendation_color}; border-radius: 20px; padding: 4rem 2rem; margin: 2rem 0; text-align: center; box-shadow: 0 12px 40px rgba(0,0,0,0.15);">
        <div style="font-size: 2rem; color: {recommendation_color}; font-weight: 700; margin-bottom: 1.5rem; opacity: 0.9;">
            May Pasok Ba sa Metro Manila?
        </div>
        <h1 style="color: {recommendation_color}; font-size: 6rem; margin: 1rem 0; font-weight: 900; line-height: 0.9; text-shadow: 2px 2px 4px rgba(0,0,0,0.1);">
            {metro_answer}
        </h1>
        <div style="color: {recommendation_color}; font-size: 1.3rem; margin: 1rem 0; font-weight: 600; opacity: 0.8;">
            {recommendation_icon} {metro_subtitle} {status_emoji}
        </div>
        <hr style="border: none; height: 2px; background: {recommendation_color}; margin: 2rem 0; opacity: 0.3;">
        <div style="color: {recommendation_color}; font-size: 1rem; opacity: 0.7;">
            {high_risk_count}/{total_lgus} LGUs high risk • {avg_prob:.0f}% avg probability
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Statistics cards
    col1, col2, col3, col4 = st.columns(4)
    
    high_risk = sum(1 for prob in lgu_probabilities.values() if prob >= 70)
    medium_risk = sum(1 for prob in lgu_probabilities.values() if 40 <= prob < 70)
    low_risk = sum(1 for prob in lgu_probabilities.values() if prob < 40)
    
    with col1:
        st.markdown(f"""
        <div class="stat-card">
            <h4 style="color: #dc2626;">High Risk LGUs</h4>
            <div style="font-size: 2.5rem; font-weight: 700; color: #dc2626;">{high_risk}</div>
            <small>≥70% probability</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="stat-card">
            <h4 style="color: #f59e0b;">Medium Risk LGUs</h4>
            <div style="font-size: 2.5rem; font-weight: 700; color: #f59e0b;">{medium_risk}</div>
            <small>40-69% probability</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="stat-card">
            <h4 style="color: #10b981;">Low Risk LGUs</h4>
            <div style="font-size: 2.5rem; font-weight: 700; color: #10b981;">{low_risk}</div>
            <small>&lt;40% probability</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        # Show the binary decision more clearly
        decision_text = "WALANG PASOK" if "WALANG" in metro_answer else "MAY PASOK"
        decision_color = "#dc2626" if "WALANG" in metro_answer else "#16a34a"
        st.markdown(f"""
        <div class="stat-card">
            <h4 style="color: #6366f1;">Final Decision</h4>
            <div style="font-size: 1.8rem; font-weight: 700; color: {decision_color};">
                {decision_text}
            </div>
            <small>Metro Manila</small>
        </div>
        """, unsafe_allow_html=True)
    
    # Real-time satellite weather map (like Zoom Earth)
    st.markdown("### Metro Manila Real-time Satellite Weather")
    
    # Option selector for different weather views
    col1, col2 = st.columns([3, 1])
    
    with col2:
        weather_layer = st.selectbox(
            "Weather Layer:",
            ["Rain & Clouds", "Satellite View", "Wind Patterns", "Temperature"],
            index=0
        )
    
    # Map layer configurations for Windy.com embed
    layer_configs = {
        "Rain & Clouds": {"overlay": "rain", "level": "surface"},
        "Satellite View": {"overlay": "satellite", "level": "surface"}, 
        "Wind Patterns": {"overlay": "wind", "level": "10m"},
        "Temperature": {"overlay": "temp", "level": "2m"}
    }
    
    selected_config = layer_configs[weather_layer]
    
    # Embed Windy.com map focused on Metro Manila
    windy_embed_url = f"""
    https://embed.windy.com/embed2.html?lat=14.6091&lon=121.0223&detailLat=14.6091&detailLon=121.0223&width=100%25&height=500&zoom=11&level={selected_config['level']}&overlay={selected_config['overlay']}&product=ecmwf&menu=&message=&marker=&calendar=now&pressure=&type=map&location=coordinates&detail=&metricWind=default&metricTemp=default&radarRange=-1
    """
    
    # Display the embedded weather map
    st.components.v1.iframe(
        src=windy_embed_url,
        height=500,
        scrolling=False
    )
    
    # Add LGU suspension overlay below the satellite map
    st.markdown("### LGU Suspension Risk Overview")
    
    try:
        # Create a simple grid view of LGUs with their risk levels
        sorted_lgus = sorted(lgu_probabilities.items(), key=lambda x: x[1], reverse=True)
        
        # Create 3 columns for risk levels
        col1, col2, col3 = st.columns(3)
        
        high_risk_lgus = [(lgu, prob) for lgu, prob in sorted_lgus if prob >= 70]
        medium_risk_lgus = [(lgu, prob) for lgu, prob in sorted_lgus if 40 <= prob < 70]
        low_risk_lgus = [(lgu, prob) for lgu, prob in sorted_lgus if prob < 40]
        
        with col1:
            st.markdown("#### 🔴 High Risk (≥70%)")
            if high_risk_lgus:
                for lgu, prob in high_risk_lgus:
                    weather = st.session_state.weather_data.get(lgu, {})
                    st.markdown(f"""
                    <div style="background: #fef2f2; border-left: 4px solid #dc2626; padding: 0.75rem; margin: 0.5rem 0; border-radius: 4px;">
                        <strong style="color: #dc2626;">{lgu}</strong><br>
                        <span style="color: #991b1b; font-size: 1.1rem; font-weight: 600;">{prob}%</span><br>
                        <small style="color: #7f1d1d;">{weather.get('weather_condition', 'N/A')} | {weather.get('precipitation', 0):.1f}mm</small>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No high-risk LGUs")
        
        with col2:
            st.markdown("#### 🟡 Medium Risk (40-69%)")
            if medium_risk_lgus:
                for lgu, prob in medium_risk_lgus:
                    weather = st.session_state.weather_data.get(lgu, {})
                    st.markdown(f"""
                    <div style="background: #fffbeb; border-left: 4px solid #f59e0b; padding: 0.75rem; margin: 0.5rem 0; border-radius: 4px;">
                        <strong style="color: #d97706;">{lgu}</strong><br>
                        <span style="color: #92400e; font-size: 1.1rem; font-weight: 600;">{prob}%</span><br>
                        <small style="color: #78350f;">{weather.get('weather_condition', 'N/A')} | {weather.get('precipitation', 0):.1f}mm</small>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No medium-risk LGUs")
        
        with col3:
            st.markdown("#### 🟢 Low Risk (<40%)")
            if low_risk_lgus:
                for lgu, prob in low_risk_lgus:
                    weather = st.session_state.weather_data.get(lgu, {})
                    st.markdown(f"""
                    <div style="background: #f0fdf4; border-left: 4px solid #10b981; padding: 0.75rem; margin: 0.5rem 0; border-radius: 4px;">
                        <strong style="color: #059669;">{lgu}</strong><br>
                        <span style="color: #047857; font-size: 1.1rem; font-weight: 600;">{prob}%</span><br>
                        <small style="color: #065f46;">{weather.get('weather_condition', 'N/A')} | {weather.get('precipitation', 0):.1f}mm</small>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No low-risk LGUs")
    
    except Exception as e:
        st.error(f"Error displaying LGU overview: {str(e)}")
    
    # Top 5 High-Risk LGUs
    st.markdown("### Top 5 High-Risk LGUs")
    
    try:
        sorted_lgus = sorted(lgu_probabilities.items(), key=lambda x: x[1], reverse=True)[:5]
        
        if sorted_lgus:
            for i, (lgu, prob) in enumerate(sorted_lgus, 1):
                weather = st.session_state.weather_data.get(lgu, {})
                prob_class = "prob-high" if prob >= 70 else "prob-medium" if prob >= 40 else "prob-low"
                
                # Get weather details safely
                weather_condition = weather.get('weather_condition', 'N/A')
                temperature = weather.get('temperature', 0)
                precipitation = weather.get('precipitation', 0)
                
                st.markdown(f"""
                <div class="stat-card">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div style="flex: 1;">
                            <h3 style="margin: 0; color: #1f2937; font-size: 1.3rem; font-weight: 700;">{i}. {lgu}</h3>
                            <p style="margin: 5px 0 0 0; color: #6b7280; font-size: 0.9rem;">
                                {weather_condition} | {temperature:.1f}°C | {precipitation:.1f}mm rain
                            </p>
                        </div>
                        <div class="probability-display {prob_class}" style="font-size: 2.5rem; margin-left: 1rem;">{prob}%</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("No LGU data available for ranking")
    
    except Exception as e:
        st.error(f"Error displaying top LGUs: {str(e)}")
        
        # Show fallback content
        st.markdown("""
        <div class="stat-card">
            <h4>Data Loading</h4>
            <p>Weather data is still being processed. Please refresh the page in a moment.</p>
        </div>
        """, unsafe_allow_html=True)
    

    
    # Top 5 High-Risk LGUs
    st.markdown("### Top 5 High-Risk LGUs")
    
    try:
        sorted_lgus = sorted(lgu_probabilities.items(), key=lambda x: x[1], reverse=True)[:5]
        
        if sorted_lgus:
            for i, (lgu, prob) in enumerate(sorted_lgus, 1):
                weather = st.session_state.weather_data.get(lgu, {})
                prob_class = "prob-high" if prob >= 70 else "prob-medium" if prob >= 40 else "prob-low"
                
                # Get weather details safely
                weather_condition = weather.get('weather_condition', 'N/A')
                temperature = weather.get('temperature', 0)
                precipitation = weather.get('precipitation', 0)
                
                st.markdown(f"""
                <div class="stat-card">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div style="flex: 1;">
                            <h3 style="margin: 0; color: #1f2937; font-size: 1.3rem; font-weight: 700;">{i}. {lgu}</h3>
                            <p style="margin: 5px 0 0 0; color: #6b7280; font-size: 0.9rem;">
                                {weather_condition} | {temperature:.1f}°C | {precipitation:.1f}mm rain
                            </p>
                        </div>
                        <div class="probability-display {prob_class}" style="font-size: 2.5rem; margin-left: 1rem;">{prob}%</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("No LGU data available for ranking")
    
    except Exception as e:
        st.error(f"Error displaying top LGUs: {str(e)}")
        
        # Show fallback content
        st.markdown("""
        <div class="stat-card">
            <h4>Data Loading</h4>
            <p>Weather data is still being processed. Please refresh the page in a moment.</p>
        </div>
        """, unsafe_allow_html=True)

def main():
    # Sticky disclaimer at the very top - always visible
    st.markdown("""
    <div style="
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        z-index: 999999;
        background: linear-gradient(135deg, #fbbf24 0%, #f59e0b 100%);
        color: #92400e;
        padding: 0.75rem 1rem;
        font-weight: 600;
        font-size: 0.9rem;
        border-bottom: 2px solid #d97706;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    ">
        ⚠️ <strong>DISCLAIMER:</strong> This system provides AI-powered recommendations based on weather data analysis. 
        Official class suspension announcements are made solely by Local Government Units (LGUs) and school administrators.
        Always check official sources for final decisions.
    </div>
    
    <!-- Add spacing to prevent content overlap -->
    <div style="height: 80px;"></div>
    """, unsafe_allow_html=True)
    
    # Initialize session state for data refresh
    if 'last_refresh' not in st.session_state:
        st.session_state.last_refresh = datetime.now()
        st.session_state.weather_data = {}
    
    # Auto-refresh every 15 minutes
    if datetime.now() - st.session_state.last_refresh > timedelta(minutes=15):
        st.session_state.weather_data = {}
        st.session_state.last_refresh = datetime.now()
        st.rerun()
    
    # Render sidebar and get selection
    selected_lgu, lgu_probabilities = render_sidebar()
    
    # Render main content based on selection
    if selected_lgu == "Metro Manila Overview":
        render_overview(lgu_probabilities)
    else:
        weather_data = st.session_state.weather_data.get(selected_lgu)
        if weather_data:
            probability = calculate_suspension_probability(weather_data)
            render_lgu_details(selected_lgu, weather_data, probability)
        else:
            st.error(f"Weather data not available for {selected_lgu}")
    
    # Footer with refresh info
    st.markdown("---")
    st.markdown(f"""
    <div style="text-align: center; color: #6b7280; font-size: 0.9rem;">
        <p>Last updated: {st.session_state.last_refresh.strftime('%Y-%m-%d %H:%M:%S')} | 
        Data refreshes every 15 minutes | 
        Powered by OpenMeteo API & Groq AI</p>
        <p><small>Built with ❤️ for Filipino families</small></p>
    </div>
    """, unsafe_allow_html=True)
    
    # Auto-refresh button
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("🔄 Refresh Weather Data", type="primary"):
            st.session_state.weather_data = {}
            st.session_state.last_refresh = datetime.now()
            st.rerun()

if __name__ == "__main__":
    main()