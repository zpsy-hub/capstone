import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import random
from typing import Dict, List
import json

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

def get_probability_color(prob):
    """Return color class based on probability"""
    if prob >= 0.7:
        return "prob-high"
    elif prob >= 0.4:
        return "prob-medium"
    else:
        return "prob-low"

def get_metro_manila_recommendation(df):
    """Get overall Metro Manila recommendation based on majority rule"""
    high_prob_cities = len(df[df['morning_probability'] >= 0.7])
    total_cities = len(df)
    
    if high_prob_cities >= total_cities * 0.6:
        return "SUSPEND", high_prob_cities / total_cities
    else:
        return "NO SUSPENSION", high_prob_cities / total_cities

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
    
    # Get overall recommendation
    recommendation, confidence = get_metro_manila_recommendation(df)
    
    # Display main recommendation
    if recommendation == "SUSPEND":
        st.markdown(f'''
        <div class="suspension-alert">
            ⚠️ HIGH PROBABILITY OF CLASS SUSPENSION
            <br>
            <span style="font-size: 1rem;">Metro Manila-wide confidence: {confidence:.1%}</span>
        </div>
        ''', unsafe_allow_html=True)
    else:
        st.markdown(f'''
        <div class="no-suspension-alert">
            ✅ LOW PROBABILITY OF CLASS SUSPENSION
            <br>
            <span style="font-size: 1rem;">Metro Manila-wide confidence: {(1-confidence):.1%}</span>
        </div>
        ''', unsafe_allow_html=True)
    
    # Prediction cycles
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🌅 Morning Prediction (05:00)")
        st.info("Updated with overnight weather data - Higher confidence")
        
    with col2:
        st.subheader("🌙 Evening Prediction (19:00)")
        st.info("24-hour forecast based - Planning purposes")
    
    # Map visualization
    st.subheader("📍 Interactive Map")
    map_fig = create_map_visualization(df)
    st.plotly_chart(map_fig, use_container_width=True)
    
    # Add some spacing before city breakdown
    st.markdown("<br>", unsafe_allow_html=True)
    
    # City-by-city breakdown
    st.subheader("🏙️ City-by-City Breakdown")
    
    # Sort cities by morning probability (highest first)
    df_sorted = df.sort_values('morning_probability', ascending=False)
    
    # Display cities in expanders
    for _, row in df_sorted.iterrows():
        with st.expander(f"{row['city']} - {row['morning_probability']:.1%} probability"):
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Morning Probability", f"{row['morning_probability']:.1%}")
                st.metric("Evening Probability", f"{row['evening_probability']:.1%}")
            
            with col2:
                st.metric("Temperature", f"{row['weather_factors']['temperature']:.1f}°C")
                st.metric("Humidity", f"{row['weather_factors']['humidity']:.1f}%")
            
            with col3:
                st.metric("Rainfall", f"{row['weather_factors']['rainfall_mm']:.1f} mm")
                st.metric("Wind Speed", f"{row['weather_factors']['wind_speed_kph']:.1f} kph")
            
            with col4:
                st.metric("6h Rain Prob", f"{row['weather_factors']['precipitation_prob_6h']:.1f}%")
                st.metric("Weather", row['weather_factors']['weather_condition'])
            
            # Additional details in a second row
            col5, col6, col7, col8 = st.columns(4)
            
            with col5:
                st.metric("Flood Risk", row['weather_factors']['flood_risk'])
            with col6:
                st.metric("TCWS Level", row['weather_factors']['tcws_level'])
            with col7:
                st.metric("Last Updated", row['weather_factors']['last_updated'])
            with col8:
                # Weather-based recommendation
                factors = row['weather_factors']
                risk_score = 0
                
                # Calculate risk score based on weather factors
                if factors['rainfall_mm'] > 15: 
                    risk_score += 2
                elif factors['rainfall_mm'] > 5: 
                    risk_score += 1
                
                if factors['wind_speed_kph'] > 50: 
                    risk_score += 2
                elif factors['wind_speed_kph'] > 30: 
                    risk_score += 1
                
                if factors['tcws_level'] > 0: 
                    risk_score += 2
                
                if factors['flood_risk'] == 'High': 
                    risk_score += 1
                
                risk_level = "🟢 Low" if risk_score <= 2 else "🟡 Medium" if risk_score <= 4 else "🔴 High"
                st.metric("Weather Risk", risk_level)
            
            # 24-hour forecast chart if data is available
            if row['weather_factors']['hourly_data']:
                forecast_chart = create_24hour_forecast_chart(
                    row['weather_factors']['hourly_data'], 
                    row['city']
                )
                if forecast_chart:
                    st.plotly_chart(forecast_chart, use_container_width=True)
    
    # RSS Feeds Section
    st.subheader("📡 Official Updates & News Feeds")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🌪️ PAGASA Updates")
        
        # Initialize session state for PAGASA RSS URL
        if 'pagasa_rss_url' not in st.session_state:
            st.session_state.pagasa_rss_url = ""
        
        # RSS URL input with sample suggestions
        sample_urls = get_sample_rss_urls()
        st.markdown("**Sample PAGASA RSS URLs:**")
        for url in sample_urls["pagasa"]:
            if st.button(f"📋 {url}", key=f"pagasa_{url}"):
                st.session_state.pagasa_rss_url = url
        
        pagasa_url = st.text_area(
            "PAGASA RSS Feed URL:", 
            value=st.session_state.pagasa_rss_url,
            placeholder="Enter PAGASA RSS feed URL...", 
            height=80,
            key="pagasa_input"
        )
        
        if st.button("🔄 Fetch PAGASA Updates", key="fetch_pagasa"):
            if pagasa_url:
                with st.spinner("Fetching PAGASA updates..."):
                    pagasa_items = fetch_rss_feed(pagasa_url)
                    st.session_state.pagasa_items = pagasa_items
            else:
                st.warning("Please enter a PAGASA RSS URL first!")
        
        # Display PAGASA updates
        st.markdown("**Latest PAGASA Updates:**")
        if 'pagasa_items' in st.session_state and st.session_state.pagasa_items:
            for item in st.session_state.pagasa_items:
                if "error" in item:
                    st.error(f"❌ {item['error']}")
                else:
                    with st.expander(f"🌩️ {item['title'][:80]}... ({item['published']})"):
                        st.write(f"**📅 Published:** {item['published']}")
                        st.write(f"**📝 Summary:** {item['summary']}")
                        st.markdown(f"**🔗 [Read Full Article]({item['link']})**")
        else:
            # Fallback sample updates
            pagasa_updates = [
                "🌩️ Thunderstorm advisory issued for Metro Manila - Valid until 6:00 PM",
                "🌧️ Yellow rainfall warning lifted for NCR as of 2:00 PM",
                "💨 Southwest monsoon continues to affect Luzon"
            ]
            for update in pagasa_updates:
                st.markdown(f"• {update}")
            st.info("👆 These are sample updates. Enter an RSS URL above to fetch real data!")
    
    with col2:
        st.subheader("📰 #WalangPasok Live Updates")
        
        # Initialize session state for News RSS URL
        if 'news_rss_url' not in st.session_state:
            st.session_state.news_rss_url = ""
        
        # Twitter #WalangPasok URLs via Google News
        st.markdown("**🐦 Twitter #WalangPasok (via Google News):**")
        for url in sample_urls["twitter_walangpasok"]:
            # Create shorter button labels
            if "#WalangPasok+when:1d" in url:
                label = "📋 #WalangPasok (24h)"
            elif "class+suspension" in url:
                label = "📋 Class Suspension (24h)"
            elif "WalangPasok+OR" in url:
                label = "📋 Multiple Hashtags (24h)"
            elif "Quezon+City+OR" in url:
                label = "📋 LGU Announcements (24h)"
            else:
                label = f"📋 Twitter Search"
                
            if st.button(label, key=f"twitter_{hash(url)}"):
                st.session_state.news_rss_url = url
        
        # Regular RSS URL input with sample suggestions
        st.markdown("**📰 Regular News RSS URLs:**")
        for url in sample_urls["news"]:
            if st.button(f"📋 {url.split('/')[-2] if '/' in url else url}", key=f"news_{url}"):
                st.session_state.news_rss_url = url
        
        news_url = st.text_area(
            "News/Twitter RSS Feed URL:", 
            value=st.session_state.news_rss_url,
            placeholder="Enter news RSS feed URL or Google News Twitter search URL...", 
            height=80,
            key="news_input"
        )
        
        if st.button("🔄 Fetch News Updates", key="fetch_news"):
            if news_url:
                with st.spinner("Fetching news updates..."):
                    news_items = fetch_rss_feed(news_url)
                    st.session_state.news_items = news_items
            else:
                st.warning("Please enter a news RSS URL first!")
        
        # Display News updates
        st.markdown("**Latest News Updates:**")
        if 'news_items' in st.session_state and st.session_state.news_items:
            for item in st.session_state.news_items:
                if "error" in item:
                    st.error(f"❌ {item['error']}")
                else:
                    with st.expander(f"📢 {item['title'][:80]}... ({item['published']})"):
                        st.write(f"**📅 Published:** {item['published']}")
                        st.write(f"**📝 Summary:** {item['summary']}")
                        st.markdown(f"**🔗 [Read Full Article]({item['link']})**")
        else:
            # Fallback sample updates
            news_updates = [
                "📢 Quezon City suspends classes in all levels - 3:00 PM",
                "🏫 Makati announces no class suspension despite heavy rains - 2:30 PM",
                "⚡ Several areas in Metro Manila experience power outages - 1:45 PM"
            ]
            for update in news_updates:
                st.markdown(f"• {update}")
            st.info("👆 These are sample updates. Try the Twitter search URLs above for real #WalangPasok data!")
    
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