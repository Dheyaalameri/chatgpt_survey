"""
ChatGPT Araştırma Panosu - Minimalist Professional UI
Üniversite Öğrencilerinin ChatGPT'ye Güven Analizi
Dataset: chatgpt_survey_son.csv (N=152)
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import numpy as np

# ==================== SAYFA AYARLARI ====================
st.set_page_config(
    page_title="ChatGPT Güven Analizi",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== DARK MODE ONLY CSS ====================
def get_dark_mode_css():
    return """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
            
            * { font-family: 'Inter', sans-serif !important; }
            
            .stApp {
                background: #0E1117;
                color: #E8E8E8;
            }
            
            h1, h2, h3 {
                font-family: 'Inter', sans-serif !important;
                font-weight: 600;
                letter-spacing: -0.5px;
                color: #FFFFFF;
            }
            
            h1 {
                font-size: 2.5rem !important;
                text-align: center;
                padding: 2rem 0 1rem 0;
                border-bottom: 1px solid rgba(255, 255, 255, 0.1);
                margin-bottom: 2rem;
            }
            
            h2 {
                font-size: 1.75rem !important;
                color: #F5F5F5;
                margin-top: 2rem;
                font-weight: 500;
            }
            
            h3 {
                font-size: 1.25rem !important;
                color: #E0E0E0;
                font-weight: 500;
            }
            
            [data-testid="stMetricValue"] {
                font-size: 2.25rem;
                color: #84CC16;
                font-weight: 600;
            }
            
            [data-testid="stMetricLabel"] {
                color: #A0A0A0;
                font-size: 0.875rem;
                font-weight: 500;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }
            
            [data-testid="stMetric"] {
                background: rgba(255, 255, 255, 0.03);
                backdrop-filter: blur(10px);
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 12px;
                padding: 1.5rem;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            }
            
            [data-testid="stSidebar"] {
                background: rgba(20, 25, 35, 0.95);
                backdrop-filter: blur(10px);
                border-right: 1px solid rgba(255, 255, 255, 0.08);
            }
            
            .stTabs [data-baseweb="tab-list"] {
                gap: 0px;
                background: rgba(255, 255, 255, 0.03);
                border-radius: 10px;
                padding: 4px;
            }
            
            .stTabs [data-baseweb="tab"] {
                background: transparent;
                border: none;
                color: #A0A0A0;
                padding: 12px 24px;
                font-weight: 500;
                font-size: 0.925rem;
                border-radius: 8px;
                transition: all 0.3s ease;
            }
            
            .stTabs [data-baseweb="tab"]:hover {
                background: rgba(255, 255, 255, 0.05);
                color: #FFFFFF;
            }
            
            .stTabs [aria-selected="true"] {
                background: rgba(132, 204, 22, 0.15);
                color: #84CC16;
                font-weight: 600;
            }
            
            .stAlert {
                background: rgba(255, 255, 255, 0.03);
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 10px;
                color: #E8E8E8;
                backdrop-filter: blur(10px);
            }
            
            .streamlit-expanderHeader {
                background: rgba(255, 255, 255, 0.03);
                border: 1px solid rgba(255, 255, 255, 0.08);
                border-radius: 8px;
                color: #FFFFFF;
                font-weight: 500;
            }
            
            .stSelectbox label, .stMultiSelect label {
                color: #A0A0A0;
                font-weight: 500;
                font-size: 0.875rem;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }
            
            [data-testid="stDataFrame"] {
                border-radius: 10px;
                overflow: hidden;
            }
            
            p {
                color: #C0C0C0;
                line-height: 1.7;
                font-size: 0.95rem;
            }
            
            hr {
                border: 0;
                height: 1px;
                background: rgba(255, 255, 255, 0.08);
                margin: 2rem 0;
            }
            
            .team-card {
                background: rgba(255, 255, 255, 0.03);
                backdrop-filter: blur(10px);
                border: 1px solid rgba(255, 255, 255, 0.08);
                border-radius: 12px;
                padding: 2rem;
                text-align: center;
                margin: 1rem 0;
                transition: all 0.3s ease;
            }
            
            .team-card:hover {
                background: rgba(255, 255, 255, 0.05);
                border-color: rgba(132, 204, 22, 0.3);
                transform: translateY(-2px);
            }
            
            .team-card h3 {
                color: #FFFFFF;
                margin-bottom: 0.5rem;
                font-weight: 600;
            }
            
            .team-card p {
                color: #84CC16;
                font-style: normal;
                font-size: 0.875rem;
                margin: 0;
            }
        </style>
        """

st.markdown(get_dark_mode_css(), unsafe_allow_html=True)

# ==================== DATA LOADING ====================
@st.cache_data
def load_data():
    df = pd.read_csv('chatgpt_survey_son.csv', encoding='utf-8')
    
    # Trust columns (indices 11-16)
    trust_column_indices = [11, 12, 13, 14, 15, 16]
    trust_columns = df.columns[trust_column_indices].tolist()
    
    trust_labels = {
        trust_columns[0]: 'Genel Güvenilirlik',
        trust_columns[1]: 'Akademik Doğruluk',
        trust_columns[2]: 'Ödevde Kullanım',
        trust_columns[3]: 'Açıklama Doğruluğu',
        trust_columns[4]: 'Karmaşık Konular',
        trust_columns[5]: 'Güvenilir Kaynak'
    }
    
    # Calculate Trust Score
    df['Trust_Score'] = df.iloc[:, trust_column_indices].mean(axis=1)
    
    # Academic Usage Intensity (sum of 4 binary columns)
    # Columns 6-9: Odev_Proje, Konu_Tekrari, Kod_Yazma, Metin_Cevirisi
    usage_cols = [6, 7, 8, 9]
    df['Academic_Usage_Intensity'] = df.iloc[:, usage_cols].sum(axis=1)
    
    # GNO mapping
    gno_mapping = {
        '0-2.00': 1.5,
        '2.01-2.50': 2.25,
        '2.51-3.00': 2.75,
        '3.01-4.00': 3.5
    }
    df['GNO_Numerik'] = df['GNO'].map(gno_mapping)
    
    # Sınıf mapping
    df['Sinif_Display'] = df['Sinif'].apply(lambda x: 'Hazırlık' if x == 0 else str(int(x)))
    
    return df, trust_columns, trust_labels

df, trust_columns, trust_labels = load_data()

# ==================== SIDEBAR ====================
st.sidebar.markdown("### ChatGPT Güven Analizi")
st.sidebar.markdown("---")

# Toplam Katılımcı
st.sidebar.metric(label="Toplam Katılımcı", value="152", delta="N")

st.sidebar.markdown("---")
st.sidebar.markdown("**Filtreler**")

# Cinsiyet Filter
cinsiyet_options = df['Cinsiyet'].dropna().unique().tolist()
cinsiyet_filter = st.sidebar.multiselect(
    "Cinsiyet",
    options=cinsiyet_options,
    default=cinsiyet_options
)

# Sınıf Filter
sinif_options = sorted(df['Sinif_Display'].dropna().unique().tolist(), 
                       key=lambda x: (x != 'Hazırlık', x))
sinif_filter = st.sidebar.multiselect(
    "Sınıf",
    options=sinif_options,
    default=sinif_options
)

# GNO Filter
gno_options = sorted(df['GNO'].dropna().unique().tolist())
gno_filter = st.sidebar.multiselect(
    "GNO Aralığı",
    options=gno_options,
    default=gno_options
)

# Apply filters
try:
    df_filtered = df[
        (df['Cinsiyet'].isin(cinsiyet_filter)) &
        (df['Sinif_Display'].isin(sinif_filter)) &
        (df['GNO'].isin(gno_filter))
    ].copy()
    
    if len(df_filtered) == 0:
        st.sidebar.warning("Filtre kombinasyonu sonuç vermiyor")
        df_filtered = df.copy()
except Exception as e:
    st.sidebar.error(f"Filtre hatası: {str(e)}")
    df_filtered = df.copy()

st.sidebar.info(f"**Seçili:** {len(df_filtered)} katılımcı")

st.sidebar.markdown("---")
st.sidebar.markdown("""
**Veri Seti Bilgileri**
- Kaynak: chatgpt_survey_son.csv
- Ölçek: Likert 5-nokta
- Hipotez: 4 test
""")

# ==================== MAIN TITLE ====================
st.markdown("<h1>ChatGPT Güven Analizi</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #A0A0A0; margin-bottom: 2rem;'>Üniversite Öğrencilerinin ChatGPT'ye Güven Düzeyleri ve Kullanım Alışkanlıkları</p>", unsafe_allow_html=True)

# ==================== TAB STRUCTURE ====================
tab1, tab2, tab3, tab4 = st.tabs([
    "Genel Bilgiler", 
    "Güven Ölçeği", 
    "Hipotez Testleri",
    "Hakkımızda"
])

# ==================== TAB 1: GENEL BİLGİLER ====================
with tab1:
    st.markdown("## Demografik ve Genel İstatistikler")
    
    # Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Örneklem Büyüklüğü",
            value=f"{len(df_filtered)}",
            delta="Katılımcı"
        )
    
    with col2:
        avg_trust = df_filtered['Trust_Score'].mean()
        st.metric(
            label="Ortalama Güven Skoru",
            value=f"{avg_trust:.2f}",
            delta="/ 5.00"
        )
    
    with col3:
        male_count = (df_filtered['Cinsiyet'] == 'Erkek').sum()
        female_count = (df_filtered['Cinsiyet'] == 'Kadın').sum()
        st.metric(
            label="Cinsiyet Dağılımı",
            value=f"{male_count}E / {female_count}K",
            delta=f"%{(male_count/len(df_filtered)*100):.0f} Erkek" if len(df_filtered) > 0 else "0%"
        )
    
    with col4:
        avg_gno = df_filtered['GNO_Numerik'].mean()
        st.metric(
            label="Ortalama GNO",
            value=f"{avg_gno:.2f}",
            delta="/ 4.00"
        )
    
    st.markdown("---")
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Cinsiyet Dağılımı")
        gender_counts = df_filtered['Cinsiyet'].value_counts()
        
        # Electric Blue & Deep Coral
        color_map = {'Erkek': '#1E90FF', 'Kadın': '#FF4B4B'}
        
        fig_gender = px.pie(
            values=gender_counts.values,
            names=gender_counts.index,
            title='',
            color=gender_counts.index,
            color_discrete_map=color_map,
            hole=0.5
        )
        fig_gender.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#E8E8E8', family='Inter'),
            showlegend=True,
            height=350,
            margin=dict(t=20, b=20, l=20, r=20)
        )
        fig_gender.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_gender, use_container_width=True)
    
    with col2:
        st.markdown("### GNO Dağılımı")
        gno_counts = df_filtered['GNO'].value_counts().sort_index()
        fig_gno = px.bar(
            x=gno_counts.index,
            y=gno_counts.values,
            title='',
            labels={'x': 'GNO Aralığı', 'y': 'Katılımcı Sayısı'},
            color=gno_counts.values,
            color_continuous_scale=[[0, '#84CC16'], [1, '#22C55E']]
        )
        fig_gno.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#E8E8E8', family='Inter'),
            showlegend=False,
            height=350,
            margin=dict(t=20, b=20, l=20, r=20)
        )
        st.plotly_chart(fig_gno, use_container_width=True)
    
    st.markdown("---")
    
    # Summary Statistics
    st.markdown("### Özet İstatistikler")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Yaş Dağılımı**")
        if len(df_filtered) > 0:
            age_stats = df_filtered['Yas'].describe()
            st.dataframe({
                'İstatistik': ['Ortalama', 'Std. Sapma', 'Min', 'Max'],
                'Değer': [f"{age_stats['mean']:.1f}", f"{age_stats['std']:.1f}", 
                         f"{age_stats['min']:.0f}", f"{age_stats['max']:.0f}"]
            }, hide_index=True, use_container_width=True)
    
    with col2:
        st.markdown("**Kullanım Sıklığı**")
        freq_counts = df_filtered['ChatGPT_Kullanim_Sikligi'].value_counts()
        st.dataframe({
            'Sıklık': freq_counts.index[:5],
            'Katılımcı': freq_counts.values[:5]
        }, hide_index=True, use_container_width=True)

# ==================== TAB 2: GÜVEN ÖLÇEĞİ ====================
with tab2:
    st.markdown("## Güven Ölçeği - Likert 5 Nokta Analiz")
    st.markdown("*1 = Kesinlikle Katılmıyorum | 5 = Kesinlikle Katılıyorum*")
    
    # Calculate Likert distribution
    likert_data = []
    for col, label in trust_labels.items():
        for score in range(1, 6):
            count = (df_filtered[col] == score).sum()
            percentage = (count / len(df_filtered)) * 100 if len(df_filtered) > 0 else 0
            likert_data.append({
                'Soru': label,
                'Puan': score,
                'Sayı': count,
                'Yüzde': percentage
            })
    
    likert_df = pd.DataFrame(likert_data)
    
    # Stacked bar chart
    fig_likert = px.bar(
        likert_df,
        x='Yüzde',
        y='Soru',
        color='Puan',
        orientation='h',
        title='',
        labels={'Yüzde': 'Yüzde (%)', 'Soru': '', 'Puan': 'Likert Puanı'},
        color_continuous_scale=[[0, '#EF4444'], [0.25, '#F59E0B'], [0.5, '#EAB308'], [0.75, '#84CC16'], [1, '#22C55E']],
        barmode='stack'
    )
    
    fig_likert.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#E8E8E8', size=11, family='Inter'),
        height=500,
        xaxis=dict(range=[0, 100]),
        margin=dict(t=20, b=20, l=20, r=20)
    )
    
    st.plotly_chart(fig_likert, use_container_width=True)
    
    st.markdown("---")
    
    # Statistics Table
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### Soru Bazında Ortalama Skorlar")
        question_stats = []
        for col, label in trust_labels.items():
            question_stats.append({
                'Güven Boyutu': label,
                'Ortalama': round(df_filtered[col].mean(), 2),
                'Std. Sapma': round(df_filtered[col].std(), 2)
            })
        
        stats_df = pd.DataFrame(question_stats).sort_values('Ortalama', ascending=False)
        st.dataframe(stats_df, use_container_width=True, hide_index=True)
    
    with col2:
        st.markdown("### En Yüksek/Düşük")
        if len(stats_df) > 0:
            st.success(f"**En Yüksek**\n\n{stats_df.iloc[0]['Güven Boyutu']}\n\n{stats_df.iloc[0]['Ortalama']}/5.00")
            st.error(f"**En Düşük**\n\n{stats_df.iloc[-1]['Güven Boyutu']}\n\n{stats_df.iloc[-1]['Ortalama']}/5.00")

# ==================== TAB 3: HİPOTEZ TESTLERİ ====================
with tab3:
    st.markdown("## Hipotez Testleri ve İstatistiksel Analiz")
    st.markdown("*Alpha Seviyesi: α = 0.05*")
    
    # H1
    st.markdown("---")
    st.markdown("### H₁: Kullanım Sıklığı ve Güven Skoru İlişkisi")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.info("""
        **Hipotez**
        
        Farklı kullanım sıklığı grupları arasında güven skorunda anlamlı fark vardır.
        
        **Test:** Kruskal-Wallis
        """)
        
        freq_col = 'ChatGPT_Kullanim_Sikligi'
        groups = []
        for category in df_filtered[freq_col].dropna().unique():
            group_data = df_filtered[df_filtered[freq_col] == category]['Trust_Score'].dropna()
            if len(group_data) > 0:
                groups.append(group_data)
        
        if len(groups) >= 2:
            h_stat, p_value = stats.kruskal(*groups)
            
            st.metric("H İstatistiği", f"{h_stat:.4f}")
            st.metric("P-değeri", f"{p_value:.4f}")
            
            if p_value < 0.05:
                st.success("Hipotez Kabul Edildi")
            else:
                st.error("Hipotez Reddedildi")
    
    with col2:
        if len(groups) >= 2:
            fig_h1 = px.box(
                df_filtered,
                x=freq_col,
                y='Trust_Score',
                color=freq_col,
                title='',
                labels={freq_col: 'Kullanım Sıklığı', 'Trust_Score': 'Güven Skoru'},
                color_discrete_sequence=['#84CC16', '#22C55E', '#10B981', '#059669', '#047857']
            )
            fig_h1.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#E8E8E8', family='Inter'),
                showlegend=False,
                height=350,
                margin=dict(t=20, b=20, l=20, r=20)
            )
            st.plotly_chart(fig_h1, use_container_width=True)
    
    # H2 - NEW: Academic Usage Intensity
    st.markdown("---")
    st.markdown("### H₂: Akademik Kullanım Yoğunluğu ve Güven İlişkisi")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.info("""
        **Hipotez**
        
        ChatGPT'yi akademik amaçlarla daha yoğun kullanan öğrencilerin güven düzeyleri, daha düşük yoğunlukta kullananlara göre anlamlı düzeyde daha yüksektir.
        
        **Test:** Mann-Whitney U
        """)
        
        # Define intensity threshold (median)
        median_intensity = df_filtered['Academic_Usage_Intensity'].median()
        high_intensity = df_filtered[df_filtered['Academic_Usage_Intensity'] > median_intensity]['Trust_Score'].dropna()
        low_intensity = df_filtered[df_filtered['Academic_Usage_Intensity'] <= median_intensity]['Trust_Score'].dropna()
        
        if len(high_intensity) > 0 and len(low_intensity) > 0:
            u_stat, p_value = stats.mannwhitneyu(high_intensity, low_intensity, alternative='greater')
            
            st.metric("U İstatistiği", f"{u_stat:.2f}")
            st.metric("P-değeri", f"{p_value:.4f}")
            st.metric("Medyan Yoğunluk", f"{median_intensity:.1f}")
            
            if p_value < 0.05:
                st.success("Hipotez Kabul Edildi")
            else:
                st.error("Hipotez Reddedildi")
    
    with col2:
        if len(high_intensity) > 0 and len(low_intensity) > 0:
            # Create intensity categories
            df_filtered['Intensity_Category'] = df_filtered['Academic_Usage_Intensity'].apply(
                lambda x: 'Yüksek Yoğunluk' if x > median_intensity else 'Düşük Yoğunluk'
            )
            
            fig_h2 = px.violin(
                df_filtered,
                x='Intensity_Category',
                y='Trust_Score',
                color='Intensity_Category',
                box=True,
                title='',
                labels={'Intensity_Category': 'Akademik Kullanım Yoğunluğu', 'Trust_Score': 'Güven Skoru'},
                color_discrete_map={'Yüksek Yoğunluk': '#84CC16', 'Düşük Yoğunluk': '#EAB308'}
            )
            fig_h2.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#E8E8E8', family='Inter'),
                showlegend=False,
                height=350,
                margin=dict(t=20, b=20, l=20, r=20)
            )
            st.plotly_chart(fig_h2, use_container_width=True)
    
    # H3
    st.markdown("---")
    st.markdown("### H₃: Cinsiyet ve Güven Skoru Farkı")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.info("""
        **Hipotez**
        
        Erkek ve kadın öğrenciler arasında güven skorunda anlamlı fark vardır.
        
        **Test:** Mann-Whitney U
        """)
        
        male_trust = df_filtered[df_filtered['Cinsiyet'] == 'Erkek']['Trust_Score'].dropna()
        female_trust = df_filtered[df_filtered['Cinsiyet'] == 'Kadın']['Trust_Score'].dropna()
        
        if len(male_trust) > 0 and len(female_trust) > 0:
            u_stat, p_value = stats.mannwhitneyu(male_trust, female_trust, alternative='two-sided')
            
            st.metric("U İstatistiği", f"{u_stat:.2f}")
            st.metric("P-değeri", f"{p_value:.4f}")
            
            if p_value < 0.05:
                st.success("Hipotez Kabul Edildi")
            else:
                st.error("Hipotez Reddedildi")
    
    with col2:
        if len(male_trust) > 0 and len(female_trust) > 0:
            fig_h3 = px.violin(
                df_filtered,
                x='Cinsiyet',
                y='Trust_Score',
                color='Cinsiyet',
                box=True,
                title='',
                labels={'Cinsiyet': 'Cinsiyet', 'Trust_Score': 'Güven Skoru'},
                color_discrete_map={'Erkek': '#1E90FF', 'Kadın': '#FF4B4B'}
            )
            fig_h3.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#E8E8E8', family='Inter'),
                showlegend=False,
                height=350,
                margin=dict(t=20, b=20, l=20, r=20)
            )
            st.plotly_chart(fig_h3, use_container_width=True)
    
    # H4
    st.markdown("---")
    st.markdown("### H₄: GNO ve Güven Skoru İlişkisi")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.info("""
        **Hipotez**
        
        Akademik başarı (GNO) ile güven skoru arasında korelasyon vardır.
        
        **Test:** Spearman Korelasyon
        """)
        
        gno_numeric = df_filtered['GNO_Numerik'].dropna()
        trust_score = df_filtered['Trust_Score'].dropna()
        common_idx = gno_numeric.index.intersection(trust_score.index)
        
        if len(common_idx) > 3:
            corr, p_value = stats.spearmanr(gno_numeric[common_idx], trust_score[common_idx])
            
            st.metric("Spearman ρ", f"{corr:.4f}")
            st.metric("P-değeri", f"{p_value:.4f}")
            
            if p_value < 0.05:
                st.success("Hipotez Kabul Edildi")
            else:
                st.error("Hipotez Reddedildi")
    
    with col2:
        if len(common_idx) > 3:
            fig_h4 = px.scatter(
                x=gno_numeric[common_idx],
                y=trust_score[common_idx],
                title='',
                labels={'x': 'GNO (Sayısal)', 'y': 'Güven Skoru'},
                trendline='ols',
                opacity=0.6
            )
            fig_h4.update_traces(marker=dict(color='#84CC16', size=8))
            fig_h4.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#E8E8E8', family='Inter'),
                height=350,
                margin=dict(t=20, b=20, l=20, r=20)
            )
            st.plotly_chart(fig_h4, use_container_width=True)

# ==================== TAB 4: HAKKIMIZDA ====================
with tab4:
    st.markdown("## Araştırma Ekibi")
    st.markdown("*ChatGPT Güven Analizi Projesi - Araştırma Yöntemleri*")
    
    st.markdown("---")
    
    # Team Members (Exactly 6 members in 3x2 grid)
    team_members = [
        {"name": "Dheya Aldain Alameri", "role": "Veri toplama, analiz "},
        {"name": "Bassam Alharogi", "role": "Rapor yazımı"},
        {"name": "Moayed Nagi", "role": "Veri toplama ve analiz"},
        {"name": "Al-Hasan Ba Said", "role": "Sunum hazırlama"},
        {"name": "Musaab Salah", "role": "Rapor yazımı"},
        {"name": "Edres Qasem", "role": "Sunum hazırlama"}
    ]
    
    st.markdown("### Ekip Üyeleri")
    
    # 3x2 Grid layout
    for i in range(0, 6, 3):
        cols = st.columns(3)
        for j in range(3):
            if i + j < len(team_members):
                member = team_members[i + j]
                with cols[j]:
                    st.markdown(f"""
                    <div class='team-card'>
                        <h3>{member['name']}</h3>
                        <p>{member['role']}</p>
                    </div>
                    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Project Info
    st.markdown("### Proje Hakkında")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Araştırma Konusu**
        
        Bu çalışma, üniversite öğrencilerinin ChatGPT'ye olan güven düzeylerini ve 
        kullanım alışkanlıklarını incelemektedir.
        
        **Metodoloji**
        - Likert 5-nokta ölçek
        - 152 katılımcı
        - 4 hipotez testi
        - SciPy istatistiksel analiz
        """)
    
    with col2:
        st.markdown("""
        **Kullanılan Teknolojiler**
        - Python
        - Streamlit
        - Plotly
        - SciPy
        - Pandas
        
        **Ders**
        
        Araştırma Yöntemleri
        """)

# ==================== FOOTER ====================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #606060; padding: 1rem 0;'>
    <p>Üniversite Araştırma Projesi | ChatGPT Güven Analizi</p>
    <p style='font-size: 0.85rem;'>Dataset: chatgpt_survey_son.csv | N=152</p>
</div>
""", unsafe_allow_html=True)
