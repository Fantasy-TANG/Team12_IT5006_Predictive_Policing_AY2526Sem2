import streamlit as st
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import calendar
import os
import pyarrow

st.set_page_config(page_title="Chicago Crime Dashboard", layout="wide", initial_sidebar_state="expanded")

current_dir = os.path.dirname(os.path.abspath(__file__))

data_path = os.path.join(current_dir, "..", "data", "Crimes_2014_2024_cleaned.parquet")

def load_data():
    return pd.read_parquet(data_path)

df = load_data()

with open(os.path.join(current_dir, "style.css")) as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

st.sidebar.markdown("<br>", unsafe_allow_html=True)
st.sidebar.markdown("<br>", unsafe_allow_html=True)
st.sidebar.markdown('#### Fundamentals of Data Analytics')
st.sidebar.markdown('#### IT5006      Group 12')

st.sidebar.markdown('''
---
''')
st.sidebar.markdown("&nbsp;" * 25 +'Contributors'+"&nbsp;" * 5)
st.sidebar.markdown("&nbsp;" * 15+'LI Mingyue' + "&nbsp;" * 5 +'LI Sitong')
st.sidebar.markdown("&nbsp;" * 15+'TANG Yun'+ "&nbsp;" * 7 +'WU Silin')


st.markdown('# Chicago Crime Dataset 2014-2024')

# Overview
violent_crimes = ['HOMICIDE', 'ROBBERY', 'BATTERY', 'CRIM SEXUAL ASSAULT', 'ASSAULT']

total_crimes = len(df)
arrest_rate = (df['Arrest'].sum() / total_crimes) * 100
violent_rate = (df['Primary Type'].isin(violent_crimes).sum() / total_crimes) * 100

st.header("A Decade Review")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Incidents", f"{total_crimes:,}")
with col2:
    st.metric("Arrest Rate %", f"{arrest_rate:.2f}%")
with col3:
    st.metric("Violent Crime %", f"{violent_rate:.2f}%")

st.caption('Violent crime refers to `HOMICIDE`, `ROBBERY`, `BATTERY`, `CRIM SEXUAL ASSAULT` and  `ASSAULT`')

c1, c2 = st.columns(2)
with c1:
    st.subheader("Yearly Crime Trend")

    yearly_counts = df.groupby('Year').size().reset_index(name='Count')

    fig_line = px.line(yearly_counts, x='Year', y='Count', 
                    markers=True, 
                    template="plotly_white",
                    color_discrete_sequence=['#0083B8'])
    fig_line.update_layout(
    xaxis_title="Year",
    yaxis_title="Number of Incidents",
    height=300, 
    margin=dict(l=20, r=20, t=20, b=20) 
)

    st.plotly_chart(fig_line, use_container_width=True)

with c2:
    st.subheader("Most Frequent Crime Types")

    crime_counts = df['Primary Type'].value_counts().to_dict()

    wc = WordCloud(width=800, height=400, 
                background_color='white', 
                colormap='viridis',
                max_font_size=150,
                max_words=30,
                random_state=100).generate_from_frequencies(crime_counts)

    fig_wc, ax = plt.subplots(figsize=(6,3))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')

    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)

    st.pyplot(fig_wc, use_container_width=True)


# Period Analysis
st.header('Dynamic Analysis: Specific Period Deep-Dive')

year_range = st.select_slider(
    label='Select year range',
    options=list(range(2014, 2025)),
    value=(2020, 2024)  # default
)

mask_year = (df['Year'] >= year_range[0]) & (df['Year'] <= year_range[1])
filtered_df = df.loc[mask_year]

st.subheader(f'Period Selection: `{year_range[0]}` - `{year_range[1]}`')
col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader(f"Monthly Crime Distribution")
    
    monthly_counts = filtered_df.groupby('Month').size().reset_index(name='Count')
    
    monthly_counts['Month Name'] = monthly_counts['Month'].apply(lambda x: calendar.month_name[x])
    
    fig_month = px.bar(
        monthly_counts, 
        x='Month Name', 
        y='Count',
        text='Count',
        color='Count',
        color_continuous_scale='GnBu', 
        height=400
    )
    
    fig_month.update_layout(
        xaxis_title="Month",
        yaxis_title="Number of Incidents",
        xaxis={'categoryorder':'array', 'categoryarray': list(calendar.month_name)[1:]},
        showlegend=False
    )
    
    st.plotly_chart(fig_month, use_container_width=True)

with col_right:
    st.subheader("Arrest Rate")
    
    arrest_counts = filtered_df['Arrest'].value_counts().reset_index()
    arrest_counts.columns = ['Status', 'Count']
    arrest_counts['Status'] = arrest_counts['Status'].replace({True: 'Arrested', False: 'Not Arrested'})
    
    period_total = len(filtered_df)
    
    fig_donut = px.pie(
        arrest_counts, 
        values='Count', 
        names='Status',
        hole=0.6, 
        color_discrete_map={'Arrested': '#2E8B57', 'Not Arrested': '#FF6347'}
    )
    
    fig_donut.add_annotation(
        text=f"Period Total<br><b>{period_total:,}</b>",
        showarrow=False,
        font_size=16
    )
    
    fig_donut.update_layout(
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
    )
    
    st.plotly_chart(fig_donut, use_container_width=True)


# Explore by crime type
st.subheader('Behavioral Patterns by Crime Type')

top_15_types = filtered_df['Primary Type'].value_counts().nlargest(15).index.tolist()

filtered_df['Primary Type'] = filtered_df['Primary Type'].astype(str)
filtered_df.loc[~filtered_df['Primary Type'].isin(top_15_types), 'Primary Type'] = 'OTHERS'
filtered_df['Primary Type'] = filtered_df['Primary Type'].astype('category')

available_types = sorted([t for t in filtered_df['Primary Type'].unique() if t != 'OTHERS'])
available_types.append('OTHERS')

selected_type = st.selectbox(
    f"Select a crime type for analysis in {year_range[0]}-{year_range[1]}:",
    available_types
)

type_df = filtered_df[filtered_df['Primary Type'] == selected_type].copy()

col_map, col_radial = st.columns([3,2])
with col_map:
    st.subheader(f" {selected_type} Space Distribution")

    map_df=type_df
    map_df['color'] = map_df['Arrest'].apply(lambda x: [46, 139, 87, 255] if x else [220, 20, 60, 80])

    st.map(
        map_df,
        latitude='Latitude',
        longitude='Longitude',
        color='color',
        size=1,
        zoom=9,
        use_container_width=True
    )
    st.caption('Spots in : GREEN - Arrested, RED - Not Arrested')

with col_radial:
    hour_counts = type_df.groupby('Hour').size().reset_index(name='Count')

    all_hours = pd.DataFrame({'Hour': range(24)})
    hour_counts = all_hours.merge(hour_counts, on='Hour', how='left').fillna(0)

    st.subheader(f"{selected_type} Crime Clock")

    fig_radial = px.bar_polar(
        hour_counts,
        r='Count',
        theta=hour_counts['Hour'] * 15,
        color='Count',
        color_continuous_scale='GnBu',
        template="plotly_white",
    )

    fig_radial.update_polars(
        angularaxis=dict(
            tickvals=[i * 15 for i in range(24)],
            ticktext=[f"{i}:00" for i in range(24)],
            direction="clockwise",
            period=360
        )
    )
    fig_radial.update_layout(height=480,
                             margin=dict(l=30, r=30, t=30, b=0),
                             coloraxis_showscale=False)
    st.plotly_chart(fig_radial, use_container_width=True)

# Explore by district
st.subheader('Explore by District')
col_heat_map, col_arrest = st.columns([3, 2])
with col_heat_map:
    st.subheader('Crime-District Heatmap')
    top_5_crimes = filtered_df['Primary Type'].value_counts().nlargest(5).index.tolist()
    heatmap_data = filtered_df[filtered_df['Primary Type'].isin(top_5_crimes)]

    heatmap_matrix = heatmap_data.pivot_table(
        index='Primary Type',
        columns='District',
        values='ID',
        aggfunc='count',
        fill_value=0
    )
    heatmap_matrix = heatmap_matrix.loc[top_5_crimes]

    fig_heatmap = px.imshow(
        heatmap_matrix,
        labels=dict(x="District"),
        x=heatmap_matrix.columns,
        y=heatmap_matrix.index,
        color_continuous_scale='Blues',
        aspect="auto"
    )

    fig_heatmap.update_layout(
        height=300,
        margin=dict(l=0, r=0, t=30, b=0),
        xaxis_nticks=len(heatmap_matrix.columns),
        coloraxis_showscale=False
    )
    fig_heatmap.update_xaxes(type='category')
    fig_heatmap.update_yaxes(showticklabels=True, title=None)
    st.plotly_chart(fig_heatmap, use_container_width=True)

with col_arrest:
    st.subheader("District Arrest Rate %")
    arrest_stats = filtered_df.groupby('District')['Arrest'].mean().reset_index()
    arrest_stats.columns = ['District', 'Arrest_Rate']
    arrest_stats['Arrest_Rate'] = arrest_stats['Arrest_Rate'] * 100

    arrest_stats = arrest_stats.sort_values('Arrest_Rate', ascending=False)

    fig_arrest = px.bar(
        arrest_stats,
        x='District',
        y='Arrest_Rate',
        color='Arrest_Rate',
        color_continuous_scale='Blues',
        labels={'Arrest_Rate': 'Arrest Rate %', 'District': 'District'}
    )

    fig_arrest.update_layout(
        height=300,
        coloraxis_showscale=False,
        margin=dict(t=30, b=0),
        xaxis_title="District",
        yaxis_title="Arrest Rate %"
    )

    fig_arrest.update_xaxes(type='category')
    st.plotly_chart(fig_arrest, use_container_width=True)

col_district_map,col_district_top=st.columns([2, 1])
with col_district_map:
    st.subheader("District Distribution")
    district_summary = filtered_df.groupby('District').agg(
        lat=('Latitude', 'mean'),
        lon=('Longitude', 'mean'),
        total_incidents=('ID', 'count')
    ).reset_index()

    selected_district_id=st.selectbox(label="Select District ID", options=district_summary['District'].unique())

    district_summary['color'] = district_summary['District'].apply(
        lambda x: "#FF8C00" if x == selected_district_id else "#1E90FF"
    )

    district_summary['radius_size'] = district_summary['total_incidents'] * 0.02

    st.map(
        district_summary,
        latitude='lat',
        longitude='lon',
        size='radius_size',
        color='color',
        zoom=10,
        use_container_width=True
    )

    st.caption("The circle in ORANGE is currently selected. The larger it is, the more crimes there are.")

with col_district_top:
    dist_detail_df = filtered_df[filtered_df['District'] == selected_district_id]

    dist_total_crimes = len(dist_detail_df)

    dist_top_5 = dist_detail_df['Primary Type'].value_counts().head(5).reset_index()
    dist_top_5.columns = ['Crime Type', 'Number of Incidents']

    st.subheader(f"District {selected_district_id} Metrics")
    st.metric(
        label=f"Total Incidents in {year_range[0]}-{year_range[1]}",
        value=f"{dist_total_crimes}"
    )

    st.write(f"**Top 5 Crime Types in {year_range[0]}-{year_range[1]}:**")

    for i, row in dist_top_5.iterrows():
        percentage = (row['Number of Incidents'] / dist_total_crimes)
        st.write(f"**{row['Crime Type']}** ({row['Number of Incidents']})")
        st.progress(percentage)

    st.caption("Showing percentage of total")