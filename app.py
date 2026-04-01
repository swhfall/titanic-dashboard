import streamlit as st
import pandas as pd
import plotly.express as px
st.set_page_config(page_title="Титаник дашборд", layout="wide") 
st.title("анализ данных по титанику")
@st.cache_data  #декоратор для кэша
def load_data():
    df = pd.read_csv("titanic.csv")
    return df
df = load_data()

st.header("1. Описательная статистика")
col1, col2 = st.columns(2)
with col1:
    column_info = pd.DataFrame({
        'Колонка': df.columns,
        'Тип данных': [str(dtype) for dtype in df.dtypes.values],
    })
    format_row = pd.DataFrame({
        'Колонка': ['Формат таблицы'],
        'Тип данных': [f'{df.shape[0]} строк × {df.shape[1]} столбцов']
    })
    final_info = pd.concat([format_row, column_info], ignore_index=True)
    st.dataframe(final_info, use_container_width=True, hide_index=True)
st.header("просмотр данных")
n = st.slider("выберите кол-во строк для отображения", min_value=5, max_value=50, value=10, step=5)
st.dataframe(df.head(n), use_container_width=True)
st.divider()
st.header("Визуализация данных")

# График 1 выживаемость
st.subheader("1. Выживаемость пассажиров")
survived_counts = df['Survived'].value_counts()
fig1 = px.pie(
    values=survived_counts.values, 
    names=['Погиб', 'Выжил'],
    title='Соотношение выживших и погибших',
    color_discrete_sequence=['#42aaff', '#d3deed']
)
st.plotly_chart(fig1, use_container_width=True)
    
# График 2 возраст vs класс
st.subheader("2. Тепловая карта: Возраст по классам")
age_pivot = df.pivot_table(
    values='PassengerId', 
    index=pd.cut(df['Age'], bins=10), 
    columns='Pclass', 
    aggfunc='count',
    fill_value=0
)
age_pivot.index = [f"{int(i.left)}-{int(i.right)}" for i in age_pivot.index]
fig2 = px.imshow(
    age_pivot,
    text_auto=True,
    aspect='auto',
    title='Тепловая карта: Количество пассажиров по возрастам и классам',
    labels=dict(x='Класс каюты', y='Возрастная группа', color='Количество'),
    color_continuous_scale='Viridis'
)
fig2.update_layout(
    title_font_size=18,
    title_x=0.5,
    height=500
)
st.plotly_chart(fig2, use_container_width=True)

# График 3 выжившие vs погибшие по классам
st.subheader("3. Выжившие и погибшие по классам")
class_survival = df.groupby(['Pclass', 'Survived']).size().reset_index(name='count')
class_survival['Survived'] = class_survival['Survived'].map({0: 'Погиб', 1: 'Выжил'})
class_survival['Pclass'] = class_survival['Pclass'].astype(str) + ' класс'
fig3 = px.bar(
    class_survival,
    x='Pclass',
    y='count',
    color='Survived',
    title='Количество выживших и погибших по классам',
    labels={'Pclass': 'Класс каюты', 'count': 'Количество пассажиров', 'Survived': 'Статус'},
    barmode='group',
    text='count',
    color_discrete_map={'Погиб': '#42aaff', 'Выжил': '#ffaacc'}
)
fig3.update_traces(textposition='outside')
fig3.update_layout(
    title_font_size=18,
    title_x=0.5
)
st.plotly_chart(fig3, use_container_width=True)

# График 4 средняя стоимость билета по классам
st.subheader("4. Линейная диаграмма: Средняя стоимость билета")
avg_fare = df.groupby('Pclass')['Fare'].mean().reset_index()
avg_fare.columns = ['Класс', 'Средняя стоимость']
avg_fare['Класс'] = avg_fare['Класс'].astype(str) + ' класс'
fig4 = px.line(
    avg_fare,
    x='Класс',
    y='Средняя стоимость',
    title='Средняя стоимость билета по классам',
    labels={'Средняя стоимость': 'Средняя цена ($)', 'Класс': ''},
    markers=True,
    line_shape='linear'
)
fig4.update_traces(
    marker=dict(size=12, color='#42aaff'),
    line=dict(color='#e6e6fa', width=3)
)
fig4.update_layout(
    title_font_size=18,
    title_x=0.5,
    yaxis_range=[0, avg_fare['Средняя стоимость'].max() + 20]
)
st.plotly_chart(fig4, use_container_width=True)

# График 5 Иерархическая диаграмма
st.subheader("5. Иерархия: класс, пол, выживаемость")
df_hierarchy = df.copy()
df_hierarchy['Статус'] = df_hierarchy['Survived'].map({0: 'Погиб', 1: 'Выжил'})
df_hierarchy['Пол'] = df_hierarchy['Sex'].map({'male': 'Мужской', 'female': 'Женский'})
df_hierarchy['Класс'] = df_hierarchy['Pclass'].astype(str) + ' класс'
fig5 = px.sunburst(
    df_hierarchy,
    path=['Класс', 'Пол', 'Статус'],
    title='Иерархическая диаграмма: класс-пол-выживаемость',
    color='Survived',
    color_continuous_scale='Viridis',
    width=800,
    height=600
)
fig5.update_layout(title_font_size=18, title_x=0.5)
st.plotly_chart(fig5, use_container_width=True)

st.header("Интерактивный график")
with st.sidebar:
    st.header("Фильтры для интерактивного графика")
    
    класс = st.multiselect(
        "Класс каюты:",
        options=sorted(df['Pclass'].unique()),
        default=sorted(df['Pclass'].unique())
    )
    пол = st.multiselect(
        "Пол:",
        options=df['Sex'].unique(),
        default=df['Sex'].unique()
    )
    возраст_от = st.slider(
        "Возраст от:",
        min_value=int(df['Age'].min()),
        max_value=int(df['Age'].max()),
        value=0
    ) 
    возраст_до = st.slider(
        "Возраст до:",
        min_value=int(df['Age'].min()),
        max_value=int(df['Age'].max()),
        value=int(df['Age'].max())
    )
отфильтрованные = df[
    (df['Pclass'].isin(класс)) & 
    (df['Sex'].isin(пол)) &
    (df['Age'] >= возраст_от) &
    (df['Age'] <= возраст_до)
]
fig_интерактивный = px.scatter(
    отфильтрованные,
    x='Age',
    y='Fare',
    color='Survived',
    title=f"Возраст vs Стоимость билета (показано {len(отфильтрованные)} пассажиров)",
    labels={'Age': 'Возраст (лет)', 'Fare': 'Стоимость ($)', 'Survived': 'Выжил'},
    color_discrete_map={0: 'red', 1: 'green'}
)
st.plotly_chart(fig_интерактивный, use_container_width=True)
st.write(f"**По выбранным фильтрам:** {len(отфильтрованные)} пассажиров")
st.write(f"**Выжило:** {отфильтрованные['Survived'].sum()} ({отфильтрованные['Survived'].mean()*100:.1f}%)")
