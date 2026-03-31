import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
st.set_page_config(page_title="Титаник дашборд", layout="wide") 
st.title("анализ данных по титанику")
@st.cache_data
def load_data():
    df = pd.read_csv("titanic.csv")
    return df
df = load_data()

st.header("1. Описательная статистика")
col1, col2 = st.columns(2)
with col1:
    st.subheader("Информация о колонках")
    column_info = pd.DataFrame({
        'Колонка': df.columns,
        'Тип данных': [str(dtype) for dtype in df.dtypes.values],
        'Пропуски': df.isnull().sum().values,
        '% пропусков': (df.isnull().sum() / len(df) * 100).round(1).values,
        'Уникальных значений': df.nunique().values
    })
    st.dataframe(column_info, use_container_width=True, hide_index=True)
with col2:
    st.subheader("Общая информация о датасете")
    st.write(f"**Форма таблицы:** {df.shape[0]} строк × {df.shape[1]} столбцов")
    st.write(f"**Всего пассажиров:** {len(df)}")
    st.write(f"**Выживших:** {df['Survived'].sum()} ({df['Survived'].mean()*100:.1f}%)")
    st.write(f"**Погибших:** {len(df) - df['Survived'].sum()} ({(1 - df['Survived'].mean())*100:.1f}%)")
    st.write(f"**Средний возраст:** {df['Age'].mean():.1f} лет")
    st.write(f"**Медианный возраст:** {df['Age'].median():.1f} лет")
    st.write(f"**Средняя стоимость билета:** ${df['Fare'].mean():.2f}")
    st.subheader("Распределение по классам")
    class_counts = df['Pclass'].value_counts().sort_index()
    for pclass, count in class_counts.items():
        st.write(f"**{pclass} класс:** {count} пассажиров ({count/len(df)*100:.1f}%)")


st.header("Визуализация данных")

# График 1: Круговая диаграмма - выживаемость
st.subheader("1. Выживаемость пассажиров")
survived_counts = df['Survived'].value_counts()
fig1 = px.pie(
    values=survived_counts.values, 
    names=['Погиб', 'Выжил'],
    title='Соотношение выживших и погибших',
    color_discrete_sequence=['#E74C3C', '#2ECC71']
)
st.plotly_chart(fig1, use_container_width=True)
    

# График 2: Гистограмма - возраст
st.subheader("2. Распределение возраста")
fig2 = px.histogram(
    df, 
    x='Age', 
    nbins=30,
    title='Распределение возраста пассажиров',
    labels={'Age': 'Возраст (лет)', 'count': 'Количество'}
)
st.plotly_chart(fig2, use_container_width=True)

# График 3: Столбчатая диаграмма - классы
st.subheader("3. Количество пассажиров по классам")
class_counts = df['Pclass'].value_counts().sort_index()
fig3 = px.bar(
    x=class_counts.index, 
    y=class_counts.values,
    title='Количество пассажиров по классам',
    labels={'x': 'Класс каюты', 'y': 'Количество'},
    color=class_counts.index,
    text=class_counts.values
)
fig3.update_traces(textposition='outside')
st.plotly_chart(fig3, use_container_width=True)

# График 4: Ящик с усами - стоимость билета
st.subheader("4. Стоимость билета по классам")
fig4 = px.box(
    df,
    x='Pclass',
    y='Fare',
    title='Стоимость билета в разных классах',
    labels={'Pclass': 'Класс', 'Fare': 'Стоимость ($)'}
)
st.plotly_chart(fig4, use_container_width=True)

# График 5: Линейчатая диаграмма - выживаемость по полу
st.subheader("5. Выживаемость по полу")
gender_survival = df.groupby('Sex')['Survived'].mean() * 100
gender_survival.index = ['Мужской', 'Женский']
fig5 = px.bar(
    x=gender_survival.index,
    y=gender_survival.values,
    title='Процент выживших по полу',
    labels={'x': 'Пол', 'y': 'Выживаемость (%)'},
    color=gender_survival.index,
    text=gender_survival.values
)
fig5.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
fig5.update_layout(yaxis_range=[0, 100])
st.plotly_chart(fig5, use_container_width=True)

st.divider()

st.header("Интерактивный график")
st.write("График меняется при выборе фильтров в боковой панели 👈")

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
