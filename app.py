
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="European Bank Churn Analytics", layout="wide")

st.title("🏦 Customer Segmentation & Churn Pattern Analytics")
st.write("Interactive dashboard based on the European Banking churn dataset.")

@st.cache_data
def load_data():
    return pd.read_csv("European_Bank.csv")

df = load_data()

# Segmentation fields
bins=[18,30,45,60,100]
labels=['Young','Adult','Middle Age','Senior']
df['AgeGroup']=pd.cut(df['Age'],bins=bins,labels=labels)

def credit_segment(score):
    if score < 500:
        return 'Low'
    elif score < 700:
        return 'Medium'
    return 'High'

df['CreditCategory']=df['CreditScore'].apply(credit_segment)

def balance_group(balance):
    if balance == 0:
        return 'Zero Balance'
    elif balance < 100000:
        return 'Low Balance'
    return 'High Balance'

df['BalanceGroup']=df['Balance'].apply(balance_group)

def tenure_group(t):
    if t <= 3:
        return 'New'
    elif t <= 7:
        return 'Mid Term'
    return 'Long Term'

df['TenureGroup']=df['Tenure'].apply(tenure_group)

# Sidebar filters
st.sidebar.header("Filters")
geo = st.sidebar.multiselect("Geography", df["Geography"].unique(), default=list(df["Geography"].unique()))
gender = st.sidebar.multiselect("Gender", df["Gender"].unique(), default=list(df["Gender"].unique()))

df = df[df["Geography"].isin(geo) & df["Gender"].isin(gender)]

# KPIs
total_customers = len(df)
churn_rate = df["Exited"].mean()*100
revenue_risk = df[df["Exited"]==1]["Balance"].sum()
premium = df[df["Balance"] > 100000]
premium_churn = premium["Exited"].mean()*100 if len(premium)>0 else 0

c1,c2,c3,c4 = st.columns(4)
c1.metric("Total Customers", f"{total_customers:,}")
c2.metric("Churn Rate", f"{churn_rate:.2f}%")
c3.metric("Revenue At Risk", f"€{revenue_risk:,.0f}")
c4.metric("Premium Churn", f"{premium_churn:.2f}%")

st.divider()

# Charts
col1,col2 = st.columns(2)

with col1:
    st.subheader("Customer Churn Distribution")
    fig, ax = plt.subplots()
    df["Exited"].value_counts().plot(kind="bar", color=["pink","purple"], ax=ax)
    ax.set_xlabel("Exited")
    ax.set_ylabel("Count")
    st.pyplot(fig)

with col2:
    st.subheader("Geography-wise Churn Rate")
    fig, ax = plt.subplots()
    (df.groupby("Geography")["Exited"].mean()*100).plot(kind="bar", color="skyblue", ax=ax)
    ax.set_ylabel("Churn Rate (%)")
    st.pyplot(fig)

col1,col2 = st.columns(2)

with col1:
    st.subheader("Gender-wise Churn")
    fig, ax = plt.subplots()
    (df.groupby("Gender")["Exited"].mean()*100).plot(kind="bar", color="orange", ax=ax)
    ax.set_ylabel("Churn Rate (%)")
    st.pyplot(fig)

with col2:
    st.subheader("Age Group Churn")
    fig, ax = plt.subplots()
    (df.groupby("AgeGroup")["Exited"].mean()*100).plot(kind="bar", color="green", ax=ax)
    ax.set_ylabel("Churn Rate (%)")
    st.pyplot(fig)

col1,col2 = st.columns(2)

with col1:
    st.subheader("Active Member vs Churn")
    fig, ax = plt.subplots()
    (df.groupby("IsActiveMember")["Exited"].mean()*100).plot(kind="bar", color="violet", ax=ax)
    ax.set_ylabel("Churn Rate (%)")
    st.pyplot(fig)

with col2:
    st.subheader("Products vs Churn")
    fig, ax = plt.subplots()
    (df.groupby("NumOfProducts")["Exited"].mean()*100).plot(kind="bar", color="hotpink", ax=ax)
    ax.set_ylabel("Churn Rate (%)")
    st.pyplot(fig)

col1,col2 = st.columns(2)

with col1:
    st.subheader("Credit Score Distribution")
    fig, ax = plt.subplots()
    sns.boxplot(x="Exited", y="CreditScore", data=df, ax=ax)
    st.pyplot(fig)

with col2:
    st.subheader("Balance Distribution")
    fig, ax = plt.subplots()
    sns.boxplot(x="Exited", y="Balance", data=df, ax=ax)
    st.pyplot(fig)

st.subheader("Correlation Heatmap")
fig, ax = plt.subplots(figsize=(8,6))
sns.heatmap(df.corr(numeric_only=True), annot=True, ax=ax)
st.pyplot(fig)

st.subheader("Dataset Preview")
st.dataframe(df.head(20))
