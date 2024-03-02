import datetime
import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
import io
from PIL import Image
import pytesseract
from langchain_community.llms import Ollama

llm = Ollama(model='Finance_llama2')

# Read transactions_2022_2023_categorized.csv
df = pd.read_csv('transactions_2022_2023_categorized.csv')

# Add year and month columns
df['Year'] = pd.to_datetime(df['Date']).dt.year
df['Month'] = pd.to_datetime(df['Date']).dt.month
df['Month Name'] = pd.to_datetime(df['Date']).dt.strftime("%b")

# Remove "Transaction" and "Transaction vs category" columns
df = df.drop(columns=['Transaction', 'Transaction vs category'])

# For Income rows, assign Name / Description to Category
df['Category'] = np.where(df['Expense/Income'] == 'Income', df['Name / Description'], df['Category'])

def make_pie_chart(df, year, label):
    sub_df = df[(df['Expense/Income'] == label) & (df['Year'] == year)]

    color_scale = px.colors.qualitative.Dark2

    pie_fig = px.pie(sub_df, values='Amount (EUR)', names='Category', color_discrete_sequence=color_scale)
    pie_fig.update_traces(textposition='inside', direction='clockwise', hole=0.3, textinfo="label+percent")

    total_expense = df[(df['Expense/Income'] == 'Expense') & (df['Year'] == year)]['Amount (EUR)'].sum()
    total_income = df[(df['Expense/Income'] == 'Income') & (df['Year'] == year)]['Amount (EUR)'].sum()

    if label == 'Expense':
        total_text = "€ " + str(round(total_expense))
        saving_rate = round((total_income - total_expense) / total_income * 100)
        saving_rate_text = "- Saving:" + str(saving_rate) + "%"
    else:
        saving_rate_text = ""
        total_text = "€ " + str(round(total_income))

    pie_fig.update_layout(
        uniformtext_minsize=10,
        uniformtext_mode='hide',
        title=dict(text=label + " Breakdown " + str(year) + saving_rate_text),
        annotations=[
            dict(
                text=total_text,
                x=0.5, y=0.5, font_size=12,
                showarrow=False
            )
        ],
    )
    return pie_fig

def make_monthly_bar_chart(df, year, label):
    df = df[(df['Expense/Income'] == label) & (df['Year'] == year)]
    total_by_month = (df.groupby(['Month', 'Month Name'])['Amount (EUR)'].sum()
                      .to_frame()
                      .reset_index()
                      .sort_values(by='Month')
                      .reset_index(drop=True))
    color_scale = px.colors.sequential.YlGn if label == "Income" else px.colors.sequential.OrRd

    bar_fig = px.bar(total_by_month, x='Month Name', y='Amount (EUR)', text_auto='.2s', title=label + " per month",
                     color='Amount (EUR)', color_continuous_scale=color_scale)
    return bar_fig

def make_grouped_bar_chart(df, label, year):
    income_df = df[df['Expense/Income'] == label]

    grouped_bar_fig = px.bar(income_df, x='Category', y='Amount (EUR)', color='Year',
                             labels={'Amount (EUR)': 'Total ' + label + ' (EUR)'},
                             title=label + " " + str(year))
    return grouped_bar_fig

# Grouped bar chart for income breakdown
income_grouped_bar_fig_2022 = make_grouped_bar_chart(df, 'Income', 2022)
income_grouped_bar_fig_2023 = make_grouped_bar_chart(df, 'Income', 2023)

# Bar charts
income_monthly_2022 = make_monthly_bar_chart(df, 2022, 'Income')
expense_monthly_2022 = make_monthly_bar_chart(df, 2022, 'Expense')
income_monthly_2023 = make_monthly_bar_chart(df, 2023, 'Income')
expense_monthly_2023 = make_monthly_bar_chart(df, 2023, 'Expense')

# Homepage
selected_feature = st.sidebar.selectbox("Select Feature", ["Finance Assistive Chat", "Financial Report Analysis", "Expense Analysis Dashboard"])


if selected_feature == "Finance Assistive Chat":
    st.title("Finance Assistive Chat")
    session_state = st.session_state

    # Initialize empty lists for prompts and results
    if 'prompt_history' not in session_state:
        session_state.prompt_history = []
    if 'result_history' not in session_state:
        session_state.result_history = []

    # Helper function to save chat history to a file
    def save_chat_history(prompt_history, result_history):
        with open("chat_history.txt", "w") as file:
            for prompt, result in zip(prompt_history, result_history):
                file.write(f"You: {prompt}\nAssistant: {result}\n\n")
    # User input for prompt
    user_prompt = st.text_input("Ask a question or provide information:", key="user_input")

    # Check if user has entered a prompt
    if st.button("Submit"):
        # Append prompt to history
        session_state.prompt_history.append(user_prompt)

        # Invoke Ollama model with the prompt
        result = llm.invoke(user_prompt + " Give short reply")

        # Append result to history
        session_state.result_history.append(result)

    # Display history with prompts aligned to the left and results to the right
    st.subheader("Chat History:")
    for i in range(len(session_state.prompt_history)):
        st.write(f"**You:** {session_state.prompt_history[i]}")
        st.write(f"**Assistant:** {session_state.result_history[i]}")
        st.write("---")  # Add a separator line between entries

    # Additional functionalities
    st.sidebar.markdown("### Additional Functionalities")

    # Clear chat history button
    if st.sidebar.button("Clear Chat History"):
        session_state.prompt_history = []
        session_state.result_history = []
        st.success("Chat history cleared!")

    # Save chat history to file button
    if st.sidebar.button("Save Chat History"):
        save_chat_history(session_state.prompt_history, session_state.result_history)
        st.success("Chat history saved to file!")




elif selected_feature == "Financial Report Analysis":
    st.title("Financial Report Analysis")
    # Your financial report analysis code here
    
    uploaded_image_ocr_tab = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])
    perform_ocr_and_invoke_model_ocr_tab = st.button("Perform OCR & Invoke Model")

# Define a callback to perform OCR and invoke the model for the OCR Model page
    if uploaded_image_ocr_tab is not None and perform_ocr_and_invoke_model_ocr_tab:
    # Your OCR and model invocation code here
        image_data_ocr_tab = uploaded_image_ocr_tab.read()

    # Open the image using PIL
        img_ocr_tab = Image.open(io.BytesIO(image_data_ocr_tab))

    # Perform OCR on the image
        ocr_result_ocr_tab = pytesseract.image_to_string(img_ocr_tab)

    # Invoke the model with the OCR result
    # Assuming llm.invoke is the correct method for model invocation
        llm_result = llm.invoke(ocr_result_ocr_tab)

    # Display the OCR result and model output
        # Display the OCR result and model output
        st.image(img_ocr_tab, caption="Uploaded Image")
        st.text("OCR Result:")
        st.text(ocr_result_ocr_tab)
        st.text("Model Output:")
        st.text(llm_result)


    elif perform_ocr_and_invoke_model_ocr_tab:
        st.warning("Please upload an image first.")


elif selected_feature == "Expense Analysis Dashboard":
    st.title("Expense Analysis Dashboard")
    # Your expense analysis dashboard code here
    financial_tabs = st.sidebar.radio("Select Year", ['2022', '2023'])
    if financial_tabs == '2022':
        st.plotly_chart(income_grouped_bar_fig_2022, use_container_width=True)
        st.plotly_chart(make_pie_chart(df, 2022, 'Expense'), use_container_width=True)
        st.plotly_chart(income_monthly_2022, use_container_width=True)
        st.plotly_chart(expense_monthly_2022, use_container_width=True)

    elif financial_tabs == '2023':
        st.plotly_chart(income_grouped_bar_fig_2023, use_container_width=True)
        st.plotly_chart(make_pie_chart(df, 2023, 'Expense'), use_container_width=True)
        st.plotly_chart(income_monthly_2023, use_container_width=True)
        st.plotly_chart(expense_monthly_2023, use_container_width=True)