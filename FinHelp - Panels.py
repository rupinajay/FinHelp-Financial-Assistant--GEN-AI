import panel as pn
import plotly.express as px
from langchain_community.llms import Ollama
import pandas as pd
import numpy as np
import io
from PIL import Image
import pytesseract

pn.extension()

llm = Ollama(model='Finance')

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

def make_grouped_bar_chart(df, label,year):
    income_df = df[df['Expense/Income'] == label]

    grouped_bar_fig = px.bar(income_df, x='Category', y='Amount (EUR)', color='Year',
                             labels={'Amount (EUR)': 'Total ' + label + ' (EUR)'},
                             title=label +" "+ str(year))
    return grouped_bar_fig

# Grouped bar chart for income breakdown
income_grouped_bar_fig_2022 = make_grouped_bar_chart(df, 'Income',2022)
income_grouped_bar_fig_2023 = make_grouped_bar_chart(df, 'Income',2023)

# Bar charts
income_monthly_2022 = make_monthly_bar_chart(df, 2022, 'Income')
expense_monthly_2022 = make_monthly_bar_chart(df, 2022, 'Expense')
income_monthly_2023 = make_monthly_bar_chart(df, 2023, 'Income')
expense_monthly_2023 = make_monthly_bar_chart(df, 2023, 'Expense')

async def callback(contents, user, instance):
    callback_handler = pn.chat.langchain.PanelCallbackHandler(instance)
    return await llm.apredict(contents, callbacks=[callback_handler])

# Create a ChatInterface with the Ollama model
chat_interface = pn.chat.ChatInterface(callback=callback)

# Create Financial Dashboard components
financial_tabs = pn.Tabs(
    ('2022', pn.Column(pn.Row(income_grouped_bar_fig_2022, make_pie_chart(df, 2022, 'Expense'),income_monthly_2022, expense_monthly_2022), width=350)),
    ('2023', pn.Column(pn.Row(income_grouped_bar_fig_2023, make_pie_chart(df, 2023, 'Expense'),income_monthly_2023, expense_monthly_2023), width=350))
)

# OCR Model page components
ocr_tab_content = pn.Column(
    pn.pane.Markdown("## OCR Model"),
    pn.widgets.FileInput(accept=".png, .jpg, .jpeg", name="Upload Image"),
    pn.widgets.Button(name="Perform OCR & Invoke Model", button_type="primary"),
    pn.pane.Str(name="Output:", width=800)  # Pane to display the output
)

# Define a callback to perform OCR and invoke the model for the OCR Model page
def perform_ocr_and_invoke_model_ocr_tab(event):
    # Get the uploaded image file
    uploaded_image_ocr_tab = ocr_tab_content[1].value
    
    if uploaded_image_ocr_tab is not None:
        # Read the image data
        image_data_ocr_tab = uploaded_image_ocr_tab
        
        # Open the image using PIL
        img_ocr_tab = Image.open(io.BytesIO(image_data_ocr_tab))
        
        # Perform OCR on the image
        ocr_result_ocr_tab = pytesseract.image_to_string(img_ocr_tab)
        
        # Invoke the model with the OCR result
        llm_result = llm.invoke(ocr_result_ocr_tab)
        
        # Update the output pane with the model output
        ocr_tab_content[-1] = pn.pane.Str(llm_result, width=800)
    else:
        ocr_tab_content[-1] = pn.pane.Str("Please upload an image first.", width=800)

# Assign the callback to the button on the OCR Model page
ocr_tab_content[2].on_click(perform_ocr_and_invoke_model_ocr_tab)

# Combine Financial Dashboard, Chat Interface, and OCR Model Dashboard into a multipage layout
multipage_layout = pn.Tabs(
    ('Financial Dashboard', financial_tabs),
    ('Chat Interface', chat_interface),
    ('OCR Model', pn.Row(ocr_tab_content))
)

# Create a template for the multipage dashboard
multipage_template = pn.template.FastListTemplate(
    title='FinHelp',
    header_background="#183a1d",
    main=multipage_layout
)

# Show the multipage dashboard
multipage_template.show()