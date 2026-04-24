from flask import Flask,render_template,request,url_for,send_file
import pandas as pd
import numpy as np
from reportlab.platypus import SimpleDocTemplate,Paragraph
from reportlab.lib.styles import getSampleStyleSheet
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report,r2_score,mean_absolute_error,mean_squared_error
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split


UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)

df = None

@app.route('/')
def home():
    return render_template("home.html")


@app.route('/preview',methods=['POST'])
def preview():
    global df

    file = request.files.get('file')
    target = request.form.get('target')

    if not file:
        return "NO file uploaded"

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    filename = file.filename  

    if filename.endswith('.csv'):
        df = pd.read_csv(filepath)

    elif filename.endswith('.xlsx'):
        df = pd.read_excel(filepath)
    else:
        return "File format not supported"
        
    preview = df.head(10).to_html(classes='table')

    rows = df.shape[0]
    cols = df.shape[1]
    missing = df.isnull().sum()
    duplicates = df.duplicated().sum()
    columns = df.columns
    data_types = df.dtypes.to_dict()
    unique_values = df.nunique().to_dict()

    outliers = {}

    for col in df.select_dtypes(include=['int64','float64']):
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        count = ((df[col] < lower) | (df[col] > upper)).sum()
        outliers[col] = count

    suggestion = {}

    for col in df.columns:
        missing_percent = (df[col].isnull().sum()/len(df)) * 100

        if missing_percent > 50:
            suggestion[col] = "Drop Column"
        else:
            suggestion[col] = "Keep"
            
    
    return render_template('dataclean.html',
        preview=preview,rows=rows,cols=cols,missing=missing,
        duplicates=duplicates,columns=columns,data_types=data_types,
        unique_values=unique_values,outliers=outliers,suggestion=suggestion,
        filename=filename, 
        target=target
    )

@app.route('/dataclean')
def dataclean():
    return render_template("dataclean.html")


@app.route('/clean',methods=['POST'])
def clean():
    filename = request.form.get('filename') 
    target = request.form.get('target')

    if not filename:
        return "No data found, please upload file first"

    filepath = os.path.join(UPLOAD_FOLDER, filename)

    if filepath.endswith('.csv'):
        df = pd.read_csv(filepath)
    else:
        df = pd.read_excel(filepath)

    if df is None:
        return "No data found, please upload file first"

    if target in df.columns:
        df = df.drop(columns=[target])
    else:
        return "Target column not found"
        
    df = df.drop_duplicates()

    df = df.fillna(df.median(numeric_only=True))

    for col in df.select_dtypes(include='object'):
        df[col] = df[col].fillna(df[col].mode()[0])

    for col in df.select_dtypes(include='object'):
        df[col] = df[col].str.strip()
    
    df = df.convert_dtypes()

    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    if 'gender' in df.columns:
        df['gender'] = df['gender'].str.lower()

    rows_before = df.shape[0]

    for col in df.select_dtypes(include=['int64','float64']):
        lower = df[col].quantile(0.01)
        upper = df[col].quantile(0.99)
        df = df[(df[col] >= lower) & (df[col] <= upper)]

    rows_after = df.shape[0]

    outlier_removed = rows_before - rows_after

    rows = df.shape[0]
    cols = df.shape[1]
    missing = df.isnull().sum()
    duplicates = df.duplicated().sum()
    columns = df.columns
    data_types = df.dtypes.to_dict()
    unique_values = df.nunique().to_dict()

    preview = df.head(10).to_html(classes='table')

    doc = SimpleDocTemplate("report.pdf")
    styles = getSampleStyleSheet()

    content = []
    
    content.append(Paragraph("Data Cleaning Report", styles['Title']))
    content.append(Paragraph("About Report", styles['Heading2']))

    content.append(Paragraph(
    "This report explains the data cleaning steps performed on the dataset. "
    "It includes information about missing values, duplicate rows, outliers removal, "
    "data types and unique values in each column.",
    styles['Normal']
    ))

    content.append(Paragraph(f"Total Rows: {rows}", styles['Normal']))
    content.append(Paragraph(f"Total Columns: {cols}", styles['Normal']))

    content.append(Paragraph(f"Rows Before Cleaning: {rows_before}", styles['Normal']))
    content.append(Paragraph(f"Rows After Cleaning: {rows_after}", styles['Normal']))
    content.append(Paragraph(f"Outliers Removed: {outlier_removed}", styles['Normal']))

    content.append(Paragraph("Missing Values:", styles['Heading2']))
    for col, val in missing.items():
        content.append(Paragraph(f"{col}: {val}", styles['Normal']))

    content.append(Paragraph(f"Duplicate Rows: {duplicates}", styles['Normal']))

    content.append(Paragraph("Data Types:", styles['Heading2']))
    for col, dtype in data_types.items():
        content.append(Paragraph(f"{col}: {dtype}", styles['Normal']))

    content.append(Paragraph("Unique Values:", styles['Heading2']))
    for col, val in unique_values.items():
        content.append(Paragraph(f"{col}: {val}", styles['Normal']))

    doc.build(content)

    return render_template('dataclean.html',
        preview=preview,rows=rows,cols=cols,
        missing=missing,duplicates=duplicates,columns=columns,
        data_types=data_types,unique_values=unique_values,
        outlier_removed=outlier_removed,
        filename=filename,  
        target=target        
    )


@app.route('/download')
def download():
    global df
    df.to_excel("cleaned.xlsx", index=False)
    return send_file("cleaned.xlsx", as_attachment=True)


@app.route('/download_report')
def download_report():
    return send_file("report.pdf", as_attachment=True)

@app.route('/visual', methods=['GET','POST'])
def visual():
    global df

    if request.method == 'POST':
        file = request.files.get('file')
        if not file:
            return "No data found, please upload file first"

        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        if file.filename.endswith('.csv'):
            df = pd.read_csv(filepath)
        else:
            df = pd.read_excel(filepath)

    if df is None:
        return render_template("visual.html")

    graphs = []

    missing = df.isnull().sum()
    missing = missing[missing > 0]

    if len(missing) > 0:
        plt.figure(figsize=(16,8))
        missing.plot(kind='bar')
        plt.xlabel("Columns", fontsize=14)
        plt.ylabel("Missing Count", fontsize=14)
        plt.xticks(rotation=45, fontsize=12)
        plt.yticks(fontsize=12)
        plt.tight_layout()
        path1 = "static/graph1.png"
        plt.savefig(path1, dpi=200)
        plt.close()
        graphs.append(path1)

    num_col = df.select_dtypes(include=['int64','float64'])

    if num_col.shape[1] > 0:
        plt.figure(figsize=(16,10))
        num_col.hist(figsize=(14,8))
        plt.tight_layout()
        path2 = "static/graph2.png"
        plt.savefig(path2, dpi=200)
        plt.close()
        graphs.append(path2)

        plt.figure(figsize=(16,8))
        num_col.plot(kind='box')
        plt.xticks(rotation=45, fontsize=12)
        plt.yticks(fontsize=12)
        plt.tight_layout()
        path3 = "static/graph3.png"
        plt.savefig(path3, dpi=200)
        plt.close()
        graphs.append(path3)

        plt.figure(figsize=(16,10))
        sns.heatmap(num_col.corr(), annot=True, fmt=".2f")
        plt.xticks(rotation=45, fontsize=12)
        plt.yticks(fontsize=12)
        plt.tight_layout()
        path4 = "static/graph4.png"
        plt.savefig(path4, dpi=200)
        plt.close()
        graphs.append(path4)

    cat_col = df.select_dtypes(include=['object'])

    if len(cat_col.columns) > 0:
        plt.figure(figsize=(16,8))
        cat_col.iloc[:,0].value_counts().plot(kind='bar')
        plt.xlabel("Category", fontsize=14)
        plt.ylabel("Count", fontsize=14)
        plt.xticks(rotation=45, fontsize=12)
        plt.yticks(fontsize=12)
        plt.tight_layout()
        path5 = "static/graph5.png"
        plt.savefig(path5, dpi=200)
        plt.close()
        graphs.append(path5)

    if num_col.shape[1] > 0:
        plt.figure(figsize=(16,8))
        num_col.iloc[:,0].plot()
        plt.xlabel("Index", fontsize=14)
        plt.ylabel("Value", fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.tight_layout()
        path6 = "static/graph6.png"
        plt.savefig(path6, dpi=200)
        plt.close()
        graphs.append(path6)

    return render_template("visual.html", graphs=graphs)


@app.route('/download_visual')
def download_visual():
    from reportlab.platypus import SimpleDocTemplate, Image

    pdf = "visual.pdf"
    doc = SimpleDocTemplate(pdf)

    elements = []

    for i in range(1,7):
        path = f"static/graph{i}.png"
        elements.append(Image(path, width=400, height=250))

    doc.build(elements)

    return send_file(pdf, as_attachment=True)

@app.route('/model',methods=['GET','POST'])
def ml_model():
    if request.method == "GET":
        return render_template("model.html")

    file = request.files.get('file')
    target = request.form.get('target')

    if not file:
        return "NO file uploaded"

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    filename = file.filename  

    if filename.endswith('.csv'):
        df = pd.read_csv(filepath)

    elif filename.endswith('.xlsx'):
        df = pd.read_excel(filepath)
    
    else:
        return "File format not supported"

    if target not in df.columns:
        return "Target column not found"
    

    X = df.drop(columns=[target])
    y = df[target]

    X_train,X_test,y_train,y_test = train_test_split(
        X,y,test_size=0.2,random_state=42
    )

    num_col = X.select_dtypes(include=['int64','float64']).columns
    cat_col = X.select_dtypes(include=['object']).columns

    preprocessor = ColumnTransformer([
        ('num','passthrough',num_col),
        ('cat',OneHotEncoder(handle_unknown='ignore'),cat_col)
    ])

    if y.nunique() <= 2:
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        task = "classification"
    else:
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        task = "regression"

    pipeline = Pipeline([
        ('preprocessing',preprocessor),
        ('model',model)
    ])

    pipeline.fit(X_train,y_train)

    y_pred = pipeline.predict(X_test)


    import pickle
    with open("model.pkl", "wb") as f:
        pickle.dump(pipeline, f)

    result = {}

    if task == "classification":
        result['rf'] = round(accuracy_score(y_test,y_pred),4)
        result['cm'] = confusion_matrix(y_test,y_pred).tolist()
    else:
        result['r2'] = round(r2_score(y_test,y_pred),4)

    return render_template("model.html", result=result)

@app.route('/download_model')
def download_model():
    return send_file("model.pkl", as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)
