import json
import gzip
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier

from sklearn.base import BaseEstimator, TransformerMixin
from datetime import datetime

st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {
        background-color: #8CC152;
    }
    </style>
    """,
    unsafe_allow_html=True
)

numerical_columns = ['EDAD_CIERRE_VIGENCIA', 'PUNTAJE_SISBEN_OFICIAL', 'Analfabetismo_Total',
                     'Analfabetismo_Parcial', 'Barreras a servicios para cuidado de la primera infancia_Total',
                     'Barreras a servicios para cuidado de la primera infancia_Parcial',
                     'Desempleo de larga duración_Total', 'Desempleo de larga duración_Parcial', 
                     'Hacinamiento crítico_Total', 'Hacinamiento crítico_Parcial', 'Sin acceso a fuente de agua mejorada_Total',
                     'Sin acceso a fuente de agua mejorada_Parcial', 'Sin aseguramiento en salud_Total',
                     'Sin aseguramiento en salud_Parcial', 'Trabajo infantil_Total', 'Trabajo infantil_Parcial',
                     'Trabajo informal_Total', 'Trabajo informal_Parcial']

categorical_columns = ['ID_ENTIDAD', 'TIPO_DOCUMENTO_BEN', 'EDAD_CORTE_REPORTE', 'GENERO_BEN',
                       'GRUPO_ETNICO_BEN01', 'MIGRANTES', 'TIPO_DOCUMENTO_ACUDIENTE', 'ZONA_UBICACION_BEN',
                       'SISBEN_OFICIAL', 'VICTIMA_OFICIAL', 'ESTADO_BENEFICIARIO', 'PresentaDiscapacidad',
                       'DiscapacidadCertificada', 'EntidadCertificaDiscapacidad', 'DiscapacidadFisica',
                       'DiscapacidadVisual', 'DiscapacidadSordoCeguera', 'DiscapacidadAuditiva', 
                       'DiscapacidadMentalCognitiva', 'DiscapacidadMentalPsicosocial', 'DiscapacidadSistemica', 
                       'DiscapacidadSensorialGustoOlfatoTacto', 'DiscapacidadPielPeloUnas', 'DiscapacidadVozHabla',
                       'DiscapacidadMultiple', 'RequiereRehabilitacion', 'RequiereAyudaTecnica', 'CuentaAyudaTecnica',
                       'CuentaRehabilitacion', 'AyudaOtraPersona']

frequency_columns = ['DEPTO_NACIMIENTO_BEN', 'MPIO_NACIMIENTO_BEN', 'PAIS_RESIDENCIA_BEN', 'DEPTO_RESIDENCIA_BEN',
                     'MPIO_RESIDENCIA_BEN', 'COD_MPIO_RESIDENCIA_BEN', 'SECRETARÍA EDUCACIÓN', 'Departamento',
                     'Municipio', 'SERVICIO_BEN', 'MODALIDAD_BEN', 'QUIEN_RESPONSABLE_BEN']

class AgeTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, date_column):
        self.date_column = date_column

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X[self.date_column] = pd.to_datetime(X[self.date_column])
        X['age'] = X[self.date_column].apply(lambda x: datetime.today().year - x.year - ((datetime.today().month, datetime.today().day) < (x.month, x.day)))
        X['birth_day_of_week'] = X[self.date_column].dt.dayofweek
        X['birth_month'] = X[self.date_column].dt.month
        return X.drop(columns=[self.date_column])

class FrequencyEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None):
        self.columns = columns
        self.freq_maps = {}

    def fit(self, X, y=None):
        for col in self.columns:
            freq = X[col].value_counts() / len(X)
            self.freq_maps[col] = freq
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.columns:
            X[col] = X[col].map(self.freq_maps[col])
        return X

# Configurar título del tablero
st.title("Tablero de Control - Modelo de Clasificación")

# Descripción del modelo
st.markdown("Este tablero presenta los resultados del modelo de clasificación.")

# Subir datos de prueba
st.sidebar.header("Cargar datos")
uploaded_file = st.sidebar.file_uploader("Sube un archivo CSV", type="csv")

# ----- Sidebar for Model Upload -----
st.sidebar.header("Cargar modelo")
uploaded_model = st.sidebar.file_uploader("Sube un archivo de modelo (PKL)", type="gz")

uploaded_json = st.sidebar.file_uploader("Carga la estructura de los datos (json)", type="json")

if uploaded_file and uploaded_json and uploaded_model:
    try:
        # Read and decode JSON file
        file_bytes = uploaded_json.read()
        dtypes = json.loads(file_bytes.decode("utf-8"))

        # Read CSV file with specified data types
        data = pd.read_csv(uploaded_file, dtype=dtypes)
        st.write("### Vista previa de los datos")
        st.write(f'Tamaño de la muestra: {data.shape[0]} filas (niñas y niños)')
        st.dataframe(data.head())

        # Load the model
        import joblib
        with st.spinner("Cargando modelo..."):
            with gzip.open(uploaded_model, 'rb') as f:
                model = joblib.load(f)
            st.success("Modelo cargado correctamente.")

        # Prepare data for prediction
        true_labels = data.iloc[:, -1]
        features = data.iloc[:, :-1]

        # Make predictions and calculate probabilities
        predictions = model.predict(features)
        probabilidades = model.predict_proba(features)
        data[['Probabilidad No Tránsito', 'Probabilidad Tránsito']] = np.round(100*probabilidades, 1)
        st.dataframe(data.head())

        # Display classification metrics
        st.write("### Métricas de Clasificación")
        report = classification_report(true_labels, predictions, output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose())

        # Display confusion matrix
        st.write("### Matriz de Confusión")
        cm = confusion_matrix(true_labels, predictions)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicción")
        ax.set_ylabel("Verdadero")
        st.pyplot(fig)

        # Display ROC curve
        if hasattr(model, "predict_proba"):
            from sklearn.metrics import roc_curve, auc
            probabilities = model.predict_proba(features)[:, 1]
            fpr, tpr, thresholds = roc_curve(true_labels, probabilities)
            roc_auc = auc(fpr, tpr)

            st.write("### Curva ROC")
            fig, ax = plt.subplots()
            ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
            ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
            ax.set_xlabel("Tasa de Falsos Positivos")
            ax.set_ylabel("Tasa de Verdaderos Positivos")
            ax.legend(loc="lower right")
            st.pyplot(fig)

        # Display feature importances
        model_ = model.named_steps['classifier']
        feature_importances = model_.feature_importances_
        feature_names = list(numerical_columns) + list(model.named_steps['preprocessor'].transformers_[1][1].named_steps['onehot'].get_feature_names_out(categorical_columns)) + frequency_columns
        indices = np.argsort(feature_importances)[::-1][:15]
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.bar(range(len(indices)), feature_importances[indices], align='center')
        ax.set_xticks(range(len(indices)), np.array(feature_names)[indices], rotation=90)
        ax.set_xlabel('Feature')
        ax.set_ylabel('Importance')
        ax.set_title('Top 10 Feature Importances')
        st.pyplot(fig)
'''
        # Display numeric distributions
        numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
        if not numeric_cols.empty:
            st.write("### Distribución de Variables Numéricas")
            for col in numeric_cols:
                fig, ax = plt.subplots()
                sns.histplot(data[col], bins=25, ax=ax)
                ax.set_title(f"Distribución de {col}")
                st.pyplot(fig)

        # Display class balance
        if 'RESULTADO CRUCE' in data.columns:
            st.write("### Balance de Clases")
            fig, ax = plt.subplots()
            sns.countplot(x='RESULTADO CRUCE', data=data, ax=ax)
            ax.set_title("Distribución de Clases")
            st.pyplot(fig)

        # Display correlation matrix
        if not numeric_cols.empty:
            st.write("### Matriz de Correlación")
            fig, ax = plt.subplots(figsize=(10, 8))
            corr_matrix = data[numeric_cols].corr()
            sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
            ax.set_title("Matriz de Correlación")
            st.pyplot(fig)

        # Display relationships between variables and labels
        if 'RESULTADO CRUCE' in data.columns and not numeric_cols.empty:
            st.write("### Relación entre Variables y Etiquetas")
            for col in numeric_cols:
                fig, ax = plt.subplots()
                sns.boxplot(y='RESULTADO CRUCE', x=col, data=data, orient="h", ax=ax)
                ax.set_title(f"{col} vs Etiquetas")
                st.pyplot(fig)
'''
    except Exception as e:
        st.error(f"Error al cargar los datos o el modelo: {e}")

else:
    st.warning("Por favor, sube un archivo CSV, un archivo de modelo (PKL) y un archivo JSON con la estructura de los datos.")
    
