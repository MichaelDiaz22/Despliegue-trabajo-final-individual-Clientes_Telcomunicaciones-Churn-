import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sklearn
import sys
import os

# Configuración de la página
st.set_page_config(
    page_title="Predicción de Churn de Clientes",
    page_icon="📊",
    layout="wide"
)

# Verificar versiones (para debugging)
st.sidebar.write(f"Python: {sys.version}")
st.sidebar.write(f"scikit-learn: {sklearn.__version__}")

# Título de la aplicación
st.title("📊 Predicción de Churn de Clientes de Telecomunicaciones")
st.markdown("Ingresa los datos del cliente para predecir la probabilidad de churn")

# Función para cargar modelos con manejo robusto de errores
@st.cache_resource
def load_model(model_path):
    try:
        model = joblib.load(model_path)
        st.sidebar.success(f"✅ Modelo cargado: {os.path.basename(model_path)}")
        return model
    except Exception as e:
        st.sidebar.error(f"❌ Error cargando {model_path}: {str(e)}")
        return None

# Cargar modelos
try:
    classical_model = load_model('best_classical_model_pipeline.joblib')
    ensemble_model = load_model('best_ensemble_model_pipeline.joblib')
except Exception as e:
    st.error(f"Error crítico al cargar modelos: {e}")
    st.stop()

# Verificar que los modelos se cargaron correctamente
if classical_model is None or ensemble_model is None:
    st.error("No se pudieron cargar uno o más modelos. Verifica los archivos .joblib")
    st.stop()

# Crear formulario para entrada de datos
with st.form("customer_data_form"):
    st.header("📋 Información del Cliente")
    
    col1, col2 = st.columns(2)
    
    with col1:
        gender = st.selectbox("Género", ['Female', 'Male'])
        SeniorCitizen = st.selectbox("Ciudadano Mayor", [0, 1], 
                                   help="0 = No, 1 = Sí")
        Partner = st.selectbox("Pareja", ['Yes', 'No'])
        Dependents = st.selectbox("Dependientes", ['Yes', 'No'])
        tenure = st.slider("Meses de Antigüedad", 0, 72, 1)
        PhoneService = st.selectbox("Servicio Telefónico", ['Yes', 'No'])
        MultipleLines = st.selectbox("Líneas Múltiples", 
                                   ['No phone service', 'No', 'Yes'])
    
    with col2:
        InternetService = st.selectbox("Servicio de Internet", 
                                     ['DSL', 'Fiber optic', 'No'])
        Contract = st.selectbox("Tipo de Contrato", 
                              ['Month-to-month', 'One year', 'Two year'])
        PaperlessBilling = st.selectbox("Facturación Sin Papel", ['Yes', 'No'])
        PaymentMethod = st.selectbox("Método de Pago", [
            'Electronic check', 
            'Mailed check', 
            'Bank transfer (automatic)', 
            'Credit card (automatic)'
        ])
        MonthlyCharges = st.number_input("Cargos Mensuales ($)", 
                                       min_value=0.0, max_value=200.0, 
                                       value=50.0, step=1.0)
        TotalCharges = st.number_input("Cargos Totales ($)", 
                                     min_value=0.0, max_value=10000.0, 
                                     value=1000.0, step=10.0)
    
    # Servicios adicionales
    st.header("🛜 Servicios Adicionales")
    
    col3, col4 = st.columns(2)
    
    with col3:
        OnlineSecurity = st.selectbox("Seguridad en Línea", 
                                    ['No', 'Yes', 'No internet service'])
        OnlineBackup = st.selectbox("Copia de Seguridad en Línea", 
                                  ['Yes', 'No', 'No internet service'])
        DeviceProtection = st.selectbox("Protección de Dispositivo", 
                                      ['No', 'Yes', 'No internet service'])
    
    with col4:
        TechSupport = st.selectbox("Soporte Técnico", 
                                 ['No', 'Yes', 'No internet service'])
        StreamingTV = st.selectbox("TV en Streaming", 
                                 ['No', 'Yes', 'No internet service'])
        StreamingMovies = st.selectbox("Películas en Streaming", 
                                     ['No', 'Yes', 'No internet service'])
    
    # Botón de predicción
    submitted = st.form_submit_button("🔮 Predecir Churn", type="primary")

# Procesar cuando se envía el formulario
if submitted:
    try:
        # Crear DataFrame con los datos de entrada
        input_data = {
            'gender': [gender],
            'SeniorCitizen': [SeniorCitizen],
            'Partner': [Partner],
            'Dependents': [Dependents],
            'tenure': [tenure],
            'PhoneService': [PhoneService],
            'MultipleLines': [MultipleLines],
            'InternetService': [InternetService],
            'OnlineSecurity': [OnlineSecurity],
            'OnlineBackup': [OnlineBackup],
            'DeviceProtection': [DeviceProtection],
            'TechSupport': [TechSupport],
            'StreamingTV': [StreamingTV],
            'StreamingMovies': [StreamingMovies],
            'Contract': [Contract],
            'PaperlessBilling': [PaperlessBilling],
            'PaymentMethod': [PaymentMethod],
            'MonthlyCharges': [MonthlyCharges],
            'TotalCharges': [TotalCharges]
        }
        
        input_df = pd.DataFrame(input_data)
        
        # Asegurar que TotalCharges sea numérico
        input_df['TotalCharges'] = pd.to_numeric(input_df['TotalCharges'], errors='coerce')
        
        # Realizar predicciones
        prediction_classical = classical_model.predict(input_df)[0]
        prediction_ensemble = ensemble_model.predict(input_df)[0]
        
        # Obtener probabilidades si están disponibles
        try:
            proba_classical = classical_model.predict_proba(input_df)[0]
            proba_ensemble = ensemble_model.predict_proba(input_df)[0]
        except:
            proba_classical = [0, 0]
            proba_ensemble = [0, 0]
        
        # Mostrar resultados
        st.header("📊 Resultados de la Predicción")
        
        col5, col6 = st.columns(2)
        
        with col5:
            st.subheader("Modelo Clásico")
            if prediction_classical == 1:
                st.error(f"🔴 **CHURN** - Probabilidad: {proba_classical[1]:.2%}")
            else:
                st.success(f"🟢 **NO CHURN** - Probabilidad: {proba_classical[0]:.2%}")
        
        with col6:
            st.subheader("Modelo Ensemble")
            if prediction_ensemble == 1:
                st.error(f"🔴 **CHURN** - Probabilidad: {proba_ensemble[1]:.2%}")
            else:
                st.success(f"🟢 **NO CHURN** - Probabilidad: {proba_ensemble[0]:.2%}")
        
        # Mostrar resumen
        st.header("📈 Resumen")
        if prediction_classical == prediction_ensemble:
            if prediction_classical == 1:
                st.error("⚠️ **ALERTA**: Ambos modelos predicen CHURN. Este cliente tiene alta probabilidad de cancelar el servicio.")
            else:
                st.success("✅ **ESTABLE**: Ambos modelos predicen NO CHURN. El cliente probablemente se mantendrá.")
        else:
            st.warning("🟡 **CONFLICTO**: Los modelos tienen predicciones diferentes. Se recomienda análisis adicional.")
        
        # Mostrar datos ingresados
        with st.expander("📋 Ver datos ingresados"):
            st.dataframe(input_df.T.rename(columns={0: 'Valor'}))
            
    except Exception as e:
        st.error(f"❌ Error durante la predicción: {str(e)}")
        st.info("💡 Verifica que todos los campos estén completos y sean válidos.")

# Información adicional en el sidebar
st.sidebar.header("ℹ️ Información")
st.sidebar.info("""
**Instrucciones:**
1. Completa todos los campos del formulario
2. Haz clic en 'Predecir Churn'
3. Revisa los resultados de ambos modelos

**Notas:**
- Churn = El cliente cancelará el servicio
- No Churn = El cliente se mantendrá
""")

# Footer
st.markdown("---")
st.markdown("*Sistema de predicción de churn - Análisis Predictivo*")
