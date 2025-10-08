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

# Título de la aplicación
st.title("📊 Predicción de Churn de Clientes de Telecomunicaciones")
st.markdown("Ingresa los datos del cliente para predecir la probabilidad de churn")

# Función para cargar modelos con manejo de compatibilidad
def load_model_with_fallback(model_path, fallback_path=None):
    try:
        model = joblib.load(model_path)
        st.sidebar.success(f"✅ Modelo cargado: {os.path.basename(model_path)}")
        return model
    except Exception as e:
        st.sidebar.warning(f"⚠️ Error con {model_path}: {str(e)}")
        
        # Intentar cargar versión compatible
        if fallback_path and os.path.exists(fallback_path):
            try:
                model = joblib.load(fallback_path)
                st.sidebar.success(f"✅ Modelo de respaldo cargado: {os.path.basename(fallback_path)}")
                return model
            except Exception as e2:
                st.sidebar.error(f"❌ Error con modelo de respaldo: {str(e2)}")
        
        return None

# Función alternativa para crear modelo simple si no se pueden cargar
def create_simple_model():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    
    # Crear un modelo simple
    numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
    categorical_features = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ]
    )
    
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=50, random_state=42))
    ])
    
    return model

# Cargar o crear modelos
classical_model = load_model_with_fallback(
    'best_classical_model_pipeline_compatible.joblib')

ensemble_model = load_model_with_fallback(
    'best_ensemble_model_pipeline_compatible.joblib')

# Si no se pudieron cargar los modelos, mostrar opción para usar modelo simple
if classical_model is None or ensemble_model is None:
    st.error("⚠️ No se pudieron cargar los modelos entrenados.")
    
    st.info("""
    **Solución:**
    1. **Opción recomendada:** Reentrena los modelos con la misma versión de scikit-learn del entorno de producción
    2. **Opción temporal:** Usa el modelo simple integrado (menor precisión)
    """)
    
    use_simple_model = st.checkbox("Usar modelo simple integrado", value=False)
    
    if use_simple_model:
        try:
            with st.spinner("Creando modelo simple..."):
                classical_model = create_simple_model()
                ensemble_model = create_simple_model()
                st.success("✅ Modelos simples creados exitosamente")
        except Exception as e:
            st.error(f"Error creando modelo simple: {e}")
            st.stop()
    else:
        st.stop()

# Crear formulario para entrada de datos
with st.form("customer_data_form"):
    st.header("📋 Información del Cliente")
    
    col1, col2 = st.columns(2)
    
    with col1:
        gender = st.selectbox("Género", ['Female', 'Male'])
        SeniorCitizen = st.selectbox("Ciudadano Mayor", [0, 1])
        Partner = st.selectbox("Pareja", ['Yes', 'No'])
        Dependents = st.selectbox("Dependientes", ['Yes', 'No'])
        tenure = st.slider("Meses de Antigüedad", 0, 72, 12)
        PhoneService = st.selectbox("Servicio Telefónico", ['Yes', 'No'])
    
    with col2:
        MultipleLines = st.selectbox("Líneas Múltiples", ['No phone service', 'No', 'Yes'])
        InternetService = st.selectbox("Servicio de Internet", ['DSL', 'Fiber optic', 'No'])
        Contract = st.selectbox("Tipo de Contrato", ['Month-to-month', 'One year', 'Two year'])
        PaperlessBilling = st.selectbox("Facturación Sin Papel", ['Yes', 'No'])
        MonthlyCharges = st.number_input("Cargos Mensuales ($)", min_value=0.0, max_value=200.0, value=50.0, step=1.0)
        TotalCharges = st.number_input("Cargos Totales ($)", min_value=0.0, max_value=10000.0, value=1000.0, step=10.0)
    
    # Servicios adicionales
    st.header("🛜 Servicios Adicionales")
    
    col3, col4 = st.columns(2)
    
    with col3:
        OnlineSecurity = st.selectbox("Seguridad en Línea", ['No', 'Yes', 'No internet service'])
        OnlineBackup = st.selectbox("Copia de Seguridad en Línea", ['Yes', 'No', 'No internet service'])
        DeviceProtection = st.selectbox("Protección de Dispositivo", ['No', 'Yes', 'No internet service'])
    
    with col4:
        TechSupport = st.selectbox("Soporte Técnico", ['No', 'Yes', 'No internet service'])
        StreamingTV = st.selectbox("TV en Streaming", ['No', 'Yes', 'No internet service'])
        StreamingMovies = st.selectbox("Películas en Streaming", ['No', 'Yes', 'No internet service'])
    
    PaymentMethod = st.selectbox("Método de Pago", [
        'Electronic check', 
        'Mailed check', 
        'Bank transfer (automatic)', 
        'Credit card (automatic)'
    ])
    
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
        
        # Asegurar que las columnas numéricas sean correctas
        input_df['SeniorCitizen'] = input_df['SeniorCitizen'].astype(int)
        input_df['tenure'] = input_df['tenure'].astype(int)
        input_df['MonthlyCharges'] = input_df['MonthlyCharges'].astype(float)
        input_df['TotalCharges'] = pd.to_numeric(input_df['TotalCharges'], errors='coerce')
        
        # Realizar predicciones
        prediction_classical = classical_model.predict(input_df)[0]
        prediction_ensemble = ensemble_model.predict(input_df)[0]
        
        # Mostrar resultados
        st.header("📊 Resultados de la Predicción")
        
        col5, col6 = st.columns(2)
        
        with col5:
            st.subheader("Modelo Clásico")
            if prediction_classical == 1:
                st.error("🔴 **CHURN** - El cliente probablemente cancelará el servicio")
            else:
                st.success("🟢 **NO CHURN** - El cliente probablemente se mantendrá")
        
        with col6:
            st.subheader("Modelo Ensemble")
            if prediction_ensemble == 1:
                st.error("🔴 **CHURN** - El cliente probablemente cancelará el servicio")
            else:
                st.success("🟢 **NO CHURN** - El cliente probablemente se mantendrá")
        
        # Mostrar resumen
        st.header("📈 Resumen")
        if prediction_classical == prediction_ensemble:
            if prediction_classical == 1:
                st.error("""
                ⚠️ **ALERTA DE CHURN** 
                
                Ambos modelos predicen que este cliente tiene alta probabilidad de cancelar el servicio.
                Se recomienda:
                - Contactar al cliente
                - Ofrecer incentivos de retención
                - Analizar causas de insatisfacción
                """)
            else:
                st.success("""
                ✅ **CLIENTE ESTABLE** 
                
                Ambos modelos predicen que el cliente se mantendrá.
                Se recomienda:
                - Continuar con el servicio actual
                - Monitorear cambios en el comportamiento
                """)
        else:
            st.warning("""
            🟡 **PREDICCIÓN INCIERTA** 
            
            Los modelos tienen predicciones diferentes. 
            Se recomienda análisis adicional y seguimiento cercano.
            """)
        
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

