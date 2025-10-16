
# Análisis del Trabajo Infantil en Bogotá  
## Metodología CRISP-DM

Este informe presenta el desarrollo del proyecto **“Diferencias territoriales y demográficas del trabajo infantil en Bogotá”**, aplicando la metodología **CRISP-DM** (Cross Industry Standard Process for Data Mining).  
El proceso se estructuró en seis fases principales: *Business Understanding*, *Data Understanding*, *Data Preparation*, *Modeling/Evaluation* y *Deployment*.  
A continuación, se describe el desarrollo de cada una.

---

## 1️. Business Understanding

### Objetivo general
Identificar la situación de los niños, niñas y adolescentes en condición de trabajo infantil y adolescente en Bogotá, con base en los registros de acompañamiento social, familiar y psicológico.  
El fin es obtener una visión clara del fenómeno y apoyar la toma de decisiones informadas.

### Objetivos específicos
- **Caracterizar** la población en condición de trabajo infantil/adolescente según variables demográficas: edad, género y localidad.  
- **Analizar** las condiciones familiares y de acompañamiento social reportadas en los registros.  
- **Identificar** diferencias relevantes entre localidades para dimensionar cómo varía el fenómeno dentro de la ciudad.  
- **Organizar** la información en formatos claros y visuales para facilitar el entendimiento y el uso posterior.

---

## 2️. Data Understanding

Durante esta fase se realizó una **exploración inicial y limpieza automatizada** de la base de datos, que contenía **5.139.043 registros y 91 variables**.

### Actividades principales
- **Anonimización de datos sensibles**, eliminando información personal identificable.  
- **Normalización de nombres de variables** para garantizar coherencia en los análisis posteriores.  
- **Generación de gráficos y cruces automáticos** para comprender la distribución de las variables demográficas y territoriales.  
- **Creación de un diccionario de variables** y exportación a Excel, identificando columnas con valores faltantes o inconsistentes.

Esta fase permitió un entendimiento general de la información y sentó las bases para las etapas siguientes.

---

## 3️. Data Preparation

El objetivo fue **depurar y estructurar el dataset** garantizando su calidad antes del análisis.

### Variables seleccionadas
Se eligieron **8 variables clave** relacionadas con las condiciones sociales y demográficas:
1. AFILIACIÓN AL SGSSS  
2. CATEGORÍAS_DE_LA_DISCAPACIDAD  
3. LOCALIDAD_FIC  
4. NACIONALIDAD  
5. SEXO  
6. ¿DÓNDE TRABAJA?  
7. VÍNCULO CON EL JEFE DE HOGAR  
8. EDAD

### Tratamiento de datos
- **Eliminación** de variables con más del 50 % de datos faltantes.  
- **Imputación de valores faltantes**:  
  - Numéricos con la **mediana**.  
  - Categóricos reemplazados por “No aplica”.  
- **Reformateo y exportación** a archivos **CSV** y **Excel**, listos para análisis exploratorio.

El resultado fue un dataset **limpio, completo y trazable**, alineado con los objetivos del proyecto.

---

## 4️. Evaluation (Modelado y análisis exploratorio)

En esta etapa se evaluaron los resultados obtenidos de acuerdo con los tres objetivos específicos del proyecto:

### a. Caracterización demográfica
Se analizó la distribución por **edad, género y localidad**, evidenciando patrones diferenciales del trabajo infantil en distintos sectores de la ciudad.


<img width="1590" height="590" alt="image" src="https://github.com/user-attachments/assets/da9d9edd-a754-4205-a7cb-146b9e3758d9" />

<img width="989" height="590" alt="image" src="https://github.com/user-attachments/assets/2210d56d-3e9c-4f77-b071-bb179273c890" />

<img width="1389" height="690" alt="image" src="https://github.com/user-attachments/assets/4991bb5e-53f3-4656-a3f1-995f4aba207b" />

<img width="1389" height="690" alt="image" src="https://github.com/user-attachments/assets/9de1c674-9ac3-4e32-a2d8-05d0d27ffecd" />

<img width="1389" height="689" alt="image" src="https://github.com/user-attachments/assets/94f51cc1-213d-417e-88d0-92a0b0cf67ec" />

<img width="1389" height="690" alt="image" src="https://github.com/user-attachments/assets/d4dd9ba3-cbc5-4a9e-a82c-c0a1da3aec23" />



### b. Condiciones familiares y sociales
Se estudiaron las relaciones de los menores con el jefe de hogar y las condiciones de afiliación al sistema de salud, con el fin de **comprender el entorno familiar** en el que surge el fenómeno.

### c. Diferencias territoriales
El análisis permitió **identificar desigualdades** significativas entre localidades, sugiriendo concentraciones más altas en ciertas zonas con vulnerabilidad social. Por ejemplo, se muestran algunas graficas a visualizar

<img width="989" height="590" alt="image" src="https://github.com/user-attachments/assets/7edc8962-067d-49e6-af98-d83204161263" />

<img width="923" height="590" alt="image" src="https://github.com/user-attachments/assets/8cf29e82-6d92-4959-944a-0a40248e525a" />

<img width="923" height="590" alt="image" src="https://github.com/user-attachments/assets/4b69a262-ecda-4a03-aadf-a8dbea95715a" />

---

## 5️. Deployment / Desarrollo

Esta fase implicó **la presentación e interpretación final de los resultados** para orientar la toma de decisiones.

### Principales hallazgos
- Mayor concentración de niños trabajadores en **Ciudad Bolívar, Kennedy y Bosa**.  
- Distribución equilibrada por **sexo**.  
- Predominio de **nacionalidad colombiana**, aunque con presencia extranjera en **Candelaria y Los Mártires**.  
- Los **mapas de calor** evidenciaron contrastes territoriales que facilitan la identificación de zonas críticas.

### Conclusión de la fase
Los resultados obtenidos permiten **visualizar diferencias territoriales y demográficas** en la problemática del trabajo infantil, aportando evidencia útil para el diseño de estrategias de prevención y atención focalizada.

---

## 6. Conclusión general

La aplicación de la metodología **CRISP-DM** permitió estructurar un proceso analítico completo, desde la comprensión del problema social hasta la obtención de resultados accionables.  
El proyecto demuestra cómo el tratamiento riguroso de datos puede **apoyar políticas públicas y estrategias sociales** enfocadas en la protección de la niñez en Bogotá.

---

**Autores:** Equipo de análisis y desarrollo, Vanessa Cortes, Paula Guevara y Lina Rozo. 

**Proyecto académico:** Python – Universidad Santo Tomás  
**Herramientas:** Python, pandas, matplotlib, seaborn, Excel

---
