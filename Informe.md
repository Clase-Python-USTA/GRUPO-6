
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

**Distribución del lugar donde ocurre el trabajo infantil**

<img width="774" height="390" alt="image" src="https://github.com/user-attachments/assets/0952f87a-337f-4e21-8aba-4c7685810c27" />

la ocurrencia se concentra en una sola ubicación. “En una UTI” aporta 60.6% de los registros, seguida por “en la calle, estacionario o ambulante” 15.8% y “No especificado” 15.6%; “en la vivienda que habita” 7.8% y el resto de lugares son marginales (~0–0.1%). Implicación: prioriza acciones donde está el volumen (UTI) y un segundo frente en calle/ambulancia; el 15.6% no especificado limita comparaciones y debe depurarse; la fracción en vivienda sugiere intervención familiar y visitas domiciliarias.

**Lugar de trabajo según afiliación al SGSSS**

<img width="871" height="390" alt="image" src="https://github.com/user-attachments/assets/971545ce-43dd-45d3-a02c-4220c5dd138a" />

UTI es el principal entorno en todas las afiliaciones: Contributivo 71%, Especial/Excepción 71%, Vinculado 60%, No especificado 58%, Subsidiado 58% y Sin aseguramiento 45%. La calle crece sobre todo en Vinculado 40% y Sin aseguramiento 33%; es menor en Subsidiado 16% y Contributivo ~9%. El trabajo en vivienda es bajo en todos (0–11%). La categoría “No especificado” mezcla entornos y concentra 29% en “Otros”, por lo que reduce precisión. Conclusión operativa: el foco general es UTI; refuerza intervención en calle para Vinculado y No asegurado.

**Lugar de trabajo según vínculo con el jefe de hogar**

<img width="889" height="765" alt="image" src="https://github.com/user-attachments/assets/885aa8b2-68e5-4074-9dfb-a29d126effc9" />

El entorno dominante es UTI para relaciones nucleares con el jefe del hogar (≈Jefe 84% UTI, Familia 68%, Hermano(a) 67%, Hijo(a) 61%). La calle se dispara en vínculos atípicos: Padre/Madre 82% calle y Nuera/Yerno 50% calle; también sube en Nieto(a) ~23% y Cónyuge ~19%. Los “no parientes” y Abuelo(a) concentran “Otros” (≈53% y 58%). Vivienda es minoritaria en todos (≲20%), con vínculos nucleares el trabajo ocurre sobre todo en UTI; con vínculos no filiales o asimétricos aumenta la exposición en calle. Prioriza intervención en calle para Padre/Madre y Nuera/Yerno y mantén acciones transversales en UTI.

**AFILIACIÓN AL SGSSS × lugar de trabajo — Top 10 combinaciones**

| Afiliación            | Lugar de trabajo                                  | n     |
|----------------------|----------------------------------------------------|-------|
| SUBSIDIADO           | 4. En una UTI                                      | 7,465 |
| CONTRIBUTIVO         | 4. En una UTI                                      | 6,382 |
| SUBSIDIADO           | 2. En la calle, estacionario o ambulante          | 2,368 |
| No especificado      | 4. En una UTI                                      | 2,022 |
| SUBSIDIADO           | No especificado                                    | 1,973 |
| 5- NO ASEGURADO      | 4. En una UTI                                      | 1,850 |
| 5- NO ASEGURADO      | 2. En la calle, estacionario o ambulante          | 1,336 |
| SUBSIDIADO           | 1. En la vivienda que habita                       | 1,003 |
| CONTRIBUTIVO         | No especificado                                    |   991 |
| No especificado      | No especificado                                    |   988 |

- **Subsidiado–UTI:** 7 465 casos (58.2% dentro de subsidiado, 25.1% del total).
- **Contributivo–UTI:** 6 382 (71.5% dentro de contributivo, 21.5% del total).
- **Calle y baja cobertura:** No asegurado–calle 1 336 (32.6% de no asegurado, 4.5% del total) y No asegurado–UTI 1 850 (45.2%, 6.2% del total).
- **En subsidiado la calle es menor:** Subsidiado–calle 2 368 (18.5% de subsidiado, 8.0% del total).
- **Vivienda es marginal:** Subsidiado–vivienda 1 003 (7.8%, 3.37% del total).
- **“No especificado” es relevante:** p. ej., No especificado–UTI 2 022 (6.8% del total), lo que reduce precisión.


**VÍNCULO CON EL JEFE DE HOGAR × lugar de trabajo — Top 10 combinaciones**

| Vínculo            | Lugar de trabajo                                  | n     |
|--------------------|----------------------------------------------------|-------|
| Hijo(a)            | 4. En una UTI                                      | 16,286|
| Hijo(a)            | 2. En la calle, estacionario o ambulante          | 4,169 |
| Hijo(a)            | No especificado                                    | 3,855 |
| Hijo(a)            | 1. En la vivienda que habita                       | 2,159 |
| Nieto(a)           | 4. En una UTI                                      |   605 |
| Otro pariente      | 4. En una UTI                                      |   404 |
| Nieto(a)           | 2. En la calle, estacionario o ambulante          |   239 |
| Familia            | 4. En una UTI                                      |   176 |
| Nieto(a)           | 4. En una UTI                                      |   145 |
| Cónyuge / Otros    | —                                                  |   —   |

- **Hijo(a)–UTI:** 16 286 (61.4% dentro de hijo[a], 54.8% del total).
- **Hijo(a)–calle:** 4 169 (15.7%, 14.0% del total).
- **Hijo(a)–vivienda:** 2 159 (8.1%, 7.26% del total).
- **Otros vínculos aportan poco al total:** Nieto(a)–UTI 605 (2.04%), Otro pariente–UTI 404 (1.36%), Nieto(a)–calle 239 (0.80%).
- **No parientes / otros:** concentran “Otros / No especificado”, no en calle ni en vivienda.

**Conclusión** Las condiciones familiares muestran que el trabajo infantil registrado ocurre mayoritariamente con hijos(as) del jefe y en UTI; las diferencias de acompañamiento indican que la exposición en calle aumenta cuando no hay aseguramiento y existe, en menor medida, aun con subsidio. Prioriza intervención transversal en UTI, y refuerza acciones en calle para no asegurados; mantén el enfoque familiar en hogares con hijos(as) del jefe y mejora el registro de “No especificado” para afinar el diagnóstico.

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
