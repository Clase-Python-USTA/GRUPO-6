# Business Understanding
El presente análisis corresponde a la fase de Data Understanding de la metodología CRISP-DM. Incluye tanto la determinación de objetivos como la exploración inicial de los datos. Esta etapa busca **comprender la estructura, distribución y calidad de la información**.  

## Objetivo de negocio  
El objetivo de negocio es comprender la situación de los niños, niñas y adolescentes en condición de trabajo infantil y adolescente en Bogotá, a partir de la información recolectada en los registros de acompañamiento social, familiar y psicológico, con el fin de contar con una visión clara de cómo se presenta este fenómeno en la ciudad.  

## Objetivos específicos 
- **Caracterizar la población en condición de trabajo infantil/adolescente** en Bogotá según variables demográficas (edad, género, localidad).  
- **Analizar las condiciones familiares y de acompañamiento social** que se reportan en los registros, con el fin de entender mejor el entorno en que se presenta el trabajo infantil.  
- **Identificar diferencias relevantes entre localidades** que permitan dimensionar cómo varía el fenómeno dentro de la ciudad.  
- Disponer de información organizada y visual que facilite la comprensión de la problemática para orientar posibles análisis posteriores o la toma de decisiones.  

## Criterios de éxito de negocio

1. **Cobertura de la información**  
   El proyecto será exitoso si logramos trabajar con la gran mayoría de los registros disponibles.  
   La meta es que al menos el **90% de la información** quede organizada y lista para el análisis,  
   sin datos perdidos o desordenados.  

2. **Entender el entorno familiar y social**  
   Se considerará un buen resultado si, a partir de los registros, podemos identificar factores que  
   influyen en el trabajo infantil y adolescente, como la situación familiar o el acompañamiento social.  
   Será un éxito si esos hallazgos son reconocidos como útiles por personas expertas o instituciones  
   que trabajan en el tema.  

3. **Comparación entre localidades**  
   El proyecto será útil si logramos mostrar cómo varía el trabajo infantil en las distintas localidades  
   de Bogotá. Al menos deberíamos poder señalar **tres diferencias claras y fáciles de entender** entre  
   zonas de la ciudad.  

4. **Presentación clara y visual**  
   Finalmente, un criterio de éxito es que la información no quede solo en tablas complicadas, sino que se presente  
   en gráficos, tableros o informes visuales que permitan comprender el problema de manera rápida y  
   sencilla, incluso para alguien que no sea experto en estadísticas.  

---

## Evaluación de la situación

**Fuente:** `base_datos_completa_nna_ti_anon.xlsx` (5.139.043 filas, 91 variables) + diccionario  
**Trabajo:** GitHub · Visual Studio Code · Python

### Inventario de recursos

**Datos disponibles**

- **Volumen:** 5.139.043 filas y **91 variables**.
- **Composición:** variables **numéricas** (mediciones y conteos), **categóricas** (perfiles, estados, territorios), **de fecha** (intervención, cierre, reposición), **texto/dirección**, **geográficas** (`Coordenadas_X/Y`) e **identificadores**.
- **Cobertura temática:** resultados del caso, salud y alertas, persona y hogar, educación y trabajo, protección social, territorio y operación del programa.


**Herramientas**
- GitHub (versionado), VS Code (edición), Python (limpieza y análisis).

### Requerimientos, presunciones y restricciones

**Requerimientos**
- **PII:** quitar o seudonimizar nombres, documento, teléfonos, correos y dirección; publicar solo agregados.
- **Códigos especiales:** convertir `99999`/“No aplica”/“Desconocido” a `NA`.
- **Dataset analítico reproducible:** selección de variables, tipos y reglas de limpieza en un script Python del repositorio.
- **Coherencia territorial:** revisar `Localidad`, `UPZ`, `Barrio`, `Barrio_priorizado`.

**Restricciones**
- Tamaño (5,1M × 91), catálogos heterogéneos, vacíos en `Edad`, direcciones variables.
- Trabajo centrado en Python dentro del repositorio.


### Riesgos y contingencias
- **PII y datos sensibles →** *Acción:* seudonimizar/eliminar, acceso restringido, publicar agregados.
- **`99999` y vacíos altos →** *Acción:* reglas de limpieza e imputación; categoría “Desconocido”; análisis de sensibilidad.
- **Inconsistencia territorial →** *Acción:* cruces entre `Localidad`/`UPZ`/`Barrio`/`Barrio_priorizado`; priorizar `Localidad`/`UPZ` si hay choque.
- **Duplicados por persona →** *Acción:* revisar con llave (`Tipo_documento`, `Numero_documento`, `Fecha_intervencion`, `Localidad`).
- **Costo computacional →** *Acción:* procesar por partes, leer por columnas, muestrear para pruebas.


### Terminología
- **Desvinculación:** salida del NNA del trabajo (`NNA_desvinculado_de_la_actividad_laboral`).
- **Trabajo protegido:** actividad permitida para adolescentes (`Adolescente_trabajo_protegido`).
- **Ficha de intervención:** registro de atención (`Id_fic`).
- **Acompañamiento / Seguimiento-cierre:** acciones del caso (`Acompanamiento#`) / fin del caso (`Fecha_seguimiento_cierre`).
- **UT:** operador del programa (`Nombre_de_la_UT`).
- **Manzana del cuidado / Barrio priorizado:** etiquetas territoriales.
- **Seudonimización:** reemplazar datos personales por códigos.
- **Dataset analítico:** versión limpia que produce el script.


### Costos previstos
- Limpieza y gobierno del dato (reglas `99999`, tipificación, PII).
- Cómputo y almacenamiento para >5M filas.
- Documentación mínima: *data card*, scripts y tablero simple.

### Beneficios medibles
- Aumento de tasa de desvinculación por `Localidad`/`UPZ` y por `UT`.
- Reducción del tiempo a cierre.
- Priorización territorial con variables categóricas existentes.
  
---

## Objetivo general de la minería de datos 
Explorar y comprender la información registrada en la base de datos sobre trabajo infantil y adolescente en Bogotá, describiendo su distribución según variables sociales, familiares, económicas, personales y de acompañamiento psicosocial, con el propósito de generar un entendimiento inicial para fases posteriores del proceso.  

## Objetivos específicos de la minería de datos   
- Describir las características de la base de datos, identificando el número de registros, las variables disponibles y sus tipos.  
- Identificar patrones y relaciones entre variables relevantes, tales como:  
  - Diferencias por localidad.  
  - Relación entre situación familiar y acompañamiento recibido.  
- Evaluar la calidad de los datos, detectando valores faltantes, inconsistencias, códigos especiales (ej. `9999`, `NA`) y posibles registros duplicados.  
- Generar visualizaciones y reportes descriptivos que permitan orientar las siguientes fases del proceso.  

## Criterios de éxito de la mineria de datos
- Obtener un resumen claro y organizado de los datos (número de registros, tipos de variables y estadísticas básicas).  
- Lograr visualizaciones comprensibles que muestren la distribución del trabajo infantil según variables clave.  
- Identificar patrones que permitan caracterizar el fenómeno en la muestra (ejemplo: diferencias entre localidades o por género).

---

## Plan del proyecto

**Objetivo general**  
Comprender y dimensionar la situación de niños, niñas y adolescentes en trabajo infantil en Bogotá, a partir de registros sociales, familiares y psicológicos, para orientar políticas y programas de protección.

## Etapas principales

1. **Comprensión del negocio**  
   - Revisar objetivos y contexto normativo.  
   - **Salida**: Documento de entendimiento del problema.  

2. **Comprensión y preparación de datos**  
   - Revisar calidad, limpiar e integrar información.  
   - **Salida**: Dataset depurado para análisis.  

3. **Análisis**  
   - Estadística descriptiva, segmentación, factores de riesgo.  
   - **Salida**: Indicadores y modelos preliminares.  

4. **Evaluación**  
   - Validar resultados frente a objetivos.  
   - **Salida**: Revisión y ajustes.  

5. **Presentación y despliegue**  
   - Dashboards, informes y recomendaciones.  
   - **Salida**: Tablero interactivo y reporte final.  

## Herramientas y técnicas

- **Python / R** → limpieza, análisis y modelado.  
- **Power BI / Looker Studio** → visualización de indicadores.  
- **Excel** → validaciones rápidas.  


## Riesgos clave
- **Datos faltantes (>99%)** → Eliminacion.
- **Datos faltantes (~40%)** → imputación y análisis sensible.  
- **Unidad de análisis poco clara** → definir NNA vs evento.   

---
