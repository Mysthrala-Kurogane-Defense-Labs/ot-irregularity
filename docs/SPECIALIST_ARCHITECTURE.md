# Especialistas de irregularidad: decisión de arquitectura

Estado: propuesta de implementación autorizada el 2026-10-08; no es un modelo validado ni un cambio al candidato congelado.

## Objetivo

Especializar por evidencia observable, no por avería supuesta. Mantener un detector general como referencia y conservar scores individuales. Los especialistas iniciales aprovechan componentes existentes:

- Integridad: calidad, cobertura, cadencia y pérdida de señal. TelemetryHealthReference ya separa canales y causas de indisponibilidad. La repetición sigue desactivada hasta demostrar utilidad sin falsas alarmas excesivas.
- Relaciones físicas: residuos multivariables condicionados por clase; RelationshipResiduals y ClassMagnitudeEnsemble. Identifica novedad en relaciones, no causas mecánicas.
- Dinámica térmica: evolución causal respecto a carga/contexto. ThermalDynamics y ensayos sample-first son investigación; no activarlos por defecto con evidencia insuficiente.
- Contexto operacional: régimen y transiciones normales. Primera fase: contexto de los otros especialistas y diagnóstico de cobertura; un detector independiente requiere justificar ganancia incremental.
- Detector general AE/Isolation Forest: mantener scores y referencia para anomalías no cubiertas. No asumir que añadir su voto mejora al ensemble.

## Contrato común

Cada resultado debe identificar run, activo, clase, ventana, especialista y versión. Incluir status available/unavailable/out_of_scope, reason, raw_score, calibrated_score, calibration_version, threshold, decisión, observaciones y contribuciones. Score calibrado no equivale a probabilidad de fallo. Ningún valor ausente se convierte en normalidad.

Cada artefacto declara roles, unidades, cadencia, calidad mínima, contexto causal y clases soportadas. Scaler y calibrador pertenecen a su especialista. La normalización semántica sigue siendo una capa común anterior. Cambios de régimen desconocidos no autorizan inventar una clase ni ajustar sobre la marcha.

La primera integración debe envolver los componentes existentes y reproducir sus salidas, sin reentrenarlos ni cambiar umbrales. Mantener la identidad de ventanas; rechazar duplicados o mezclas de versiones. Conservar contribuciones por especialista; no presentar errores de AE como causalidad de Isolation Forest.

## Combinación y estado de alarma

Conservar todas las salidas aunque exista un score agregado. Configurar qué especialistas participan, pesos/umbrales y tratamiento explícito de indisponibilidad. No renormalizar silenciosamente pesos cuando falta un especialista. Publicar cobertura de evaluación junto a la decisión.

La política actual congelada continúa intacta hasta terminar su confirmación. Una nueva unión necesita medir falsas alarmas conjuntas, no solo las de cada componente. Seleccionar combinaciones únicamente en desarrollo, con ablation de cada especialista y presupuesto de falsas alarmas por clase. Persistencia e histéresis son una política posterior, versionada por separado, con tests de eventos breves y latencia. Integridad urgente no hereda automáticamente retardos físicos.

Streaming requiere estado por run/activo y un reloj de evaluación: el silencio final no puede depender de la llegada de otra muestra. Reinicios, datos tardíos y cambio de cadencia necesitan contratos explícitos antes de afirmar soporte continuo.

## Evaluación y promoción

Primero completar el test congelado existente. Después usar desarrollo para determinar errores y un candidato nuevo; excluir todos los tests consumidos. Comparar generalista, cada especialista y ensemble sobre exactamente las mismas ventanas e intervalos originales. Reportar detección por familia, pérdidas exactas, PR-AUC, precisión/recall, falsas ventanas y episodios por exposición, latencia y disponibilidad; bootstrap por run y variación por semillas.

Todo cambio necesita superar los criterios declarados sin esconder regresiones en clases o eventos. Un specialist que solo mejora su familia pero rompe el presupuesto global no se activa. El modelo público debe funcionar sin infraestructura privada y serializar todos los componentes necesarios desde la CLI.

## Destilación, condicionada a evidencia

No comenzar hasta tener un teacher estable y útil en confirmación independiente. Entrenar un student con objetivos múltiples para reproducir scores y disponibilidad de especialistas, además de su score general; un único escalar perdería parte de la explicabilidad. No usar test/challenge para etiquetas teacher de entrenamiento. Separar datos de ajuste, calibración, selección y confirmación también durante destilación.

Comparar student con teacher y generalista en detecciones exactas, falsas alarmas, ranking, latencia de inferencia, memoria y cobertura. La compresión solo se acepta si cumple tolerancias declaradas antes del test. No asumir que puede comprimir en una red sin estado la lógica de calidad/cadencia/reloj. Mantener esos controles deterministas fuera del student cuando su contrato lo exija.

## Riesgos y secuencia

Riesgos: más especialistas pueden sumar falsas alarmas, la especialización puede sobreajustar al simulador, y el student puede copiar sesgos/errores del teacher. Mitigar con fuentes diversas, contexto normal, evidencia por clase y datos independientes; más parámetros no prueba mejora.

Secuencia: confirmación congelada → contrato y adaptadores con paridad exacta → diagnóstico de cobertura y especialista térmico/contextual acotado → nueva selección/confirmación → bundle/CLI → validación industrial → decidir si destilar. La política MKDL mantiene bloqueada la entrega afectada mientras haya hallazgos requeridos sin resolver.
