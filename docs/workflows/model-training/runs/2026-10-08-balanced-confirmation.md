# Confirmación independiente del contexto equilibrado — 2026-10-08

## Alcance y procedencia

Candidato único seleccionado en desarrollo: mediana por clase con referencia contextual normal ampliada; baseline y canal de integridad conservados. Seis comparadores, mismos inputs y ventanas. Protocolo en `docs/BALANCED_CONTEXT_CONFIRMATION_PROTOCOL.md`; ninguna recalibración sobre test.

Lab 718babb7772c3a21f0b87c403f628540cbce58db, versión 0.6.0. Fuentes pseudo-sintéticas CC BY 4.0, atribución Mysthrala Kurogane Defense Labs / [OT Irregularity Lab](https://github.com/Mysthrala-Kurogane-Defense-Labs/ot-irregularity-lab). Mixtas 910571–910573: 240 runs generados por fuente, solo 36 test evaluados por fuente. Normal 920641: 180 runs de una hora. Se excluyeron 7.776 semillas de runs previos; todos los tests de esta confirmación quedan consumidos.

## Reparación operativa y reproducibilidad

Freeze original 0a3644d anterior a generación. El primer intento falló al serializar progreso tras guardar un run: el vector contextual sobrescribía `total`, contador de runs. Corrección d1a8574: renombrar exclusivamente el vector a `context_combined`; misma fórmula. No había métricas agregadas. Se conserva el intento original y se repite desde cero en un directorio nuevo, con freeze enmendado que enlaza el hash original y registra el único archivo de ejecución modificado. No se presenta esta enmienda como previa a generar las fuentes.

El primer run repetido coincide exactamente: 18 ventanas y archivo Parquet byte a byte. La reanudación completa de los 288 checkpoints terminó con resultados byte a byte idénticos (SHA-256 f66d2b295192a36c38ea7c570d87c2d3a41f7dbd14cbc7ef1695d392b92fef28). 144 tests locales pasan tras la corrección. El [resumen de freeze](../../../results/balanced-confirmation-freeze-20261008.json) conserva hashes de modelos, referencias, runtime, protocolos, fuentes previas y enmienda.

El protocolo contiene una descripción heredada confusa en su quinto control: el quinto es la mediana **por clase sin contexto** y el sexto es el único candidato contextual ampliado, tal como especifica su sección final y ejecuta el código congelado. Se conserva el texto congelado para no cambiar su hash durante la evaluación.

## Reproducción

Usar checkout d1a8574 y el entorno CUDA del experimento. Los argumentos siguientes son directorios portables, no rutas internas:

```powershell
$env:PYTHONPATH = "$PWD/src"
$env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'
python scripts/evaluate_consensus.py `
  --output <directorio-con-freeze-enmendado-y-referencias> `
  --baseline <baseline-v04> `
  --relationships <tres-miembros-congelados> `
  --health-reference <health-reference.json> `
  --sources <holdout-910571> <holdout-910572> <holdout-910573> <normal-exposure-920641> `
  --device cuda:0
```

El evaluador verifica hashes del código y modelos, procedencia y separación de semillas; solo reanuda checkpoints con hashes idénticos. Los artefactos grandes y fuentes no se incluyen en este PR, por lo que estos comandos requieren obtener o regenerar los inputs exactos. Los scores son rankings, no probabilidades de avería. Estas pruebas no validan operación de planta ni silencio final en streaming.

## Resultado

288 runs completados; 1.611 ventanas mixtas, 103 eventos originales (25 físicos), 42.872 ventanas normales y 714,533 horas-activo normales, 178,633 por clase.

| Indicador | Primario con integridad | Clase sola | Contexto ampliado |
| --- | ---: | ---: | ---: |
| Eventos | 75/103 | 75/103 | 75/103 |
| Eventos físicos | 13/25 | 13/25 | 13/25 |
| Falsas ventanas mixtas | 4 | 5 | 5 |
| PR-AUC | .689247 | .678614 | .678994 |
| Precisión | .964286 | .955752 | .955752 |
| Recall por ventana | .542714 | .542714 | .542714 |
| Falsas ventanas normales | 150 | 93 | 110 |
| Falsas ventanas/activo-día normal | 5.0383 | 3.1237 | 3.6947 |

**Candidato rechazado para promoción:** pasa normalidad, falla el criterio mixto por cinco falsas ventanas frente a cuatro y pérdida de una detección de integridad de v0.4. No pierde eventos del primario con integridad ni físicos requeridos. La pérdida respecto a v0.4 es `seed-910572::test-00003/PUMP-01/evt-1`, familia de ground truth `sensor_drift`, intervalo [1735689702000000, 1735689858000000). No se usa este caso para ajustar umbrales o elegir una nueva combinación.

Normalidad: CNC 2,0153; compresor 6,8520; transportador 2,5527; bomba 3,3588 falsas ventanas/activo-día. El descenso global frente al primario tiene IC95% pareado [-2,2146; -0,5042]/día. No domina al comparador por clase: 110 frente a 93 falsas ventanas con igual detección. Tampoco demuestra mejora de ranking: ΔPR-AUC IC95% [-.032696; .011811], incluye cero. Δdetección total y física [0,0] porque las decisiones de eventos coinciden.

Disponibilidad explícita: 38 ventanas mixtas sin especialista relacional; contexto usado en 855 mixtas y 20.248 normales, fallback por contexto desconocido en 718 mixtas y 22.624 normales. La auditoría verificó los 288 hashes de predicción/checkpoint y el enlace al freeze. [Métricas completas](../../../results/balanced-confirmation-20261008.json), [auditoría](../../../results/balanced-confirmation-audit-20261008.json).

## Decisión y trabajo restante

Conservar los componentes como investigación y no sustituir el modelo público ni destilar este teacher. La cobertura normal de 32 grupos se ha ampliado, pero eso no prueba ganancia de detección. La siguiente hipótesis necesita desarrollo separado: detectar deriva lenta y dinámica térmica mediante evidencia temporal específica, conservando integridad y un presupuesto conjunto de falsas alarmas. No escoger una unión sobre estos resultados: la unión ya falló en desarrollo. Nuevos candidatos necesitan nueva confirmación; estas cuatro fuentes están consumidas.

La corrección de ejecución fue revisada mediante diff: solo cambia el nombre local del vector contextual, no la fórmula, referencias o políticas. Tests y auditoría local no equivalen a CI remoto, release ni validación industrial.
