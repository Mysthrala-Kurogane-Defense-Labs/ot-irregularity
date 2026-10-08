# Desarrollo equilibrado por régimen — 2026-10-08

Protocolo b3846d0 registrado en commit antes de generar. Lab limpio y fijado a 718babb7772c3a21f0b87c403f628540cbce58db. Generación completada: nueve fuentes, seeds 930101–930109, 24 runs/fuente, 12 train/calibración y 12 validation/desarrollo, cero test. Cuatro clases por run, una hora por run; ocho regímenes estables y una fuente de transiciones. Licencia de datos CC BY 4.0, Mysthrala Kurogane Defense Labs / OT Irregularity Lab.

Verificada la exposición declarada de cada régimen estable en los manifiestos: 345,600 segundos-activo por fuente. Warmup/cooldown sostenidos son estrés sintético de cobertura, no prevalencia industrial. [Manifiestos resumidos y hashes](../../../results/balanced-development-sources-20261008.json).

Preparación y scoring iniciados desde 60fd471 con CUDA. Verificación de manifiestos/suites/semillas disjuntas antes de scoring; hashes de telemetría, metadata, escenario y truth por run; abortar si hay eventos o particiones de test. Checkpoints por run con hashes y contrato de reanudación que rechaza cambios en código o inputs. No reentrenamiento de redes/scalers. El comparador contextual ampliado está implementado y pendiente de ejecución cuando termine la preparación.

142 tests locales pasan, incluidos rechazos de fuente/licencia/semilla/partición incorrectas. Estado científico pendiente: generación y tests no prueban mejora. Criterios de promoción permanecen en [protocolo](../../../BALANCED_REGIME_PROTOCOL.md); no destilación ni release.

## Resultado de desarrollo

Preparación completada, 216 checkpoints/hashes verificados. Todos los 32 grupos clase/régimen estable alcanzan la cobertura mínima. Candidato ampliado cumple los criterios de desarrollo: conserva todos los eventos del primario, 92/124 eventos y 20/35 físicos, 4 falsas ventanas históricas, 21 en normal-development 920611. PR-AUC .711941 frente a .717892 primario y .708312 clase-only. Conserva intercambio desfavorable respecto a clase-only: 3→4 falsas históricas y 14→21 normales; no ocultarlo.

Validación equilibrada nueva: 25,712 ventanas /428.533 horas-activo. Falsas ventanas primario 160, clase-only 129, contexto anterior 136, contexto ampliado 110. Episodios del ampliado: 108. Tasas por clase: CNC 3.136, compresor 5.601, transportador 8.065, bomba 7.841 por activo-día. No se usaron estas ventanas en calibración. Se selecciona el único candidato declarado para nueva confirmación congelada, no para promoción. [Métricas](../../../results/balanced-context-20261008.json).
