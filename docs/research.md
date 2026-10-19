# Research — base para enseñarle color a la red

1. **O'Donovan et al. *Color Compatibility From Large Datasets*** —
   `dgp.toronto.edu/~donovan/color/colorcomp.pdf`. 104k temas Kuler + 383k COLOURLovers;
   334 features (RGB/CIELab/HSV/CHSV) + regresión de rating. Base del head de
   compatibilidad de G y del feature set de C.
2. **Chou & Wang (2024) *Color Theme Generation Using CNNs* (ACM WSSE)** — Adobe Color
   3.578 imgs → Berlin & Kay → Munsell → MiniBatch-KMeans (456 colores) → ResNet101 por
   categoría + consenso. Referencia del pipeline cura → cuantiza → entrena → consenso.
3. **Bisecting K-Means + CGAN en WikiArt (2024)** — extractor clustering + generador
   neuronal, eval **CIEDE2000** (23.58 vs 24.83 baseline). Nuestra métrica/loss de G.
4. **Review *Color Palette Generation From Digital Images* (Wiley, dic-2024)** —
   taxonomía histograma / clustering / NN (CNN, GAN, atención, autoencoder) + trade-offs.
   Justifica E como soft-KMeans/atención entrenable en TF.
5. **`boltuix/color-pedia` (HuggingFace, ~50k, Parquet, MIT)** — dataset día-0 para G.
6. **Sharma et al. *n-Grams + K-means* (2023, API colourlovers `palettes/top`)** — receta de
   cura: top-1000 × 8 hueOptions ≈ 7k paletas de 5 colores + cuantización KMeans.
7. **QCharm (Springer 2018, espacio Kobayashi + COLOURLovers)** — etiquetas cognitivas;
   aparcado post-MVP.
8. **Fundamentos:** Munsell, Berlin & Kay (11→9 categorías), **CIELAB + CIEDE2000** como
   espacio y métrica perceptual (entrenamos en Lab, nunca en RGB directo).

> Nota de honestidad técnica: la literatura no muestra extractor puramente neuronal que
> supere a KMeans en calidad/velocidad (NeuQuant 1994 sigue siendo la referencia). Por eso
> E es una **cuantización diferenciable destilada de KMeans**: en inferencia no hay sklearn,
> pero en entrenamiento KMeans es la maestra.
