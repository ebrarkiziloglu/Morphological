
**All evaluations are averaged across 5 seeds.**

## Epoch 0:
epoch | model | Ablative_TR | Instrumental_TR | Locative_TR | Mixed_TR | NaN_prep | dative-case | no-case | other-case | ALL
-- | -- | -- | -- | -- | -- | -- | -- | -- | -- | --
0 | EN | 53.76 | 49.19 | 43.53 | 42.29 | 57.9 | 50.24 | 48.81 | 56.17 | 46.12
0 | EN shuffled | 51.83 | 52.57 | 44.07 | 46.35 | 59.22 | 43.39 | 31.19 | 54.43 | 46.81
0 | TR | 49.9 | 47.54 | 47.83 | 42.63 | 51.98 | 30.94 | 25.52 | 42.95 | 46.36
0 | TR shuffled | 45.65 | 46.47 | 41.79 | 50.25 | 51.26 | 27.95 | 21.19 | 40.77 | 43.96
0 | TR stripped | 50.98 | 52.75 | 48.27 | 40.45 | 53.53 | 34.65 | 25.07 | 46.78 | 47.02
0 | TR stripped & shuffled | 49.41 | 47.93 | 43.16 | 52.91 | 53.35 | 37.01 | 34.78 | 44.04 | 46.32
0 | CH | 53.89 | 48.46 | 48.55 | 41.52 | 55.69 | 38.19 | 30 | 49.18 | 47.61
0 | CH shuffled | 52.13 | 47.75 | 42.95 | 36.55 | 52.4 | 34.17 | 19.25 | 51.15 | 43.27

## Epoch 1:
epoch | model | Ablative_TR | Instrumental_TR | Locative_TR | Mixed_TR | NaN_prep | dative-case | no-case | other-case | ALL
-- | -- | -- | -- | -- | -- | -- | -- | -- | -- | --
1 | DE BabyLM regular (only phase 2) | 86.84 | 96.39 | 73.15 | 88.31 | 61.26 | 33.15 | 45.07 | 36.07 | 77.26
1 | DE BabyLM shifted (only phase 2) | 87.12 | 95.7 | 72.23 | 86.24 | 61.08 | 31.02 | 42.39 | 34.32 | 76.25
1 | EN + DE NLLB | 51.96 | 49.92 | 42.65 | 43.36 | 57.84 | 45.59 | 46.42 | 54.32 | 45.53
1 | EN + reset_embeddings + DE NLLB | 41.83 | 41.12 | 46.91 | 44.53 | 42.81 | 51.1 | 58.21 | 41.64 | 45.31
1 | EN + reset_embeddings + DE NLLB shifted | 45.55 | 46.7 | 49.7 | 46.52 | 46.17 | 45.59 | 48.81 | 46.89 | 48
1 | EN + DE NLLB shifted | 47.36 | 48.11 | 50.24 | 62.43 | 42.51 | 49.92 | 61.94 | 38.47 | 51.63
1 | TR + DE NLLB | 50.59 | 50.83 | 47.44 | 43.55 | 55.87 | 31.26 | 22.84 | 47.32 | 46.93
1 | TR + reset_embeddings + DE NLLB | 42.86 | 43.47 | 50.29 | 46.24 | 45.57 | 50.55 | 53.13 | 44.92 | 47.73
1 | TR + reset_embeddings + DE NLLB shifted | 41.15 | 39.34 | 41.82 | 41.7 | 42.04 | 42.13 | 41.34 | 46.34 | 41.58
1 | TR + DE NLLB shifted | 53.01 | 53.67 | 41.08 | 52.03 | 49.22 | 58.03 | 68.96 | 47.87 | 47.1
1 | TR shuffled + DE NLLB | 46.72 | 47.31 | 41.63 | 52.62 | 53.77 | 28.19 | 20.9 | 43.83 | 44.68
1 | TR stripped + DE NLLB | 52.84 | 54.54 | 48.68 | 40.88 | 57.01 | 35.35 | 21.94 | 50.93 | 47.88
1 | TR stripped + reset_embeddings + DE NLLB | 44.04 | 43.36 | 49.91 | 45.37 | 46.41 | 44.8 | 51.19 | 44.81 | 47.36
1 | CH + DE NLLB | 54.03 | 49.74 | 47.18 | 43.94 | 55.03 | 37.8 | 23.73 | 50.38 | 47.42
1 | CH + reset_embeddings + DE NLLB | 52.27 | 50.97 | 46.93 | 49.93 | 51.08 | 49.92 | 49.25 | 47.1 | 48.81
 
## Epoch 3:
epoch | model | Ablative_TR | Instrumental_TR | Locative_TR | Mixed_TR | NaN_prep | dative-case | no-case | other-case | ALL
-- | -- | -- | -- | -- | -- | -- | -- | -- | -- | --
3 | DE BabyLM regular (only phase 2) | 91.12 | 97.74 | 78.28 | 94.9 | 69.4 | 34.65 | 54.33 | 31.26 | 82.08
3 | DE BabyLM shifted (only phase 2) | 90.85 | 97.69 | 78.36 | 94.43 | 69.88 | 32.05 | 57.01 | 32.24 | 82
  |   |   |   |   |   |   |   |   |   |  
  |   |   |   |   |   |   |   |   |   |  
3 | EN + DE NLLB | 48.19 | 47.61 | 36.72 | 42.25 | 48.26 | 34.57 | 38.66 | 46.78 | 40.79
3 | EN + reset_embeddings + DE NLLB | 38.12 | 39.02 | 46.26 | 43.69 | 43.23 | 50.87 | 59.4 | 41.09 | 44.18
3 | EN + reset_embeddings + DE NLLB shifted | 43.47 | 44.14 | 45.51 | 42.75 | 43.05 | 41.5 | 43.43 | 43.28 | 44.33
3 | EN + DE NLLB shifted | 59.48 | 62.45 | 56.34 | 62.57 | 57.54 | 63.39 | 77.46 | 58.58 | 59.05
3 | TR + DE NLLB | 55.19 | 56.57 | 47.74 | 51.02 | 61.2 | 41.89 | 34.63 | 55.96 | 50.41
3 | TR + reset_embeddings + DE NLLB | 45.18 | 46.08 | 52.36 | 49.29 | 49.34 | 54.02 | 57.31 | 49.51 | 50.25
3 | TR + reset_embeddings + DE NLLB shifted | 40.24 | 37.9 | 44.72 | 43.2 | 41.92 | 42.44 | 45.37 | 43.83 | 43.06
3 | TR + DE NLLB shifted | 53.13 | 56.91 | 39.76 | 45.22 | 49.1 | 61.73 | 77.01 | 52.02 | 45.77
3 | TR shuffled + DE NLLB | 54.5 | 53.99 | 46.02 | 57.57 | 60.18 | 40 | 37.01 | 57.6 | 50.44
3 | TR stripped + DE NLLB | 59.85 | 60.8 | 49.89 | 49.6 | 63.77 | 45.91 | 37.16 | 63.17 | 52.57
3 | TR stripped + reset_embeddings + DE NLLB | 46.65 | 46.38 | 48.48 | 45.89 | 46.89 | 45.98 | 48.06 | 43.83 | 47.34
3 | CH + DE NLLB | 55.13 | 56.11 | 47.08 | 48.64 | 56.71 | 42.52 | 23.58 | 54.1 | 49.25
3 | CH + reset_embeddings + DE NLLB | 53.47 | 52.66 | 47.1 | 49.6 | 53.23 | 49.37 | 48.06 | 48.74 | 49.22

## Epoch 5:
epoch | model | Ablative_TR | Instrumental_TR | Locative_TR | Mixed_TR | NaN_prep | dative-case | no-case | other-case | ALL
-- | -- | -- | -- | -- | -- | -- | -- | -- | -- | --
5 | DE BabyLM regular (only phase 2) | 92.66 | 98.06 | 79.46 | 95.46 | 73.23 | 36.69 | 60.75 | 34.86 | 83.35
5 | DE BabyLM shifted (only phase 2) | 92.66 | 98.17 | 79.92 | 95.31 | 72.46 | 35.67 | 60.75 | 35.96 | 83.53
  |   |   |   |   |   |   |   |   |   |  
  |   |   |   |   |   |   |   |   |   |  
5 | EN + DE NLLB | 46.51 | 45.49 | 34.76 | 41.23 | 44.79 | 33.07 | 34.93 | 43.28 | 38.93
5 | EN + reset_embeddings + DE NLLB | 37.6 | 38.83 | 46.01 | 43.28 | 43.29 | 50.08 | 59.1 | 41.31 | 43.88
5 | EN + reset_embeddings + DE NLLB shifted | 43.1 | 43.66 | 44.64 | 42.04 | 42.63 | 40.47 | 41.79 | 42.84 | 43.61
5 | EN + DE NLLB shifted | 62.01 | 66.22 | 57.3 | 62.86 | 60.9 | 68.66 | 81.04 | 63.93 | 60.67
5 | TR + DE NLLB | 56.55 | 57.37 | 47.81 | 52.77 | 61.98 | 44.09 | 37.61 | 57.81 | 51.19
5 | TR + reset_embeddings + DE NLLB | 47.01 | 48 | 53.21 | 50.22 | 50.66 | 54.17 | 57.01 | 50.6 | 51.32
5 | TR + reset_embeddings + DE NLLB shifted | 40.52 | 37.9 | 45.76 | 43.72 | 41.98 | 42.99 | 45.97 | 43.5 | 43.74
5 | TR + DE NLLB shifted | 52.55 | 57.74 | 40.11 | 44.45 | 49.7 | 62.13 | 77.76 | 53.77 | 45.88
5 | TR shuffled + DE NLLB | 55.63 | 54.65 | 46.65 | 57.8 | 60.96 | 42.91 | 42.09 | 60.55 | 51.24
5 | TR stripped + DE NLLB | 60.85 | 61.23 | 49.76 | 51.23 | 63.89 | 49.37 | 40.9 | 65.14 | 53.17
5 | TR stripped + reset_embeddings + DE NLLB | 47.43 | 47.25 | 48.68 | 46.16 | 47.19 | 46.38 | 48.06 | 43.83 | 47.69
5 | CH + DE NLLB | 55.92 | 57.51 | 47.98 | 48.98 | 58.2 | 43.62 | 25.97 | 56.28 | 50.15
5 | CH + reset_embeddings + DE NLLB | 53.59 | 52.8 | 47.27 | 49.72 | 53.05 | 48.98 | 47.46 | 49.18 | 49.34

# Different Configs:

## Config 1: Regular L1 + regular DE --> no modifications to the training

epoch | model | Ablative_TR | Instrumental_TR | Locative_TR | Mixed_TR | NaN_prep | dative-case | no-case | other-case | ALL
-- | -- | -- | -- | -- | -- | -- | -- | -- | -- | --
1 | EN + DE NLLB | 51.96 | 49.92 | 42.65 | 43.36 | 57.84 | 45.59 | 46.42 | 54.32 | 45.53
1 | TR + DE NLLB | 50.59 | 50.83 | 47.44 | 43.55 | 55.87 | 31.26 | 22.84 | 47.32 | 46.93
1 | TR stripped + DE NLLB | 52.84 | 54.54 | 48.68 | 40.88 | 57.01 | 35.35 | 21.94 | 50.93 | 47.88
1 | CH + DE NLLB | 54.03 | 49.74 | 47.18 | 43.94 | 55.03 | 37.8 | 23.73 | 50.38 | 47.42
3 | EN + DE NLLB | 48.19 | 47.61 | 36.72 | 42.25 | 48.26 | 34.57 | 38.66 | 46.78 | 40.79
3 | TR + DE NLLB | 55.19 | 56.57 | 47.74 | 51.02 | 61.2 | 41.89 | 34.63 | 55.96 | 50.41
3 | TR stripped + DE NLLB | 59.85 | 60.8 | 49.89 | 49.6 | 63.77 | 45.91 | 37.16 | 63.17 | 52.57
3 | CH + DE NLLB | 55.13 | 56.11 | 47.08 | 48.64 | 56.71 | 42.52 | 23.58 | 54.1 | 49.25
5 | EN + DE NLLB | 46.51 | 45.49 | 34.76 | 41.23 | 44.79 | 33.07 | 34.93 | 43.28 | 38.93
5 | TR + DE NLLB | 56.55 | 57.37 | 47.81 | 52.77 | 61.98 | 44.09 | 37.61 | 57.81 | 51.19
5 | TR stripped + DE NLLB | 60.85 | 61.23 | 49.76 | 51.23 | 63.89 | 49.37 | 40.9 | 65.14 | 53.17
5 | CH + DE NLLB | 55.92 | 57.51 | 47.98 | 48.98 | 58.2 | 43.62 | 25.97 | 56.28 | 50.15
  |   |   |   |   |   |   |   |   |   |  
  |   |   |   |   |   |   |   |   |   |  

## Config 2: Regular L1 + Embed. reset + regular DE
epoch | model | Ablative_TR | Instrumental_TR | Locative_TR | Mixed_TR | NaN_prep | dative-case | no-case | other-case | ALL
-- | -- | -- | -- | -- | -- | -- | -- | -- | -- | --
1 | EN + reset_embeddings + DE NLLB | 41.83 | 41.12 | 46.91 | 44.53 | 42.81 | 51.1 | 58.21 | 41.64 | 45.31
1 | TR + reset_embeddings + DE NLLB | 42.86 | 43.47 | 50.29 | 46.24 | 45.57 | 50.55 | 53.13 | 44.92 | 47.73
1 | TR stripped + reset_embeddings + DE NLLB | 44.04 | 43.36 | 49.91 | 45.37 | 46.41 | 44.8 | 51.19 | 44.81 | 47.36
1 | CH + reset_embeddings + DE NLLB | 52.27 | 50.97 | 46.93 | 49.93 | 51.08 | 49.92 | 49.25 | 47.1 | 48.81
3 | EN + reset_embeddings + DE NLLB | 38.12 | 39.02 | 46.26 | 43.69 | 43.23 | 50.87 | 59.4 | 41.09 | 44.18
3 | TR + reset_embeddings + DE NLLB | 45.18 | 46.08 | 52.36 | 49.29 | 49.34 | 54.02 | 57.31 | 49.51 | 50.25
3 | TR stripped + reset_embeddings + DE NLLB | 46.65 | 46.38 | 48.48 | 45.89 | 46.89 | 45.98 | 48.06 | 43.83 | 47.34
3 | CH + reset_embeddings + DE NLLB | 53.47 | 52.66 | 47.1 | 49.6 | 53.23 | 49.37 | 48.06 | 48.74 | 49.22
5 | EN + reset_embeddings + DE NLLB | 37.6 | 38.83 | 46.01 | 43.28 | 43.29 | 50.08 | 59.1 | 41.31 | 43.88
5 | TR + reset_embeddings + DE NLLB | 47.01 | 48 | 53.21 | 50.22 | 50.66 | 54.17 | 57.01 | 50.6 | 51.32
5 | TR stripped + reset_embeddings + DE NLLB | 47.43 | 47.25 | 48.68 | 46.16 | 47.19 | 46.38 | 48.06 | 43.83 | 47.69
5 | CH + reset_embeddings + DE NLLB | 53.59 | 52.8 | 47.27 | 49.72 | 53.05 | 48.98 | 47.46 | 49.18 | 49.34
  |   |   |   |   |   |   |   |   |   |  

## Config 3: Regular L1 + shifted DE
epoch | model | Ablative_TR | Instrumental_TR | Locative_TR | Mixed_TR | NaN_prep | dative-case | no-case | other-case | ALL
-- | -- | -- | -- | -- | -- | -- | -- | -- | -- | --
1 | EN + DE NLLB shifted | 47.36 | 48.11 | 50.24 | 62.43 | 42.51 | 49.92 | 61.94 | 38.47 | 51.63
1 | TR + DE NLLB shifted | 53.01 | 53.67 | 41.08 | 52.03 | 49.22 | 58.03 | 68.96 | 47.87 | 47.1
3 | EN + DE NLLB shifted | 59.48 | 62.45 | 56.34 | 62.57 | 57.54 | 63.39 | 77.46 | 58.58 | 59.05
3 | TR + DE NLLB shifted | 53.13 | 56.91 | 39.76 | 45.22 | 49.1 | 61.73 | 77.01 | 52.02 | 45.77
5 | EN + DE NLLB shifted | 62.01 | 66.22 | 57.3 | 62.86 | 60.9 | 68.66 | 81.04 | 63.93 | 60.67
5 | TR + DE NLLB shifted | 52.55 | 57.74 | 40.11 | 44.45 | 49.7 | 62.13 | 77.76 | 53.77 | 45.88
  |   |   |   |   |   |   |   |   |   |  

## Config 4: Regular L1 + Embed. reset + shifted DE
epoch | model | Ablative_TR | Instrumental_TR | Locative_TR | Mixed_TR | NaN_prep | dative-case | no-case | other-case | ALL
-- | -- | -- | -- | -- | -- | -- | -- | -- | -- | --
1 | EN + reset_embeddings + DE NLLB shifted | 45.55 | 46.7 | 49.7 | 46.52 | 46.17 | 45.59 | 48.81 | 46.89 | 48
1 | TR + reset_embeddings + DE NLLB shifted | 41.15 | 39.34 | 41.82 | 41.7 | 42.04 | 42.13 | 41.34 | 46.34 | 41.58
3 | EN + reset_embeddings + DE NLLB shifted | 43.47 | 44.14 | 45.51 | 42.75 | 43.05 | 41.5 | 43.43 | 43.28 | 44.33
3 | TR + reset_embeddings + DE NLLB shifted | 40.24 | 37.9 | 44.72 | 43.2 | 41.92 | 42.44 | 45.37 | 43.83 | 43.06
5 | EN + reset_embeddings + DE NLLB shifted | 43.1 | 43.66 | 44.64 | 42.04 | 42.63 | 40.47 | 41.79 | 42.84 | 43.61
5 | TR + reset_embeddings + DE NLLB shifted | 40.52 | 37.9 | 45.76 | 43.72 | 41.98 | 42.99 | 45.97 | 43.5 | 43.74

## Config 5: Shuffled L1 + regular DE

