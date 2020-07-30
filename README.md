# exploradome_tangram
Tangram form detection from live video stream

## Approach

TBD
Keywords : Hu Moments, distances

## Testing

### Unit Tests

We test each function individually

### Integration Tests

#### Static tests
We compare an image with our dataset. We need several images by class.

#### Video tests
We compare a video with our dataset. We need different sequences with a clear solution


## TODOS

### Récupération des caractéristiques des images
- [x] fonction pour récupérer les Hu moments
- [x] récupération des Hu moments dataset tangrams
- [x] processing des images pour récupérer leur contour
- [x] récupérer le nombre de sommets dans une image
- [ ] utiliser les sommets pour déterminer si on peut calculer les probabilités - Nicolas, Bastien
- [x] écriture des fonctions de calcul de distances
- [ ] convertir distances en probabilités - Gauthier

### Prédictions
- [x] affichage des prédictions en temps réel

### Tests
- [ ] tests unitaires - Renata
- [x] stratégie de test
- [ ] optimisation du code - Nohossat
- [ ] découpage des videos - Nohossat
- [ ] tests d'intégration - TBD

### Version control
- [ ] ecriture du readme - TBD
- [ ] ré-organisation des fichiers dans le GitHub - Nohossat


[Dataset](https://drive.google.com/drive/folders/1pmuPaserBOOIrdrdmM8uy592v4ylJlHx?usp=sharing)
