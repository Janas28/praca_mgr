# Metody Uczenia Maszynowego do Modelowania Oddechu

To repozytorium zawiera kod źródłowy metod i eksperymentów związanych z rozdziałami 3 i 4 pracy magisterskiej zatytułowanej **"Metody uczenia maszynowego do modelowania oddechu"**. Repozytorium zawiera również dane opisane w rozdziałąch 4.1, 4.2 oraz 4.3.Celem tej pracy jest opracowanie skutecznych metod uczenia maszynowego do modelowania ludzkiego oddechu na podstawie sygnałów dźwiękowych oraz ocena ich skuteczności w porównaniu do tradycyjnych podejść. Repozytorium jest zorganizowane w cztery główne foldery, z których każdy odpowiada różnym metodom i eksperymentom.

## Struktura Repozytorium

- **inne_eksperymenty/**  
  Zawiera dodatkowe eksperymenty związane z detekcją oddechu, które zostały przeprowadzone w celu weryfikacji wpływu różnych parametrów na ogólną skuteczność modeli, oraz weryfikacji skuteczność opisanych modeli w pracy z innymi metodami.

- **metoda_detekcji_czestosci_oddechu_ml/**  
  Ten folder zawiera kod źródłowy metody detekcji częstotliwości oddechu opartej na uczeniu maszynowym. Kod odnosi się do metody opisanej w rozdziale 3.2. oraz eksperymentów opisanych w rozdziale 4.5.

- **metoda_detekcji_wartosci_szczytowych_spektrogramu/**  
  Ten folder zawiera implementację metody detekcji wartości szczytowych w analizie spektrogramu. Kod odnosi się do metody opisanej w rozdziale 3.1. oraz eksperymentów opisanych w rozdziale 4.4.

- **metoda_klasyfikacji_cykli_oddechowych_ml/**  
  Ten folder zawiera metodę klasyfikacji cykli oddechowych opartą na uczeniu maszynowym. Podejście to obejmuje wykorzystanie drzew decyzyjnych i sieci neuronowych do klasyfikacji cykli oddechowych na podstawie cech wyodrębnionych z sygnałów dźwiękowych. Kod odnosi się do metody opisanej w rozdziale 3.3. i 3.4. oraz eksperymentów opisanych w rozdziale 4.6, 4.7, 4.8 i 4.9.
