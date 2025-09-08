# creating our circuit fingerprinting dataset

Download and extract the processed dataset from https://github.com/pylls/ol-measurements-and-fp artifacts repo:

```
wget https://dart.cse.kau.se/ol-measurements-and-fp/onionloc.tar.gz
tar -xf onionloc.tar.gz
```

Next, extract circuit kinds (as mapped in the dataset), remove intro and hsdir kinds, move rend circuits to class 1, rename, and pack.

```
python onionfp-files-to-kinds.py
rm -rf results/1 results/2
mv results/3 results/1
mv results circuitfp-general-rend
tar -czf circuitfp-general-rend.tar.gz circuitfp-general-rend
```

