This repo contains
- `MeanFieldTester` a package for testing mean-field models against a spiking neural network
- `drafts` a collections of scripts I have used throughout, some possible drafts etc.



# Install
virtenv `mf-csng`

PyNN
- to test simulators
- have similar implementation to mozaik
nest==3.4
- the one we run simulations with

Brian2==2.8.0

TVB

mozaik
- loading data stores, connectivities etc without switching virtenvs

1. follow [mozaik](https://github.com/csng-mff/mozaik) instructions
    1. pip packages
    2. Imagen
    3. pynn
    4. nest 3.4
    5. mozaik 
2. extra packages
`pip3 install jupyter sympy`
`pip3 install brian2`
`pip3 install moviepy tvb-library tvb-data`


# Notes
wintermute sbatch

run on small nodes
`#SBATCH --exclude=w[9,11,13-17]`

run on big nodes
`#SBATCH --exclude=w[1-8,10,12]`