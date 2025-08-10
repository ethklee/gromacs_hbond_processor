# gromacs_hbond_processor

The analysis of hydrogen bonds from molecular dynamics (MD) simulations is surprisingly difficult, even using the extremely useful python packages out there specifically for analyzing MD simulations. For example, the hbond_analysis modules in MDAnalysis is slow and doesn't seem to work half the time.

I find that the quickest and best way to analyze hydrogen bonds is by using GROMACS. Unfortunately, the output of gmx_hbond is pretty confusing and the documentation is lacking. Here, I provide some python functions I have written to parse through the confusing gmx_bond output files and extract key information.

Check out the examples folder for a quick demo.

# Features

- Compute hydrogen bond occupancy statistics from MD simulations
- Parse '.ndx' and '.xpm' output files from gmx_hbond
- Identify hydrogen bond donor and acceptor atoms
- Generate residue-level hydrogen bond maps
- Compare hydrogen bonding statistics across multiple systems (even if they have different structures, ex. mutants)

# Requirements

- Python 3.7+
- numpy
- pandas
