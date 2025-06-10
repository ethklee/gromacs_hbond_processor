"""
compare.py

Functions for comparing hydrogen bonds across multiple GROMACS simulations.

Includes tools for comparing shared hydrogen bonds, differences between systems
with different '.gro' files, and visualizing occupancy changes.

Author: Ethan Lee
Created: 2021-06-19
"""

import numpy as np
import pandas as pd
from .parser import hbond_ndx_import, gro_import, hbond_xpm_import
from .utils import cut_occ, get_gro_data, make_donor_acceptor_columns

def compare_hbonds(ndx_files, gro_file, xpm_files, cutoff=10):
    """
    Compare hydrogen bonds across multiple GROMACS simulations 
    using the same '.gro' structure file.

    Extracts occupancy data from each system and builds a table showing
    which bonds are present in each and their corresponding occupancy percentages.

    Table is sorted based on occupancy.

    Parameters
    ----------
    ndx_files : list of str
        List of '.ndx' files from 'gmx hbond', one per system.
    gro_file : str
        Path to a single '.gro' structure file shared across all systems.
    xpm_files : list of str
        List of '.xpm' files from 'gmx hbond', one per system.
    cutoff : float, optional
        Minimum occupancy (%) to include a bond from a system (default is 10%).

    Returns
    -------
    pandas.DataFrame
        DataFrame comparing all hydrogen bonds and their occupancies across systems.
        Includes donor/acceptor labels and one column per system.
    """

    # Load ndx, gro, and occupancy data for both systems
    ndx_data = [np.array(hbond_ndx_import(n)[1][-1]).T for n in ndx_files]
    ndx_data_T = [hbond_ndx_import(n)[1][-1] for n in ndx_files]
    gro_data = gro_import(gro_file)
    occupancies = [np.around(np.flip(hbond_xpm_import(x)[2]), 2) for x in xpm_files] # flip values because xpm file returns them backwards

    # Get occupancies over the cutoff
    cutoff_indices = [cut_occ(occ, cutoff=cutoff) for occ in occupancies]

    # Slice donors, hydrogens, and acceptors based on occupancy indices
    sliced_ndx = []
    for i in range(len(ndx_data)):
        sliced_ndx.append([n[cutoff_indices[i]] for n in ndx_data[i]])

    # Concatenate and remove duplicates
    hbond_nodup = []
    for n in sliced_ndx:
        hbond_nodup.append(np.array(n).T)
    hbond_nodup = np.unique(np.concatenate(hbond_nodup), axis=0)

    # Find occupancies
    all_bocc = []
    for bond in hbond_nodup:
        bocc = []
        for i, nd in enumerate(ndx_data_T):
            if bond.tolist() in nd:
                bocc.append(occupancies[i][nd.index(bond.tolist())])
            else:
                bocc.append(0)
        all_bocc.append(bocc)

    # Get atom data from gro file
    atom_properties = [get_gro_data(atom_indices, gro_data) for atom_indices in hbond_nodup.T]

    # Make donor and acceptor columns
    donor, acceptor = make_donor_acceptor_columns(atom_properties)

    # Make dataframe
    final_occ = np.array(all_bocc).T
    d1 = {'Donor': donor,
          'Acceptor': acceptor
         }
    d2 = {'File ' + str(i+1) + ' Occ (%)': oc for (i, oc) in enumerate(final_occ)}
    d3 = {**d1, **d2}
    df = pd.DataFrame(d3)
    df_final = df.sort_values(df.columns[2], ascending=False).reset_index(drop=True)

    return df_final

def compare_hbonds_diff_structures(dfa, dfb):
    """
    Compare hydrogen bonds between two systems using different '.gro' files.

    Matches donor-acceptor pairs by string labels (not atom numbers) assuming
    residue numbering is consistent across systems. Used for comparing systems 
    with minor structural differences (e.g., truncated domains).

    This function is now redundant ever since I wrote
    compare_hbonds_multiple_diff_structures(), which just does what this
    function does but with the capability to compare >2 systems. I am keeping
    this one here anyway in case I ever need to come back and adapt it to do
    something else.

    Parameters
    ----------
    dfa : pandas.DataFrame
        Output from 'hbond_df()' for system 1.
    dfb : pandas.DataFrame
        Output from 'hbond_df()' for system 2.

    Returns
    -------
    pandas.DataFrame
        A comparison table of all hydrogen bonds found in either system,
        with occupancy percentages from both systems.
    """

    # Convert from pandas dataframes to lists
    donors_a = dfa['Donor'].tolist()
    donors_b = dfb['Donor'].tolist()
    acceptors_a = dfa['Acceptor'].tolist()
    acceptors_b = dfb['Acceptor'].tolist()
    occs_a = dfa['Occupancy (%)'].tolist()
    occs_b = dfb['Occupancy (%)'].tolist()

    # Combine donor and acceptor strings
    da_a = [f"{d} {a}" for d, a in zip(donors_a, acceptors_a)]
    da_b = [f"{d} {a}" for d, a in zip(donors_b, acceptors_b)]

    # Add all lines from first dataframe to new list
    combined_da = []
    combined_occs = []
    for i, label in enumerate(da_a):
        combined_da.append(label)
        if label in da_b:
            temp_occ = occs_b[da_b.index(label)]
        else:
            temp_occ = 0

        combined_occs.append([occs_a[i], temp_occ])

    # Add all hbonds in the second dataframe that aren't in the first to the list
    for i, label in enumerate(da_b):
        if label not in da_a:
            combined_da.append(label)
            combined_occs.append([0, occs_b[i]])

    # Create dataframe
    data = {'Donor/Acceptor': combined_da,
            'File 1 Occpuancy (%)': np.array(combined_occs).T[0],
            'File 2 Occpuancy (%)': np.array(combined_occs).T[1]}

    df = pd.DataFrame(data)
    df_final = df.sort_values(df.columns[1], ascending=False).reset_index(drop=True)

    return df_final

def compare_hbonds_multiple_diff_structures(dfs):
    """
    Compare hydrogen bonds across multiple systems with different '.gro' files.

    Combines occupancy data from each input DataFrame, using donor/acceptor
    string labels to match bonds. Assumes residue numbering is consistent
    across systems, even if atom numbering differs.

    Parameters
    ----------
    dfs : list of pandas.DataFrame
        List of DataFrames generated by 'hbond_df()' for different systems.

    Returns
    -------
    pandas.DataFrame
        A table listing all unique hydrogen bonds across systems, with
        occupancy percentages from each file.
    """

    # Convert from pandas dataframes to lists
    donors = []
    acceptors = []
    occs = []
    for d in dfs:
        donors.append(d['Donor'].tolist())
        acceptors.append(d['Acceptor'].tolist())
        occs.append(d['Occupancy (%)'].tolist())

    # Combine donor and acceptor strings
    da_all = []
    for i in range(len(donors)):
        da = []
        for j in range(len(donors[i])):
            da.append(donors[i][j] + ' ' + acceptors[i][j])
        da_all.append(da)

    # Separate first dataframe from the rest
    da_a = da_all[0].copy()
    da_b = [da_all[i].copy() for i in range(1, len(da_all))]
    occs_a = occs[0].copy()
    occs_b = [occs[i].copy() for i in range(1, len(occs))]

    # Add all lines from first dataframe to new list
    new_da = []
    new_occs = []
    for i in range(len(da_a)):
        new_da.append(da_a[i])
        temp_occ = []
        temp_occ.append(occs_a[i])
        for j in range(len(da_b)):
            if da_a[i] in da_b[j]:
                temp_occ.append(occs_b[j][da_b[j].index(da_a[i])])
            else:
                temp_occ.append(0)
        new_occs.append(temp_occ)

    # Add all hbonds from the rest of the dataframes that aren't in the first to the list
    for j in range(len(da_b)):
        for i in range(len(da_b[j])):
            if da_b[j][i] not in new_da:
                new_da.append(da_b[j][i])
                temp_occ = []
                for l in range(j+1):
                    temp_occ.append(0)
                temp_occ.append(occs_b[j][i])
                for k in range(j+1, len(da_b)):
                    if da_b[j][i] in da_b[k]:
                        temp_occ.append(occs_b[k][da_b[k].index(da_b[j][i])])
                    else:
                        temp_occ.append(0)
                new_occs.append(temp_occ)

    # Create dataframe
    d1 = {'Donor/Acceptor': new_da}
    d2 = {'File ' + str(i+1) + ' Occ (%)': oc for (i, oc) in enumerate(np.array(new_occs).T)}
    d3 = {**d1, **d2}
    df = pd.DataFrame(d3)
    df_final = df.sort_values(df.columns[1], ascending=False).reset_index(drop=True)

    return df_final