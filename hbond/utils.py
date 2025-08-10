"""
utils.py

Functions for processing hydrogen bond data from GROMACS files.

These functions support common operations like occupancy filtering and atom
property extraction from '.gro' files.

Author: Ethan Lee
Created: 2021-06-19
"""

def cut_occ(occ, cutoff=10):
    """
    Filter hydrogen bonds by occupancy percentage.

    Parameters
    ----------
    occ : array-like
        List or array of hydrogen bond occupancy values (in %).
    cutoff : float, optional
        Occupancy threshold. Bonds with occupancy >= cutoff are kept.

    Returns
    -------
    list of int
        Indices of occupancy values above the cutoff.
    """

    return [i for i, o in enumerate(occ) if o >= cutoff]

def get_gro_data(atom_indices, gro_data):
    """
    Extract properties for a list of atoms from parsed GROMACS .gro data.

    Parameters
    ----------
    atom_indices : list of int
        List of atom indices (1-based, as used in GROMACS).
    gro_data : list of list
        Parsed data from a '.gro' file, where each inner list represents an atom:
        [residue number, residue name, atom name, atom number, x, y, z].

    Returns
    -------
    list of list
        Properties of the specified atoms as sublists:
        [residue number, residue name, atom name, atom number] for each atom.
    """

    return [gro_data[i-1][0:4] for i in atom_indices]

def make_donor_acceptor_columns(atom_properties):
    """
    Construct label strings for donor and acceptor atoms in hydrogen bonds.

    Takes atom metadata (from '.gro' file) for donors, hydrogens, and acceptors,
    and returns formatted strings for use in tables or DataFrames.

    Parameters
    ----------
    atom_properties : list of list of list
        A nested list containing properties of donor, hydrogen, and acceptor atoms,
        in the format:
        [donor_props, hydrogen_props, acceptor_props],
        where each sublist is a list of:
        [residue number, residue name, atom name, atom number].

    Returns
    -------
    tuple of list
        - donor_labels : list of str
            Formatted labels like "RES123 DON@H".
        - acceptor_labels : list of str
            Formatted labels like "RES456 ACC".
    """

    donor_labels = []
    acceptor_labels = []

    donors, hydrogens, acceptors = atom_properties

    # Construct donor and acceptor labels
    for d, h, a in zip(donors, hydrogens, acceptors):
        donor_labels.append(f"{d[1]}{d[0]} {d[2]}@{h[2]}")
        acceptor_labels.append(f"{a[1]}{a[0]} {a[2]}")

    return donor_labels, acceptor_labels