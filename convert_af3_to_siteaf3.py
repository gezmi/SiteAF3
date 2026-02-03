#!/usr/bin/env python3
"""Convert AlphaFold3 JSON input to SiteAF3 JSON format."""

import argparse
import json
import sys


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert AlphaFold3 JSON input to SiteAF3 JSON format."
    )
    parser.add_argument("--input", required=True, help="AF3 JSON input file")
    parser.add_argument("--output", required=True, help="SiteAF3 JSON output file")
    parser.add_argument(
        "--rec-struct-path", required=True, help="Path to receptor PDB structure"
    )
    parser.add_argument(
        "--receptor-chains",
        default="A",
        help="Comma-separated receptor chain IDs (default: A)",
    )
    parser.add_argument(
        "--ligand-chains",
        default=None,
        help="Comma-separated ligand chain IDs (default: all non-receptor chains)",
    )

    msa_group = parser.add_mutually_exclusive_group()
    msa_group.add_argument(
        "--keep-msa", action="store_true", default=True, help="Keep MSA data (default)"
    )
    msa_group.add_argument(
        "--remove-msa", action="store_true", help="Strip all MSA data"
    )

    template_group = parser.add_mutually_exclusive_group()
    template_group.add_argument(
        "--keep-templates",
        action="store_true",
        default=True,
        help="Keep template data (default)",
    )
    template_group.add_argument(
        "--remove-templates", action="store_true", help="Strip all template data"
    )

    parser.add_argument("--pocket-path", default=None, help="Pocket PDB path")
    parser.add_argument("--hotspot-path", default=None, help="Hotspot PDB path")
    parser.add_argument("--name", default=None, help="Override case name")
    parser.add_argument(
        "--seeds", default=None, help="Override model seeds (comma-separated ints)"
    )
    return parser.parse_args()


def extract_chain_ids(sequence_entry):
    """Extract chain IDs from an AF3 sequence entry. Handles both str and list ids."""
    for mol_type in ("protein", "rna", "dna", "ligand"):
        if mol_type in sequence_entry:
            raw_id = sequence_entry[mol_type].get("id")
            if isinstance(raw_id, list):
                return raw_id
            elif raw_id is not None:
                return [raw_id]
    return []


def extract_all_chain_ids(af3_data):
    """Return the set of all chain IDs across all AF3 sequences."""
    ids = set()
    for entry in af3_data.get("sequences", []):
        ids.update(extract_chain_ids(entry))
    return ids


def _af3_type(entry):
    """Return the AF3 molecule type key for a sequence entry."""
    for mol_type in ("protein", "rna", "dna", "ligand"):
        if mol_type in entry:
            return mol_type
    return None


def _build_ligand_entry(mol_type, mol_data, chain_id, remove_msa, remove_templates):
    """Build a single SiteAF3 ligand entry from AF3 sequence data."""
    # Map AF3 "ligand" type -> SiteAF3 "small_molecule"
    out_type = "small_molecule" if mol_type == "ligand" else mol_type

    inner = {"id": chain_id}

    if out_type == "small_molecule":
        if "ccdCodes" in mol_data:
            inner["ccdCodes"] = mol_data["ccdCodes"]
        if "smiles" in mol_data:
            inner["smiles"] = mol_data["smiles"]
    else:
        # protein, rna, dna — copy sequence
        if "sequence" in mol_data:
            inner["sequence"] = mol_data["sequence"]

    # MSA fields (only relevant for protein and rna)
    if out_type in ("protein", "rna"):
        if remove_msa:
            if out_type == "protein":
                inner["pairedMsa"] = ""
            # SiteAF3 uses snake_case for ligand unpaired_msa
            inner["unpaired_msa"] = ""
        else:
            if "unpairedMsa" in mol_data:
                inner["unpaired_msa"] = mol_data["unpairedMsa"]
            if "pairedMsa" in mol_data:
                inner["pairedMsa"] = mol_data["pairedMsa"]

    # Templates (protein only)
    if out_type == "protein":
        if remove_templates:
            inner["templates"] = []
        elif "templates" in mol_data:
            inner["templates"] = mol_data["templates"]

    # Modifications — pass through if present
    if "modifications" in mol_data:
        inner["modifications"] = mol_data["modifications"]

    return {out_type: inner}


def convert(af3_data, receptor_chains, ligand_chains, rec_struct_path,
            pocket_path, hotspot_path, remove_msa, remove_templates):
    """Convert AF3 dict to SiteAF3 dict."""

    # --- Build receptor entry ---
    receptor_entry = {
        "rec_struct_path": rec_struct_path,
        "fixed_chain_id": sorted(receptor_chains),
    }
    if pocket_path:
        receptor_entry["pocket_path"] = pocket_path
    if hotspot_path:
        receptor_entry["hotspot_path"] = hotspot_path

    # Collect receptor MSA from AF3 sequences whose IDs are all in receptor_chains
    for seq_entry in af3_data.get("sequences", []):
        mol_type = _af3_type(seq_entry)
        if mol_type is None:
            continue
        chain_ids = extract_chain_ids(seq_entry)
        if all(cid in receptor_chains for cid in chain_ids):
            mol_data = seq_entry[mol_type]
            if remove_msa:
                receptor_entry["unpairedMsa"] = ""
                receptor_entry["pairedMsa"] = ""
            else:
                if "unpairedMsa" in mol_data:
                    receptor_entry["unpairedMsa"] = mol_data["unpairedMsa"]
                if "pairedMsa" in mol_data:
                    receptor_entry["pairedMsa"] = mol_data["pairedMsa"]

    # --- Build ligand list ---
    ligands = []
    for seq_entry in af3_data.get("sequences", []):
        mol_type = _af3_type(seq_entry)
        if mol_type is None:
            continue
        mol_data = seq_entry[mol_type]
        chain_ids = extract_chain_ids(seq_entry)

        # Determine which chain IDs from this entry are ligand chains
        lig_ids = [cid for cid in chain_ids if cid in ligand_chains]
        if not lig_ids:
            continue

        for cid in lig_ids:
            ligands.append(
                _build_ligand_entry(mol_type, mol_data, cid, remove_msa, remove_templates)
            )

    # --- Assemble output ---
    result = {
        "name": af3_data.get("name", "converted_case"),
        "modelSeeds": af3_data.get("modelSeeds", [42]),
        "receptor": [receptor_entry],
        "ligand": ligands,
        "dialect": "SiteAF3",
        "version": af3_data.get("version", 3),
    }

    # Pass through optional top-level fields
    if "userCCD" in af3_data:
        result["userCCD"] = af3_data["userCCD"]
    elif "userCCDPath" in af3_data:
        result["userCCDPath"] = af3_data["userCCDPath"]
    else:
        result["userCCD"] = None

    if "bondedAtomPairs" in af3_data:
        result["bondedAtomPairs"] = af3_data["bondedAtomPairs"]

    return result


def main():
    args = parse_args()

    with open(args.input) as f:
        af3_data = json.load(f)

    # Validate input dialect
    dialect = af3_data.get("dialect", "")
    if dialect != "alphafold3":
        print(
            f"Warning: Input dialect is '{dialect}', expected 'alphafold3'. "
            "Proceeding anyway.",
            file=sys.stderr,
        )

    receptor_chains = set(c.strip() for c in args.receptor_chains.split(","))
    all_chain_ids = extract_all_chain_ids(af3_data)

    if args.ligand_chains:
        ligand_chains = set(c.strip() for c in args.ligand_chains.split(","))
    else:
        ligand_chains = all_chain_ids - receptor_chains

    if not ligand_chains:
        print("Error: No ligand chains identified.", file=sys.stderr)
        sys.exit(1)

    # Override name/seeds if specified
    if args.name:
        af3_data["name"] = args.name
    if args.seeds:
        af3_data["modelSeeds"] = [int(s.strip()) for s in args.seeds.split(",")]

    result = convert(
        af3_data,
        receptor_chains=receptor_chains,
        ligand_chains=ligand_chains,
        rec_struct_path=args.rec_struct_path,
        pocket_path=args.pocket_path,
        hotspot_path=args.hotspot_path,
        remove_msa=args.remove_msa,
        remove_templates=args.remove_templates,
    )

    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Converted: {args.input} -> {args.output}")
    print(f"  Receptor chains: {sorted(receptor_chains)}")
    print(f"  Ligand chains: {sorted(ligand_chains)}")
    print(f"  Ligand entries: {len(result['ligand'])}")


if __name__ == "__main__":
    main()
