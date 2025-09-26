from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski, MolSurf, QED
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
import pandas as pd

# ---- Synthetic Accessibility Score (from Ertl & Schuffenhauer, 2009) ----
from rdkit.Chem import rdMolDescriptors
import math
from EDM.src.delinker_utils.sascorer import calculateScore as calculate_sa_score

# ---- Property Functions ----
def calculate_physicochemical_properties(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        # Calculate properties
        mw = Descriptors.MolWt(mol)
        logp = Crippen.MolLogP(mol)
        hbd = Lipinski.NumHDonors(mol)
        hba = Lipinski.NumHAcceptors(mol)
        psa = MolSurf.TPSA(mol)
        rotatable_bonds = Descriptors.NumRotatableBonds(mol)
        synthetic_accessibility = calculate_sa_score(mol)
        
        return {
            'SMILES': smiles,
            'MW': mw,
            'LogP': logp,
            'HBD': hbd,
            'HBA': hba,
            'PSA': psa,
            'RotatableBonds': rotatable_bonds,
            'SA': synthetic_accessibility
        }
    except:
        return None


def predict_admet_properties(mol):
    mw = Descriptors.MolWt(mol)
    logp = Crippen.MolLogP(mol)
    psa = MolSurf.TPSA(mol)
    
    permeability = 'High' if logp > 2.5 and psa < 250 else 'Low'
    hba = Lipinski.NumHAcceptors(mol)
    hbd = Lipinski.NumHDonors(mol)
    herg_risk = 'High' if logp > 3 and hba > 3 else 'Low'
    cyp_risk = 'Low' if mw < 800 and hba < 10 else 'Moderate'
    num_heteroatoms = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() not in [6, 1])
    stability = 'High' if num_heteroatoms < 10 else 'Moderate'
    
    params = FilterCatalogParams()
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
    catalog = FilterCatalog(params)
    pains_alert = 'Yes' if catalog.HasMatch(mol) else 'No'
    lipinski = sum([mw > 500, logp > 5, hbd > 5, hba > 10])
    
    return {
        'Permeability': permeability,
        'hERG_Risk': herg_risk,
        'CYP_Inhibition': cyp_risk,
        'Metabolic_Stability': stability,
        'PAINS_Alert': pains_alert,
        'Lipinski_Violations': lipinski,
        'QED': QED.qed(mol)
    }


# ---- Run for both SMILES ----
smiles_list = [
    "c1cc2cc(c1O)CCC(=O)/C=C\\COCCC(=O)N2",
    "c1c2c(ccc1O/C=C\\CC(=O)CC/C=C\\CC2)NC(=O)C"
]

results = []
for smi in smiles_list:
    props = calculate_physicochemical_properties(smi)
    mol = Chem.MolFromSmiles(smi)
    admet = predict_admet_properties(mol)
    props.update(admet)
    results.append(props)

# ---- Export to CSV ----
df = pd.DataFrame(results)
df.to_csv("molecule_properties.csv", index=False)

print(df)

