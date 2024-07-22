"""Scale factors for the analysis

This module loads and sets up the scale factor objects
"""
import os.path
from coffea import util
from coffea.btag_tools import BTagScaleFactor
from coffea.jetmet_tools import CorrectedJetsFactory, JECStack
from coffea.lookup_tools import extractor

cwd = os.path.dirname(__file__)

# produced using ttgamma/utils/getBtagEfficiencies.py
taggingEffLookup = util.load(f"{cwd}/taggingEfficienciesDenseLookup.coffea")

bJetScales = BTagScaleFactor(f"{cwd}/Btag/DeepCSV_2016LegacySF_V1.btag.csv", "MEDIUM")

puLookup = util.load(f"{cwd}/puLookup.coffea")
puLookup_Down = util.load(f"{cwd}/puLookup_Down.coffea")
puLookup_Up = util.load(f"{cwd}/puLookup_Up.coffea")

Jetext = extractor()
Jetext.add_weight_sets(
    [
        f"* * {cwd}/JEC/Summer16_07Aug2017_V11_MC_L1FastJet_AK4PFchs.jec.txt",
        f"* * {cwd}/JEC/Summer16_07Aug2017_V11_MC_L2Relative_AK4PFchs.jec.txt",
        f"* * {cwd}/JEC/Summer16_07Aug2017_V11_MC_Uncertainty_AK4PFchs.junc.txt",
        f"* * {cwd}/JEC/Summer16_25nsV1_MC_PtResolution_AK4PFchs.jr.txt",
        f"* * {cwd}/JEC/Summer16_25nsV1_MC_SF_AK4PFchs.jersf.txt",
    ]
)
Jetext.finalize()
Jetevaluator = Jetext.make_evaluator()

jec_names = [
    "Summer16_07Aug2017_V11_MC_L1FastJet_AK4PFchs",
    "Summer16_07Aug2017_V11_MC_L2Relative_AK4PFchs",
    "Summer16_07Aug2017_V11_MC_Uncertainty_AK4PFchs",
    "Summer16_25nsV1_MC_PtResolution_AK4PFchs",
    "Summer16_25nsV1_MC_SF_AK4PFchs",
]

jec_inputs = {name: Jetevaluator[name] for name in jec_names}
jec_stack = JECStack(jec_inputs)

name_map = jec_stack.blank_name_map
name_map["JetPt"] = "pt"
name_map["JetMass"] = "mass"
name_map["JetEta"] = "eta"
name_map["JetA"] = "area"

name_map["ptGenJet"] = "pt_gen"
name_map["ptRaw"] = "pt_raw"
name_map["massRaw"] = "mass_raw"
name_map["Rho"] = "rho"

jet_factory = CorrectedJetsFactory(name_map, jec_stack)


ele_id_sf = util.load(f"{cwd}/MuEGammaScaleFactors/ele_id_sf.coffea")
ele_id_err = util.load(f"{cwd}/MuEGammaScaleFactors/ele_id_err.coffea")

ele_reco_sf = util.load(f"{cwd}/MuEGammaScaleFactors/ele_reco_sf.coffea")
ele_reco_err = util.load(f"{cwd}/MuEGammaScaleFactors/ele_reco_err.coffea")

mu_id_sf = util.load(f"{cwd}/MuEGammaScaleFactors/mu_id_sf.coffea")
mu_id_err = util.load(f"{cwd}/MuEGammaScaleFactors/mu_id_err.coffea")

mu_iso_sf = util.load(f"{cwd}/MuEGammaScaleFactors/mu_iso_sf.coffea")
mu_iso_err = util.load(f"{cwd}/MuEGammaScaleFactors/mu_iso_err.coffea")

mu_trig_sf = util.load(f"{cwd}/MuEGammaScaleFactors/mu_trig_sf.coffea")
mu_trig_err = util.load(f"{cwd}/MuEGammaScaleFactors/mu_trig_err.coffea")
