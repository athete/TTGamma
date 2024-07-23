import hist
import coffea.processor as processor
from coffea.analysis_tools import PackedSelection, Weights
from coffea.nanoevents import NanoAODSchema, NanoEventsFactory
from coffea.nanoevents.methods import nanoaod


NanoAODSchema.warn_missing_crossrefs = False

import awkward as ak
import numpy as np
import pickle
import re

def select_muons(events):
    """
    Select tight and loose muons

    Tight muons have a pT of at least 30 GeV, |eta| < 2.4, pass the tight muon ID cut, and 
    have a relative isolation of less than 0.15
    """

    muonSelectTight = (
        (events.Muon.pt >= 30) & 
        (abs(events.Muon.eta) < 2.4) &
        (events.Muon.tightId) & 
        (events.Muon.pfRelIso04_all < 0.15)
    )

    muonSelectLoose = (
        (events.Muon.pt > 15) & 
        (abs(events.Muon.eta) < 2.4) & 
        ((events.Muon.isPFcand) & (events.Muon.isTracker | events.Muon.isGlobal)) &
        (events.Muon.pfRelIso04_all < 0.25) & 
        np.invert(muonSelectTight)
    )

    return events.Muon[muonSelectTight], events.Muon[muonSelectLoose]

def select_electrons(events):
    """
    Select tight and loose electrons

    Tight electrons should have a pt of at least 35 GeV, |eta| < 2.1, pass the cut based electron
    id, and pass the eta gap, DXY, and DZ cuts defined below
    """

    eleEtaGap = (abs(events.Electron.eta) < 1.4442) | (abs(events.Electron.eta) > 1.566)
    elePassDXY = (abs(events.Electron.eta) < 1.479) & (abs(events.Electron.dxy) < 0.05) |  (abs(events.Electron.eta) > 1.479) & (abs(events.Electron.dxy) < 0.1)
    elePassDZ = (abs(events.Electron.eta) < 1.479) & (abs(events.Electron.dz) < 0.1) | (abs(events.Electron.eta) > 1.479) & (abs(events.Electron.dz) < 0.2)

    eleSelectTight = (
        (events.Electron.pt > 34) &
        (events.Electron.cutBased > 0) & 
        ((eleEtaGap) & (elePassDXY) & (elePassDZ))
    )

    eleSelectLoose = (
        (events.Electron.pt > 15)
        & (abs(events.Electron.eta) < 2.4)
        & eleEtaGap
        & (events.Electron.cutBased >= 1)
        & elePassDXY
        & elePassDZ
        & np.invert(eleSelectTight)
    )

    return events.Electron[eleSelectTight], events.Electron[eleSelectLoose]

def select_photons(photons):
    """Select tight and loose photons
    """

    photonSelect = (
        (photons.pt > 20) & 
        (abs(photons.eta) < 1.4442) &
        (photons.isScEtaEE | photons.isScEtaEB) & 
        (photons.electronVeto) & 
        np.invert(photons.pixelSeed)
    )

    # cut-based ID is precomputed, here we ask for "medium"
    photonID = photons.cutBased >= 2

     # if we want to remove one component of the cut-based ID we can
    # split out the ID requirement using the vid (versioned ID) bitmap
    # this is enabling Iso to be inverted for control regions
    photon_MinPtCut = (photons.vidNestedWPBitmap >> 0 & 3) >= 2
    photon_PhoSCEtaMultiRangeCut = (photons.vidNestedWPBitmap >> 2 & 3) >= 2
    photon_PhoSingleTowerHadOverEmCut = (photons.vidNestedWPBitmap >> 4 & 3) >= 2
    photon_PhoFull5x5SigmaIEtaIEtaCut = (photons.vidNestedWPBitmap >> 6 & 3) >= 2
    photon_ChIsoCut = (photons.vidNestedWPBitmap >> 8 & 3) >= 2
    photon_NeuIsoCut = (photons.vidNestedWPBitmap >> 10 & 3) >= 2
    photon_PhoIsoCut = (photons.vidNestedWPBitmap >> 12 & 3) >= 2

    # photons passing all ID requirements, without the charged hadron isolation cut applied
    photonID_NoChIso = (
        photon_MinPtCut
        & photon_PhoSCEtaMultiRangeCut
        & photon_PhoSingleTowerHadOverEmCut
        & photon_PhoFull5x5SigmaIEtaIEtaCut
        & photon_NeuIsoCut
        & photon_PhoIsoCut
    )

    # select tightPhotons, the subset of photons passing the photonSelect cut and the photonID cut
    tightPhotons = photons[photonSelect & photonID]
    # select loosePhotons, the subset of photons passing the photonSelect cut and all photonID cuts
    # except the charged hadron isolation cut applied (photonID_NoChIso)
    loosePhotons = photons[photonSelect & photonID_NoChIso]

    return tightPhotons, loosePhotons

