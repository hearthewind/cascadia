"""Mass spectrometry data parsers."""
from __future__ import annotations

import gzip
import logging
import os
import pickle
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from os import PathLike
from pathlib import Path

import numpy as np
from pyteomics.mgf import MGF
from pyteomics.mzml import MzML
from pyteomics.mzxml import MzXML
from tqdm.auto import tqdm
from pyteomics import mass
import pandas as pd


from .. import utils
from ..primitives import MassSpectrum

LOGGER = logging.getLogger(__name__)


class BaseParser(ABC):
    """A base parser class to inherit from.

    Parameters
    ----------
    ms_data_file : PathLike
        The peak file to parse.
    ms_level : int
        The MS level of the spectra to parse.
    preprocessing_fn : Callable or Iterable[Callable], optional
        The function(s) used to preprocess the mass spectra.
    valid_charge : Iterable[int], optional
        Only consider spectra with the specified precursor charges. If `None`,
        any precursor charge is accepted.
    id_type : str, optional
        The Hupo-PSI prefix for the spectrum identifier.
    """

    def __init__(
        self,
        ms_data_file: PathLike,
        ms_level: int,
        preprocessing_fn: Callable | Iterable[Callable] | None = None,
        valid_charge: Iterable[int] | None = None,
        id_type: str = "scan",
    ) -> None:
        """Initialize the BaseParser."""
        self.path = Path(ms_data_file)
        self.ms_level = ms_level
        if preprocessing_fn is None:
            self.preprocessing_fn = []
        else:
            self.preprocessing_fn = utils.listify(preprocessing_fn)

        self.valid_charge = None if valid_charge is None else set(valid_charge)
        self.id_type = id_type
        self.offset = None
        self.precursor_mz = []
        self.precursor_charge = []
        self.scan_id = []
        self.mz_arrays = []
        self.intensity_arrays = []
        self.annotations = None

    @abstractmethod
    def open(self) -> Iterable:
        """Open the file as an iterable."""

    @abstractmethod
    def parse_spectrum(self, spectrum: dict) -> MassSpectrum | None:
        """Parse a single spectrum.

        Parameters
        ----------
        spectrum : dict
            The dictionary defining the spectrum in a given format.

        Returns
        -------
        MassSpectrum or None
            The parsed mass spectrum or None if it is skipped.
        """

    def read(self) -> BaseParser:
        """Read the ms data file.

        Returns
        -------
        Self
        """
        n_skipped = 0
        with self.open() as spectra:
            for spectrum in tqdm(spectra, desc=str(self.path), unit="spectra"):
                try:
                    spectrum = self.parse_spectrum(spectrum)
                    if spectrum is None:
                        continue

                    if self.preprocessing_fn is not None:
                        for processor in self.preprocessing_fn:
                            spectrum = processor(spectrum)

                    self.mz_arrays.append(spectrum.mz)
                    self.intensity_arrays.append(spectrum.intensity)
                    self.precursor_mz.append(spectrum.precursor_mz)
                    self.precursor_charge.append(spectrum.precursor_charge)
                    self.scan_id.append(_parse_scan_id(spectrum.scan_id))
                    if self.annotations is not None:
                        self.annotations.append(spectrum.label)
                except (IndexError, KeyError, ValueError):
                    n_skipped += 1

        if n_skipped:
            LOGGER.warning(
                "Skipped %d spectra with invalid precursor info", n_skipped
            )

        self.precursor_mz = np.array(self.precursor_mz, dtype=np.float64)
        self.precursor_charge = np.array(
            self.precursor_charge,
            dtype=np.uint8,
        )

        self.scan_id = np.array(self.scan_id)

        # Build the index
        sizes = np.array([0] + [s.shape[0] for s in self.mz_arrays])
        self.offset = sizes[:-1].cumsum()
        self.mz_arrays = np.concatenate(self.mz_arrays).astype(np.float64)
        self.intensity_arrays = np.concatenate(self.intensity_arrays).astype(
            np.float32
        )
        return self

    @property
    def n_spectra(self) -> int:
        """The number of spectra."""
        return self.offset.shape[0]

    @property
    def n_peaks(self) -> int:
        """The number of peaks in the file."""
        return self.mz_arrays.shape[0]
    
class MsgpParser(BaseParser):
    """Parse augmented mass spectra from an asf file.

    Parameters
    ----------
    ms_data_file : PathLike
        The asf file to parse.
    ms_level : int
        The MS level of the spectra to parse.
    preprocessing_fn : Callable or Iterable[Callable], optional
        The function(s) used to preprocess the mass spectra.
    valid_charge : Iterable[int], optional
        Only consider spectra with the specified precursor charges. If `None`,
        any precursor charge is accepted.
    annotations : bool
        Include peptide annotations.
    """

    def __init__(
        self,
        ms_data_file: PathLike,
        ms_level: int = 2,
        preprocessing_fn: Callable | Iterable[Callable] | None = None,
        valid_charge: Iterable[int] | None = None,
        annotations: bool = False,
    ) -> None:
        """Initialize the MgfParser."""
        super().__init__(
            ms_data_file,
            ms_level=ms_level,
            preprocessing_fn=preprocessing_fn,
            valid_charge=valid_charge,
            id_type="index",
        )
        if annotations:
            self.annotations = []

        self.rt_arrays = []
        self.level_arrays = []
        self.fragment_arrays = []
        self.global_rts = []
        self._counter = -1

        self.top_n = 150
    
    def parse_spectrum(self, spectrum: dict) -> MassSpectrum:
        pass

    def open(self) -> Iterable[dict]:
        pass

    def read(self) -> BaseParser:
        spec_header = pd.read_csv(self.path, index_col='feature_id')
        directory_path = str(self.path.parent) + '/'

        n_skipped = 0
        for index, row in tqdm(spec_header.iterrows(), total=len(spec_header), desc=f'Reading file {self.path.name}'):
            with open(os.path.join(directory_path, row['MSGP File Name']), 'rb') as f:
                f.seek(row['MSGP Datablock Pointer'])
                spec = pickle.loads(gzip.decompress(f.read(int(row['MSGP Datablock Length']))))

            mzs = []
            intensities = []
            rts = []
            levels = []
            precursor_mz = None
            precursor_charge = 0
            scan_id = 0
            label = None
            rt = None

            precursor_mz = float(row['precursor_mz'])
            precursor_charge = int(row['charge'])
            scan_id = row.name
            label = row['mod_sequence'].replace('m', 'M[Oxidation]').replace('c', 'C[Carbamidomethyl]')
            rt = row['rt']

            related_ms1 = spec['related_ms1']
            related_ms2 = spec['related_ms2']

            fragment_labels = []

            for _, spectrum in related_ms1.iterrows():
                sorted_intensity_idxs = np.argsort(spectrum.intensity)[-self.top_n:]

                mzs.extend(list(spectrum.mz[sorted_intensity_idxs]))
                intensities.extend(list(spectrum.intensity[sorted_intensity_idxs]))
                rts.extend([spectrum.rt] * len(spectrum.mz[sorted_intensity_idxs]))
                levels.extend([1] * len(spectrum.mz[sorted_intensity_idxs]))
                fragment_labels.extend([0] * len(spectrum.mz[sorted_intensity_idxs]))

            for _, spectrum in related_ms2.iterrows():
                sorted_intensity_idxs = np.argsort(spectrum.intensity)[-self.top_n:]

                mzs.extend(list(spectrum.mz[sorted_intensity_idxs]))
                intensities.extend(list(spectrum.intensity[sorted_intensity_idxs]))
                rts.extend([spectrum.rt] * len(spectrum.mz[sorted_intensity_idxs]))
                levels.extend([2] * len(spectrum.mz[sorted_intensity_idxs]))
                fragment_labels.extend([0] * len(spectrum.mz[sorted_intensity_idxs]))

            spectrum = MassSpectrum(
                filename=str(self.path),
                scan_id=scan_id,
                mz=mzs,
                intensity=intensities,
                rt=np.array(rts),
                level=np.array(levels),
                precursor_mz=precursor_mz,
                precursor_charge=precursor_charge,
                label=label,
                fragment_label=np.array(fragment_labels)
            )

            if self.preprocessing_fn is not None:
                for processor in self.preprocessing_fn:
                    spectrum = processor(spectrum)

            self.global_rts.append(rt)
            self.mz_arrays.append(spectrum.mz)
            self.intensity_arrays.append(spectrum.intensity)
            self.rt_arrays.append(spectrum.rt)
            self.level_arrays.append(spectrum.level)
            self.fragment_arrays.append(spectrum.fragment_label)
            self.precursor_mz.append(spectrum.precursor_mz)
            self.precursor_charge.append(spectrum.precursor_charge)
            self.scan_id.append(_parse_scan_id(spectrum.scan_id))
            if self.annotations is not None:
                self.annotations.append(spectrum.label)

        if n_skipped:
            LOGGER.warning(
                "Skipped %d spectra with invalid precursor info", n_skipped
            )

        self.precursor_mz = np.array(self.precursor_mz, dtype=np.float64)
        self.precursor_charge = np.array(
            self.precursor_charge,
            dtype=np.uint8,
        )

        self.scan_id = np.array(self.scan_id)
        self.global_rts = np.array(self.global_rts)

        # Build the index
        sizes = np.array([0] + [s.shape[0] for s in self.mz_arrays])
        self.offset = sizes[:-1].cumsum()
        self.mz_arrays = np.concatenate(self.mz_arrays).astype(np.float64)
        self.intensity_arrays = np.concatenate(self.intensity_arrays).astype(
            np.float32
        )
        self.rt_arrays = np.concatenate(self.rt_arrays).astype(np.float32)
        self.level_arrays = np.concatenate(self.level_arrays).astype(np.uint8)
        self.fragment_arrays = np.concatenate(self.fragment_arrays).astype(np.uint8)

        return self

def _parse_scan_id(scan_str: str | int) -> int:
    return hash(scan_str)
