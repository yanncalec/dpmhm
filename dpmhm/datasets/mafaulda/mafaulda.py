"""mafaulda dataset."""

import os
import pathlib
import itertools
import json
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import pandas as pd

# TODO(mafaulda): Markdown description  that will appear on the catalog page.
_DESCRIPTION = """
Machinery Fault Database.

This database is composed of 1951 multivariate time-series acquired by sensors on a SpectraQuest's Machinery Fault Simulator (MFS) Alignment-Balance-Vibration (ABVT). The 1951 comprises six different simulated states: normal function, imbalance fault, horizontal and vertical misalignment faults and, inner and outer bearing faults. This section describes the database.

For more information please contact Felipe M. L. Ribeiro (felipe.ribeiro@smt.ufrj.br).

Description
===========

Fault Types
-----------
- Normal Sequences: There are 49 sequences without any fault, each with a fixed rotation speed within the range from 737 rpm to 3686 rpm with steps of approximately 60 rpm.
- Imbalance Faults: Simulated with load values within the range from 6 g to 35 g. For each load value below 30 g, the rotation frequency assumed in the same 49 values employed in the normal operation case. For loads equal to or above 30 g, however, the resulting vibration makes impracticable for the system to achieve rotation frequencies above 3300 rpm, limiting the number of distinct rotation frequencies. The table below presents the number of sequences per weight.
- Horizontal Parallel Misalignment: This type of fault was induced into the MFS by shifting the motor shaft horizontally of 0.5 mm, 1.0 mm, 1.5 mm, and 2.0 mm. Using the same range for the rotation frequency as in the normal operation for each horizontal shift, the table below presents the number of sequences per degree of misalignment.
- Vertical Parallel Misalignment: This type of fault was induced into the MFS by shifting the motor shaft horizontally of 0.51 mm, 0.63 mm, 1.27 mm, 1.40 mm, 17.8 mm and 1.90 mm. Using the same range for the rotation frequency as in the normal operation for each vertical shift, the table below presents the number of sequences per degree of misalignment.
- Bearing Faults: This type of fault was induced into one of the three components (outer track, rolling elements, and inner track) and at either the underhang position (between the rotor and the motor) or the overhang position (between the bearing and the motor). Bearing faults are practically imperceptible when there is no imbalance. So, three masses of 6 g, 20 g, and 35 g were added to induce a detectable effect, with different rotation frequencies as before.

Data Acquisition System
-----------------------
- Three Industrial IMI Sensors, Model 601A01 accelerometers on the radial, axial and tangencial directions:
Sensibility: (±20%) 100 mV per g (10.2 mV per m/s2);
Frequency range: (±3 dB) 16-600000 CPM (0.27-10.000 Hz);
Measurement range: ±50 g (±490 m/s2).
- One IMI Sensors triaxial accelerometer, Model 604B31, returning data over the radial, axial and tangencial directions:
Sensibility: (±20%) 100 mV per g (10.2 mV per m/s2);
Frequency range: (±3 dB) 30-300000 CPM (0.5-5.000 Hz);
Measurement range: ±50 g (±490 m/s2)
- Monarch Instrument MT-190 analog tachometer
- Shure SM81 microphone with frequency range of 20-20.000 Hz
- Two National Instruments NI 9234 4 channel analog acquisition modules, with sample rate of 51.2 kHz

Homepage
--------
http://www02.smt.ufrj.br/~offshore/mfs/page_01.html

Original data
=============
Sampling rate: 50 kHz
Recording duration: 5 seconds, 25000 samples
Size: 30 Gb

The database is composed by several CSV (Comma-Separated Values) files, each one with 8 columns, one column for each sensor, according to:
- column 1: tachometer signal that allows to estimate rotation frequency;
- columns 2 to 4: underhang bearing accelerometer (axial, radiale tangential direction);
- columns 5 to 7: overhang bearing accelerometer (axial, radiale tangential direction);
- column 8: microphone.

Modifications
=============
For bearing faults the original data used the lables [`ball_fault`, `cage_fault`, `outer_race`] respectively for the components [rolling elements, inner track, outer track]. These labels also seem to be misspecified when comparing to the list showing the number of files of each class in the original description.
"""

# TODO(mafaulda): BibTeX citation
_CITATION = """
@misc{ribeiro_mafaulda_2014,
	title = {{MAFAULDA}: {Machinery} {Fault} {Database}},
	url = {http://www02.smt.ufrj.br/~offshore/mfs/},
	author = {Ribeiro, Felipe M.L.},
	year = {2014},
}"""

_FAULT_TYPE = ['normal', 'horizontal-misalignment', 'vertical-misalignment', 'imbalance', 'underhang', 'overhang']

_DATA_URLS = ['http://www02.smt.ufrj.br/~offshore/mfs/database/mafaulda/'+f+'.zip' for f in _FAULT_TYPE]  # The original '.tgz' files provided cannot be read by tfds


class MAFAULDA(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for mafaulda dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    # TODO(mafaulda): Specifies the tfds.core.DatasetInfo object
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
          'signal': tfds.features.Tensor(shape=(None, 8), dtype=tf.float64),

          'label': tfds.features.ClassLabel(names=['Normal', 'Faulty']),

          'metadata': {
            'DataLabel': tf.string,  # [normal, imbalance:6g, horizontal-misalignment:0.5mm, ...]
            'FileName': tf.string,
          },
        }),
        # If there's a common (input, target) tuple from the
        # features, specify them here. They'll be used if
        # `as_supervised=True` in `builder.as_dataset`.
        supervised_keys = None,  # Set to `None` to disable
        homepage='http://www02.smt.ufrj.br/~offshore/mfs/page_01.html',
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Return SplitGenerators.
    """
    if dl_manager._manual_dir.exists():  # prefer to use manually downloaded data
      datadir = dl_manager._manual_dir
      # print(datadir)
      return {
          ft: self._generate_examples(datadir/(ft+'zip')) for ft in _FAULT_TYPE
      }
    else:  # automatically download data
      # datadir = dl_manager.download(_DATA_URLS)
      datadir = [dl_manager.download(url) for url in _DATA_URLS]
      fdict = self._fname_parser(datadir)

      return {
          ft: self._generate_examples(fdict[ft]) for ft in _FAULT_TYPE
          # 'train': self._generate_examples(datadir),
      }

  def _fname_parser(self, path):
    fd = {}
    for zp in path:
      # print('parser', zp)
      with open(str(zp)+'.INFO') as fp:
        dd=json.load(fp)
        fd[dd['original_fname'].split('.zip')[0]] = zp

    return fd

  def _generate_examples(self, path):
    for fname, fobj in tfds.download.iter_archive(path, tfds.download.ExtractMethod.ZIP):
        x = pd.read_csv(fobj).values

        _dscrp = pathlib.Path(fname).parent.parts
        metadata = {
          'DataLabel': ':'.join(_dscrp),
          'FileName': fname,
        }

        yield hash(frozenset(metadata.items())), {
          'signal': x,
          'label': 'Normal' if _dscrp[0]=='normal' else 'Faulty',
          'metadata': metadata,
      }
