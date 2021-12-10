# ATLAS Parser

This module is a parser for ATLAS dataset which contains masks for brains with brain stroke. An example of one sample is in *test_nii_to_seq* folder. The folder structure is the following:

```
atlas_parser
│   README.md
│   file001.txt
|	nii_to_sequence.py
│
└───test_nii_to_seq
    │   031769_LesionSmooth_1_stx.nii.gz
    │   031769_LesionSmooth_stx.nii.gz
    |	031769_t1w_deface_stx.nii.gz
|
└───no_stroke
    │   [X].png
    │   ...
|
└───stroke
    │   [X].png
    │   ...
```

1. *nii_to_sequence.py* is a script for parsing brain sample from *test_nii_to_seq* folder. It analyzes the masks and retrieves slices of brain with and without brain stroke. You can either run it on the test data provided in this folder or import it as a module.

2. *test_nii_to_seq* folder contains a 3D brain model (*031769_t1w_deface_stx.nii.gz*) and two 3D masks for areas with brain strokes: (*031769_LesionSmooth_stx.nii.gz* and *031769_LesionSmooth_1_stx.nii.gz*). The path to them is hardcoded in *nii_to_sequence.py*.

3. both *no_stroke* and *stroke* folders contain 15 samples of *.png* images without and with brain stroke respectively. Every time you run the script, new samples will be generated.

If you run this script multiple times, we advise to do this:

```bash
rm -r no_stroke & rm -r yes_stroke & python nii_to_sequence.py
```

Or, if you want to preserve previous runs, change `save_to` dictionary in the script.

**The full version of the dataset is under license and can be obtained by following [this link](http://fcon_1000.projects.nitrc.org/indi/retro/atlas.html) and filling a form to obtain encryption key.**