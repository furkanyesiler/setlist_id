# Investigating the efficacy of music version retrieval systems for setlist identification

This repository contains the instructions for downloading and using the new ASID dataset and the code used for the experiments described in the following paper:

> F. Yesiler, E. Molina, J. Serrà, and E. Gómez. Investigating the efficacy of music version retrieval systems for setlist identification., 2021.

## 1 - ASID: Automatic Setlist Identification Dataset
**ASID** is a new dataset for the setlist identification task that we make available along with the paper above. It contains pre-extracted features, metadata, Youtube or Soundcloud links, and timestamp annotations for 75 concerts and all the relevant reference songs (i.e., the songs that are played in each concert). Concert durations range between 21.7 minutes and 2.5 hours, with a total duration of 99.5 hours. The total number of reference songs is 1,298, with a total duration of 90.1 hours.

**ASID** includes a variety of use cases regarding audio quality and genres. For this, we have selected three categories for audio quality: `AQ-A`, `AQ-B`, and `AQ-C`. `AQ-A` contains high-quality recordings, mainly coming from broadcast recordings or official releases. `AQ-B` contains professionally recorded concerts, mainly from small venues (in general, we observe that the mixing/mastering quality for concerts in `AQ-B` is inferior to the ones in `AQ-A`). Lastly, `AQ-C` contains smartphone or video camera recordings from varying-size venues/events. In terms of genre, we categorize the concerts into 5 main groups: `pop/commercial`, `rock/metal`, `indie/alternative`, `hip-hop/rap`, and `electronic`. The number of concerts for each audio quality and genre can be seen in the following table (the numbers in paranthesis indicate the concerts in the development set, see below):

| **Genre** | **AQ-A** | **AQ-B** | **AQ-C** | **Total** |
|-------|------|-----|---|-----|
|Pop/Commercial| 8 (5)  |3|3|14 (5)|
|Rock/Metal|8 (3)|7|6|21 (3)|
|Indie/Alternative|5|7|3|15|
|Hip-hop/Rap|5 (2)|0|3|8 (2)|
|Electronic|6|1|0|7|
|**Total**|32 (10)|18|15|65 (10)|

We use 10 concerts (14.3 hours) as a separate development set to for hyperparameter tuning, and to train a classifier for the match revision step. The references for the development set include 180 songs. The remaining 65 concerts (85.2 hours) and the related reference set is used for the main results. The total number of annotated segments for the evaluation set is 1,138, with a duration of 80.6 hours.

The pre-extracted features and the metadata are licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).

### 1.1 - Downloading the dataset

Currently, the data is hosted on a Google Drive folder. You can download it using Python as follows: 

```bash
git clone https://github.com/furkanyesiler/setlist_id.git
cd setlist_id
python3 -m venv venv
source venv/bin/activate
pip install -r requirements_asid.txt
```

After installing the requirements, you can run the `download_asid.py` script. The usage is shown below:

```bash
python download_asid.py --help
```
```
usage: download_asid.py [-h] [--source {gdrive,zenodo}]                                                                                                                    [--outputdir OUTPUTDIR] [--unpack] [--remove]                                                                                                                                       

Download script for ASID

optional arguments:
-h, --help            show this help message and exit
--source {gdrive,zenodo}
                      from which source to download the files. you can
                      either download from Google Drive (gdrive) or from
                      Zenodo (zenodo) (default: gdrive)
--outputdir OUTPUTDIR
                      directory to store the dataset (default: ./data/)
--unpack              unpack the zip files (default: False)
--remove              remove zip files after unpacking (default: False)
```

Example usage:

```bash
python download_asid.py --source gdrive --outputdir ./data/ --unpack --remove
```

### 1.2 - Contents
When you use the `download_asid.py` script, following files/folders will be downloaded:

* **ASID_metadata/ASID-Annotations.xlsx:** This file contains the annotations in a "human-readable" way. You can open the file with applications that read spreadsheets, or with Python (using the `pandas` library). An example of how to read the file using Python can be found in `ASID-annotations_notebook.ipynb`.

* **ASID_metadata/ASID_eval_groundtruth.csv:** The ground truth annotations for the concerts in the evaluation subset. This file is a parsed version of the `ASID-Annotations.xlsx` file for our evaluation script.

* **ASID_metadata/ASID_development_groundtruth.csv:** The ground truth annotations for the concerts in the development subset. This file is a parsed version of the `ASID-Annotations.xlsx` file for our evaluation script.

* **ASID_eval_concerts:** The pre-extracted cremaPCP features for the concerts in the evaluation subset.

* **ASID_development_concerts:** The pre-extracted cremaPCP features for the concerts in the development subset.

* **ASID_eval_references:** The pre-extracted cremaPCP features for the reference songs in the evaluation subset.

* **ASID_development_references:** The pre-extracted cremaPCP features for the reference songs in the development subset.

* **ASID_onsets:** The pre-extracted onset strength envelops for both the concerts and the reference songs in both the evaluation and the development subsets.

## 2 - Replicating the experiments
The procedure for replicating the experiments consists of 2 stages:

* Extracting the raw results (matches) using a sliding window
* Postprocessing the raw results to clear potential false positives and analyze the results

### 2.1 - Installing the requirements
The code requires Python3.6+. The following commands can be used for installing the requirements:

```bash
git clone https://github.com/furkanyesiler/setlist_id.git
cd setlist_id
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2.2 - Creating `.lst` files for query and reference sets
The recommended inputs for the `setlist_benchmark_systems.py` script are `.lst` files that contain queries and references in each line. After downloading the dataset, you can run the following commands from the main directory:

```bash
ls -d ./data/ASID_development_concerts/*.npy > ./data/development_queries.lst
ls -d ./data/ASID_development_references/*.npy > ./data/development_references.lst
ls -d ./data/ASID_eval_concerts/*.npy > ./data/eval_queries.lst
ls -d ./data/ASID_eval_references/*.npy > ./data/eval_references.lst
```

### 2.3 - Extracting the raw results
The code for the first step can be found in `setlist_benchmark_systems.py`. The arguments for that script are as the following:

```bash
python setlist_benchmark_systems.py --help
```
```
usage: setlist_benchmark_systems.py [-h] [-q QUERY] [-r REFERENCE]
                                    [-out OUTPUT] [-s {qmax,2dftm,re-move}]
                                    [-sr SAMPLE_RATE]
                                    [--segment_size SEGMENT_SIZE]
                                    [--hop_size HOP_SIZE]
                                    [--num_workers NUM_WORKERS]

Comparing similarities among a query and references.

optional arguments:
  -h, --help            show this help message and exit
  -q QUERY, --query QUERY
                        path to query.
  -r REFERENCE, --reference REFERENCE
                        path to reference list.
  -out OUTPUT, --output OUTPUT
                        path to store the results.
  -s {qmax,2dftm,re-move}, --system {qmax,2dftm,re-move}
                        system to use
  -sr SAMPLE_RATE, --sample_rate SAMPLE_RATE
                        directory to store the embeddings.
  --segment_size SEGMENT_SIZE
                        segment size (in seconds) for the identification
                        window.
  --hop_size HOP_SIZE   hop size (in seconds) for the identification window.
  --num_workers NUM_WORKERS
                        num_of_workers for parallel processing.
```

Example usage:
```bash
python setlist_benchmark_systems.py -q ./data/development_queries.lst \
                                    -r ./data/development_references.lst \
                                    -out ./development_results_re-move_120_30.csv \
                                    -s re-move \
                                    -sr 44100 \
                                    --segment_size 120 \
                                    --hop_size 30 \
                                    --num_workers 1

python setlist_benchmark_systems.py -q ./data/eval_queries.lst \
                                    -r ./data/eval_references.lst \
                                    -out ./eval_results_re-move_120_30.csv \
                                    -s re-move \
                                    -sr 44100 \
                                    --segment_size 120 \
                                    --hop_size 30 \
                                    --num_workers 1
```

### 2.4 - Postprocessing and analyzing the results
The code for the final step can be found in `create_result_reports.py`. The arguments for that script are as the following:

```bash
python create_result_reports.py --help
```
```
usage: create_result_reports.py [-h] [-rv RES_VAL_FILENAME]
                                [-re RES_EVAL_FILENAME] [-gv GT_VAL_FILENAME]
                                [-ge GT_EVAL_FILENAME] [--skip {0,1}]
                                [--concerts {aq_a,aq_b,aq_c,pop,rock,indie,rap,electronic}]

Training and evaluation code for Re-MOVE experiments.

optional arguments:
  -h, --help            show this help message and exit
  -rv RES_VAL_FILENAME, --res_val_filename RES_VAL_FILENAME
                        Path to the results file for the development set.
  -re RES_EVAL_FILENAME, --res_eval_filename RES_EVAL_FILENAME
                        Path to the results file for the evaluation set.
  -gv GT_VAL_FILENAME, --gt_val_filename GT_VAL_FILENAME
                        Path to the ground truth annotations for the
                        development set.
  -ge GT_EVAL_FILENAME, --gt_eval_filename GT_EVAL_FILENAME
                        Path to the ground truth annotations for the
                        evaluation set.
  --skip {0,1}          Whether to skip every second result on the res_eval
                        file.
  --concerts {aq_a,aq_b,aq_c,pop,rock,indie,rap,electronic}
                        Which subset of concerts to create the resultsfor.
                        Default is all concerts.
```


Example usage:
```bash
python create_result_reports.py -rv ./development_results_re-move_120_30.csv \
                                -re ./eval_results_re-move_120_30.csv \
                                -gv ./data/ASID_metadata/ASID_development_groundtruth.csv \
                                -ge ./data/ASID_metadata/ASID_eval_groundtruth.csv
```

## Questions
For any questions you may have, feel free to create an issue or contact [me](mailto:furkan.yesiler@upf.edu).

## License
The code in this repository is licensed under [Affero GPL v3](https://www.gnu.org/licenses/agpl-3.0.en.html).

The pre-extracted features and the metadata for ASID are licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).

## References
Please cite our reference if you plan to use the code in this repository:
```
@inproceedings{yesiler2021,
    author = "Furkan Yesiler and Emilio Molina and Joan Serrà and Emilia Gómez",
    title = "Investigating the efficacy of music version retrieval systems for setlist identification",
    year = "2021"
}
```

## Acknowledgments

This work has received funding from the European Union's Horizon 2020 research and innovation programme under the Marie Skłodowska-Curie grant agreement No. 765068 (MIP-Frontiers).

<img src="https://upload.wikimedia.org/wikipedia/commons/b/b7/Flag_of_Europe.svg" height="64" hspace="20">
