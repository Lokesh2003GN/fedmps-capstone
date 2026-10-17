# CHAPTER 6 | Appendix

This appendix gives the repository structure, the commands that reproduce the results, and core code listings, taken directly from the source files when this document is built.

## 6.1 Repository Structure

```
fedmps-capstone/
├── training/          01_prepare_data.py, 02_run_experiments.py, 03_check_local_only.py, aws_logs/
├── cctv_benchmark/    chokepoint_recognition.py, chokepoint_video.py, run_on_aws.sh
├── application/       server_app.py, node_app.py, run_video_scan.py, demo_reset.py
├── report_figures/    scripts that draw the figures and tables
├── fedmps/            shared library: data/, models/, fl/, privacy/, eval/, pipeline/, store.py
├── results/           measured results (JSON)
├── checkpoints/       adapter_final.pt (the trained federated adapter)
└── docs/              review documents and this thesis
```

## 6.2 Reproducing the Results

```
# VGGFace2: data preparation and all experiments (AWS GPU instance)
python training/01_prepare_data.py --source folder --folder ~/vggface2 --backbone insightface \
    --device cuda --max-identities 8631 --max-images-per-identity 20 --out-suffix _vgg_big
python training/02_run_experiments.py --all --artifacts-suffix _vgg_big --out experiments_vgg_big.json
python training/03_check_local_only.py --artifacts-suffix _vgg_big

# Real CCTV benchmark on ChokePoint (AWS GPU instance)
bash cctv_benchmark/run_on_aws.sh ~/chokepoint

# The application
python -m streamlit run application/server_app.py --server.port 8501
python -m streamlit run application/node_app.py --server.port 8502
python application/run_video_scan.py --video media/demo_node2_person1.mp4 --node 2 \
    --enrol media/person-1.jpg --enrol-name "Priya Sharma" --fps-cap 0 --max-frames 60
```

## 6.3 Contrastive Loss for Local Training (`fedmps/fl/node.py`)

!code[fedmps/fl/node.py](infonce_loss)

## 6.4 Pairwise Masks for Secure Aggregation (`fedmps/privacy/mask.py`)

!code[fedmps/privacy/mask.py](SecureAggregator.node_mask)

## 6.5 Track-level Voting (`fedmps/pipeline/match.py`)

!code[fedmps/pipeline/match.py](TrackVoter.observe)
