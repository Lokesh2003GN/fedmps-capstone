# CHAPTER 6 | Appendix

This appendix gives the repository structure, the commands that reproduce the results and core code listings, taken directly from the source files when this document is built.

## 6.1 Repository Structure

The project is organised into folders for training, the real-CCTV benchmark, the application, the report figures and the shared library, as listed below.

```
fedmps-capstone/
├── training/          01_prepare_data.py, 01b_prepare_stream.py, 02_run_experiments.py,
│                      03_check_local_only.py, 04_pilot.py, aws_logs/
├── cctv_benchmark/    chokepoint_recognition.py, chokepoint_video.py, run_on_aws.sh
├── application/       server_app.py, node_app.py, run_video_scan.py, demo_reset.py
├── report_figures/    scripts that draw the figures and tables
├── fedmps/            shared library: data/, models/ (adapter.py, head.py), fl/, privacy/,
│                      eval/, pipeline/, store.py
├── results/           measured results as JSON files
├── checkpoints/       adapter_vgghq_h256.pt from the VGGFace2-HQ run;
│                      adapter_final.pt from the earlier run, used by the application
└── docs/              review documents and this thesis
```

## 6.2 Reproducing the Results

The commands below download and prepare VGGFace2-HQ and run the pilot and every experiment on the AWS GPU instance, run the real-CCTV benchmark, and start the application with a demonstration scan. The dataset is downloaded with the Kaggle web API onto the instance only, and `HOW_TO_RUN.md` gives the full steps.

```
# VGGFace2-HQ: streaming preparation, on the AWS GPU instance (84 min)
python training/01b_prepare_stream.py --folder ~/data/VGGface2_HQ --out-suffix _vgghq \
    --feat-dir /opt/dlami/nvme/fedmps/vgghq --device cuda --workers 6 --threads 4 \
    --gpu-in-main --save-crops

# Pilot on training identities only (19 min)
python training/04_pilot.py --artifacts-suffix _vgghq --device cuda --rounds 30

# 200 federated rounds and the privacy experiments (64 min)
python training/02_run_experiments.py --artifacts-suffix _vgghq \
    --only three_way dp_sweep masking templates fedprox mia --rounds 200 --aux-rounds 30 \
    --device cuda --fp16-store --adapter-kind mlp --hidden 256 --mu 0.3 --epoch-frac 0.15 \
    --eval-every 1 --no-figures --out experiments_vgghq.json
python training/02_run_experiments.py --artifacts-suffix _vgghq \
    --only degradation operating_points --no-figures --out experiments_vgghq_base.json

# Figures and the comparison table, on the laptop
python report_figures/make_fig12_convergence_200.py
python report_figures/make_comparison_table.py

# Real CCTV benchmark on ChokePoint, on the AWS GPU instance
bash cctv_benchmark/run_on_aws.sh ~/chokepoint

# The application
python -m streamlit run application/server_app.py --server.port 8501
python -m streamlit run application/node_app.py --server.port 8502
python application/run_video_scan.py --video media/demo_node2_person1.mp4 --node 2 \
    --enrol media/person-1.jpg --enrol-name "Priya Sharma" --fps-cap 0 --max-frames 60
```

## 6.3 Contrastive Loss for Local Training

The symmetric InfoNCE loss of {eq:infonce}, taken from `fedmps/fl/node.py`, is computed as follows.

!code[fedmps/fl/node.py](infonce_loss)

## 6.4 Pairwise Masks for Secure Aggregation

The pairwise mask that one node adds to its update in each round, taken from `fedmps/privacy/mask.py`, is computed as follows.

!code[fedmps/privacy/mask.py](SecureAggregator.node_mask)

## 6.5 Track-level Voting

The 3-of-5 vote that turns frame-level hits into one alert per track, taken from `fedmps/pipeline/match.py`, is implemented as follows.

!code[fedmps/pipeline/match.py](TrackVoter.observe)
