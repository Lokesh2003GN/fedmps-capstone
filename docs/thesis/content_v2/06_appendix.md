# CHAPTER 6 | Appendix

This appendix gave the repository structure of the project, the commands that reproduced every result reported in Chapter 4 and three core code listings taken directly from the source files when this document was built, so that the listings always matched the code that produced the results. Section 6.1 gave the repository structure, Section 6.2 gave the commands, and Sections 6.3 to 6.5 gave the contrastive loss used for local training, the pairwise mask used for SA and the track-level vote used by the live pipeline.

## 6.1 Repository Structure

The project was organised into folders for training, the real-CCTV benchmark, the application, the report figures and the shared library, as the following listing showed. The checkpoints folder held the adapter of the final VGGFace2-HQ run, which the application and the benchmark loaded first, and the adapter of the earlier run, which was kept for reference.

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
├── checkpoints/       adapter_vgghq_h256.pt from the VGGFace2-HQ run, used everywhere;
│                      adapter_final.pt from the earlier run
└── docs/              review documents and this thesis
```

## 6.2 Reproducing the Results

The commands below prepared VGGFace2-HQ and ran the pilot and every experiment on the AWS GPU instance, ran the real-CCTV benchmark, and started the application with the node 0 test scan. The dataset was downloaded with the Kaggle web API onto the instance only, and `HOW_TO_RUN.md` gave the full steps, including the installation of the GPU builds of PyTorch and ONNX Runtime.

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

# Real CCTV benchmark on ChokePoint with the latest adapter, on the AWS GPU instance
bash cctv_benchmark/run_on_aws.sh ~/chokepoint

# The application and the node 0 test scan, on the laptop
python application/demo_reset.py
python application/run_video_scan.py --video media/node0.mp4 --node 0 \
    --camera "cam-A (site 0)" --enrol media/person.jpg --enrol-name "Test Subject" --fps-cap 0
python -m streamlit run application/server_app.py --server.port 8501
python -m streamlit run application/node_app.py --server.port 8502
```

## 6.3 Contrastive Loss for Local Training

The symmetric InfoNCE loss of {eq:infonce} was computed by the following function, taken from `fedmps/fl/node.py`. It scored every degraded embedding of a batch against every clean embedding, removed the other images of the same person from the denominator and averaged the loss over both directions.

!code[fedmps/fl/node.py](infonce_loss)

## 6.4 Pairwise Masks for SA

The pairwise mask that one node added to its update in each round was computed by the following method, taken from `fedmps/privacy/mask.py`. It derived one seed per pair of nodes and per round from the shared Diffie–Hellman secret and added or subtracted the resulting mask according to the order of the two nodes.

!code[fedmps/privacy/mask.py](SecureAggregator.node_mask)

## 6.5 Track-level Voting

The three-of-five vote that turned frame-level hits into one alert per track was implemented by the following method, taken from `fedmps/pipeline/match.py`. It kept the recent match decisions of each track and case and fired once when enough of them agreed.

!code[fedmps/pipeline/match.py](TrackVoter.observe)
