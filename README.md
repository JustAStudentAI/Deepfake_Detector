# Deepfake_Detector
**To run the model**

```
conda create -n deepfake_model_env python=3.9
conda activate deepfake_model_env
```

```
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

```
conda install transformers matplotlib tqdm scikit-learn
```

```
python deepfake_detector.py
```
<br>
<br>

**To run the streamlit app**
```
conda create -n deepfake_app_env python=3.9
conda activate deepfake_app_env
```

```
pip install torch torchvision transformers matplotlib tqdm scikit-learn streamlit sqlalchemy pillow
```

```
streamlit run app_st.py
```

Open your web browser and navigate to the URL provided by Streamlit (e.g., http://localhost:8501)
