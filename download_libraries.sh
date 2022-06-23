# **************** UNCOMMENT AND RUN THIS CELL IF YOU NEED TO INSTALL/ UPGRADE TSAI & SKTIME ****************
stable = True # Set to True for latest pip version or False for main branch in GitHub
pip install {"tsai -U" if stable else "git+https://github.com/timeseriesAI/tsai.git"} >> /dev/null
pip install sktime -U  >> /dev/null
pip install catch22
git clone https://github.com/CenkEralp/MiniRocket_Experiments.git
cd MiniRocket_Experiments