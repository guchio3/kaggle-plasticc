cd ~/work
jupyter notebook --generate-config

echo "c.NotebookApp.ip = '0.0.0.0'" | sudo tee -a ~/.jupyter/jupyter_notebook_config.py
echo "c.NotebookApp.password = u'sha1:0dd78f8e5ea5:a42fcd92b59a8bc969b0c60d41872fe6de415fb5'" | sudo tee -a ~/.jupyter/jupyter_notebook_config.py
echo "c.NotebookApp.open_browser = False" | sudo tee -a ~/.jupyter/jupyter_notebook_config.py
echo "c.NotebookApp.port = 8000" | sudo tee -a ~/.jupyter/jupyter_notebook_config.py
echo "c.NotebookApp.notebook_dir = '/home/naoya.taguchi/workspace/kaggle/plasticc-2018_after_pack/notebook/'" | sudo tee -a ~/.jupyter/jupyter_notebook_config.py
