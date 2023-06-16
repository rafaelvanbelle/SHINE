FROM python:3.7
COPY requirements.txt .

RUN pip install numpy==1.19
RUN pip install torch==1.12.0
RUN pip install torch-scatter -f https://data.pyg.org/whl/torch-1.12.0+cpu.html
RUN pip install torch-sparse -f https://data.pyg.org/whl/torch-1.12.0+cpu.html
RUN pip install torch-geometric==2.1
RUN pip install -r requirements.txt