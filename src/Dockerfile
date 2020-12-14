FROM python:3.8
RUN pip install sentence-transformers scikit-learn scipy numpy matplotlib ipython pandas nltk
RUN pip install xlrd==1.2.0
COPY CTModel.py /tmp/
COPY Contracts.py /tmp/
COPY 2006.xlsx /tmp/
COPY Foia.csv /tmp/
