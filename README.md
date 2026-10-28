no# AIORI-3-HACKATHON-PROJECTS
This repository will contain the code and resources for the AIORI-3 hackathon that will be developed during the AIORI-3

# YOUTUBE LINK 
https://youtu.be/ed6vltzb-A8
please watch with subtitles 

# HOW TO RUN 
```bash
pip install streamlit pandas numpy torch matplotlib
python -m streamlit run app.py
```

Or, for a quicker approach:

```bash
python main.py --data PEMS08.npz
```

Replace `PEMS08.npz` with your dataset.

# BRIEF DESCRIPTION 
Our problem is PS-003: Federated Traffic-Flow Prediction with Communication-Cost Evaluation.

Cameras and sensors on roads collect traffic data. Normally, all of this data is sent to one central computer, which uses it to predict future traffic. This uses a lot of internet data, and it keeps people's private travel records in one place.

Our project uses federated learning. Federated means the data stays where it is collected. Each location trains its own model and sends only the model's updates, called weights, to the central computer. The raw data is never sent.

Our challenge is to compare two methods. The centralised method sends all data to one place. The federated method keeps data at each location. Each method is called a baseline

We will use one public traffic dataset, which is data that anyone can download. We will split it across 3 to 5 simulated clients. A client is one location, like a road junction.

We will test two kinds of data splits. IID means every client has similar traffic data. Non-IID means every client has different traffic data. Real roads are usually non-IID, so we will study how this affects the results.

We will build a simulator using PyTorch, NumPy and Pandas. Our model will be a compact forecasting model, which means a small model that predicts traffic flow.

To combine what each client learned, we will use FedAvg, short for Federated Averaging. It takes the average of the updates from all clients and makes one shared model.

We will use a chronological split. This means we train on older data and test on newer data. 

We will measure three things. First, accuracy, using MAE and RMSE. MAE is the average size of the model's mistakes. RMSE is similar, but it punishes big mistakes more. For both, a smaller number is better.

Second, communication cost, which is the number of bytes sent over the network in each training round. A round is one cycle of clients training and sending updates.

Third, convergence, which means how quickly the model stops improving and settles on its best result.

We will run every experiment with 3 random seeds. A seed controls the random starting point of training

As the problem statement says, we will not use live cameras, we will not deploy across a whole city, and we will not claim formal privacy guarantees. thank you.
