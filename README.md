# Monolith to Microservices - Representing Application Software through Heterogeneous GNN (IJCAI 2022)

README file for the paper "Monolith to Microservices: Representing Application Software through Heterogeneous GNN"

Step A. : Replicate conda environment
1. conda env create -f environment.yml
2. conda activate CHGNN

Step B. : Run the CHGNN model 
1. python Graph.py --model=AE_EGCN_Separate --data=acme --code=with_edge_loss
