# Monolith to Microservices - Representing Application Software through Heterogeneous Graph Neural Network (IJCAI 2022)

This paper explores the idea of using heterogeneous graph neural networks (Het-GNN) to partition old legacy monoliths into candidate microservices. We additionally take membership constraints that come from a subject matter expert who has deep domain knowledge of the application. Using these two ideas, we create **CHGNN** (Community Aware Heterogeneous Graph Neural Network) - a Het-GNN model that performs constrained clustering to generate high quality partitions.

This work was conducted at IBM Research. The authors of this paper are **Alex Mathai**, **Sambaran Bandyopadhyay**, **Utkarsh Desai** and **Srikanth Tamilselvam**. 

Our IJCAI paper can be found [here](https://www.ijcai.org/proceedings/2022/0542.pdf). The extended version of our work that containers many more qualitative and quantitative results can be found on [arxiv](https://arxiv.org/pdf/2112.01317.pdf).

Below are some steps to run our model.

**Step A** : Replicate conda environment
1. conda env create -f environment.yml
2. conda activate CHGNN

**Step B** : Run the CHGNN model 
1. python Graph.py --model=AE_EGCN_Separate --data=acme --code=with_edge_loss
