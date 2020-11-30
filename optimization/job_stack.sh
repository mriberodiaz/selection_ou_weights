 #!/bin/bash
for clients in 10 20
do
	bazel run main:federated_trainer -- --task=shakespeare --total_rounds=120 --random=True --prob_transmit=0.47 --clients_per_round=$clients --client_optimizer=sgd --server_optimizer=sgd --experiment_name=shakespeare_random3_ou_clients$clients --client_learning_rate=0.1 --server_learning_rate=1.0 --estimation=ou
done


 