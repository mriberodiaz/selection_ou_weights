 #!/bin/bash
 for estimation in ou zero ignore
 do
	for clients in 10 20 30
	do
		bazel run main:federated_trainer -- --task=synthetic --total_rounds=500  --clients_per_round=$clients --client_optimizer=sgd --server_optimizer=sgd --experiment_name=synthetic_${estimation}_clients${clients} --client_learning_rate=0.1 --server_learning_rate=1.0 --estimation=ou --rounds_per_checkpoint=50
	done
done


 