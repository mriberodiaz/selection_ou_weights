# Copyright 2019, Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""An implementation of the FedAvg algorithm with learning rate schedules.

This is intended to be a somewhat minimal implementation of Federated
Averaging that allows for client and server learning rate scheduling.

The original FedAvg is based on the paper:

Communication-Efficient Learning of Deep Networks from Decentralized Data
    H. Brendan McMahan, Eider Moore, Daniel Ramage,
    Seth Hampson, Blaise Aguera y Arcas. AISTATS 2017.
    https://arxiv.org/abs/1602.05629
"""

import collections
from typing import Callable, Optional, Union

import attr
from absl import logging
import tensorflow as tf
import tensorflow_federated as tff
from tensorflow_federated.python.core.api import intrinsics
from tensorflow_federated.python.tensorflow_libs import tensor_utils
from optimization.shared import estimation_utils
from tensorflow.python.ops import clip_ops
from tensorflow_model_optimization.python.core.internal import tensor_encoding as te
from tensorflow_federated.python.core.templates import measured_process
from tensorflow_federated.python.core.impl.types import type_conversions
from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.core.api import computation_base
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.api import placements


# Convenience type aliases.
ModelBuilder = Callable[[], tff.learning.Model]
OptimizerBuilder = Callable[[float], tf.keras.optimizers.Optimizer]
ClientWeightFn = Callable[..., float]
LRScheduleFn = Callable[[Union[int, tf.Tensor]], Union[tf.Tensor, float]]


def _initialize_optimizer_vars(model: tff.learning.Model,
                               optimizer: tf.keras.optimizers.Optimizer):
  """Ensures variables holding the state of `optimizer` are created."""
  delta = tf.nest.map_structure(tf.zeros_like, _get_weights(model).trainable)
  model_weights = _get_weights(model)
  grads_and_vars = tf.nest.map_structure(lambda x, v: (x, v), delta,
                                         model_weights.trainable)
  optimizer.apply_gradients(grads_and_vars, name='server_update')
  assert optimizer.variables()


def _get_weights(model: tff.learning.Model) -> tff.learning.ModelWeights:
  return tff.learning.ModelWeights.from_model(model)


@attr.s(eq=False, frozen=True)
class ServerState(object):
  """Structure for state on the server.

  Fields:
  -   `model`: A dictionary of the model's trainable and non-trainable
        weights.
  -   `optimizer_state`: The server optimizer variables.
  -   `round_num`: The current training round, as a float.
  """
  model = attr.ib()
  optimizer_state = attr.ib()
  round_num = attr.ib()

  predicted_delta = attr.ib()
  Sx = attr.ib()
  Sxx = attr.ib()
  Sy = attr.ib()
  Syy = attr.ib()
  Sxy = attr.ib()

  num_participants = attr.ib()
  global_norm_mean = attr.ib()
  global_norm_std = attr.ib()
  threshold = attr.ib()
  delta_aggregate_state = attr.ib()
  # This is a float to avoid type incompatibility when calculating learning rate
  # schedules.


@tf.function
def server_update(model, server_optimizer, server_state, weights_delta,num_participants, global_norm_mean, global_norm_var):
  """Updates `server_state` based on `weights_delta`.

  Args:
    model: A `KerasModelWrapper` or `tff.learning.Model`.
    server_optimizer: A `tf.keras.optimizers.Optimizer`. If the optimizer
      creates variables, they must have already been created.
    server_state: A `ServerState`, the state to be updated.
    weights_delta: A nested structure of tensors holding the updates to the
      trainable variables of the model.

  Returns:
    An updated `ServerState`.
  """
  # Initialize the model with the current state.
  model_weights = model.weights
  tff.utils.assign(model_weights, server_state.model)

  initial_weights = tf.nest.map_structure(tf.identity, model_weights.trainable)
  initial_weights_squared = tf.nest.map_structure(tf.math.square, model_weights.trainable)
  Sx = tf.nest.map_structure(lambda a, b : a+ b, server_state.Sx, initial_weights)
  Sxx = tf.nest.map_structure(lambda a, b : a+ b, server_state.Sxx, initial_weights_squared)

  tff.utils.assign(server_optimizer.variables(), server_state.optimizer_state)

  # Apply the update to the model.
  grads_and_vars = tf.nest.map_structure(
      lambda x, v: (-1.0 * x, v), tf.nest.flatten(weights_delta),
      tf.nest.flatten(model_weights.trainable))
  server_optimizer.apply_gradients(grads_and_vars, name='server_update')

  new_weights_squared = tf.nest.map_structure(tf.math.square, model_weights.trainable)
  Sy = tf.nest.map_structure(lambda a, b : a+ b, server_state.Sy, model_weights.trainable)
  Syy = tf.nest.map_structure(lambda a, b : a+ b, server_state.Syy, new_weights_squared)

  Sxy = tf.nest.map_structure(lambda a, b : a + b, server_state.Sxy, tf.nest.map_structure(tf.math.multiply, initial_weights, model_weights.trainable))

  n = tf.cast(server_state.round_num, tf.float32)
  if server_state.round_num<1:
    predicted_delta = server_state.predicted_delta
    A_has_non_finite = tf.constant(0,tf.int32)
    B_has_non_finite = tf.constant(0,tf.int32)
  else:
    n=n+1.0
    # def calc_A(sx,sy,sxy, sxx, n):
    
    def calc_A(sx,sy,sxy, sxx):
      num = n*sxy - tf.math.multiply(sx,sy)
      den = n*sxx - tf.math.square(sx)
      res = num/den 
      res = tf.where(tf.math.is_nan(res), tf.zeros_like(res), res)
      res = tf.where(tf.math.is_inf(res), tf.zeros_like(res), res)
      return res
    # def calc_B(sx,sy,a,n):
    
    def calc_B(sx,sy,a):
      return (sy - tf.math.multiply(a,sx))/n

    def predict_next(current, A,B):
      return tf.math.multiply(A,current)+B

    A = tf.nest.map_structure(calc_A, Sx, Sy, Sxy, Sxx)
    A, A_has_non_finite = (tensor_utils.zero_all_if_any_non_finite(A))
    # if A_has_non_finite>0:
    logging.info(f'A has non finite at round {n}')
    B = tf.nest.map_structure(calc_B, Sx,Sy,A)
    B, B_has_non_finite = (tensor_utils.zero_all_if_any_non_finite(B))
    # if B_has_non_finite:
    logging.info(f'B has non finite at round {n}')

    pred = tf.nest.map_structure(predict_next, model_weights.trainable, A,B)
    predicted_delta = tf.nest.map_structure(lambda x,y: x-y, pred,model_weights.trainable )

  #global_norm_stddev = tf.sqrt(global_norm_var)
  # Create a new state based on the updated model.
  predicted_delta, has_non_finite_delta = (
   tensor_utils.zero_all_if_any_non_finite(predicted_delta))
  global_norm_std = tf.sqrt(global_norm_var)
  return tff.utils.update_state(
      server_state,
      model=model_weights,
      optimizer_state=server_optimizer.variables(),
      round_num=server_state.round_num + 1.0,
      predicted_delta = predicted_delta,
      Sx = Sx,
      Sxx = Sxx,
      Sy = Sy,
      Syy = Syy,
      Sxy = Sxy,
      num_participants = num_participants,
      global_norm_mean = global_norm_mean,
      global_norm_std = global_norm_std,
      threshold = global_norm_mean - global_norm_std, 
      )


@attr.s(eq=False, order=False, frozen=True)
class ClientOutput(object):
  """Structure for outputs returned from clients during federated optimization.

  Fields:
  -   `weights_delta`: A dictionary of updates to the model's trainable
      variables.
  -   `client_weight`: Weight to be used in a weighted mean when
      aggregating `weights_delta`.
  -   `model_output`: A structure matching
      `tff.learning.Model.report_local_outputs`, reflecting the results of
      training on the input dataset.
  -   `optimizer_output`: Additional metrics or other outputs defined by the
      optimizer.
  """
  weights_delta = attr.ib()
  client_weight = attr.ib()
  real_client_weight = attr.ib()
  model_output = attr.ib()
  optimizer_output = attr.ib()
  client_transmitted = attr.ib()
  client_delta_norm = attr.ib()

@tf.function
def client_center_square_grad(grad, mu):
  return tf.square(tf.math.subtract(grad,mu))

def create_client_update_fn():
  """Returns a tf.function for the client_update.

  This "create" fn is necesessary to prevent
  "ValueError: Creating variables on a non-first call to a function decorated
  with tf.function" errors due to the client optimizer creating variables. This
  is really only needed because we test the client_update function directly.
  """
  @tf.function
  def client_update(model,
                    dataset,
                    initial_weights,
                    client_optimizer,
                    predicted_delta,
                    threshold,
                    client_weight_fn=None):
    """Updates client model.

    Args:
      model: A `tff.learning.Model`.
      dataset: A 'tf.data.Dataset'.
      initial_weights: A `tff.learning.ModelWeights` from server.
      client_optimizer: A `tf.keras.optimizer.Optimizer` object.
      client_weight_fn: Optional function that takes the output of
        `model.report_local_outputs` and returns a tensor that provides the
        weight in the federated average of model deltas. If not provided, the
        default is the total number of examples processed on device.

    Returns:
      A 'ClientOutput`.
    """

    model_weights = _get_weights(model)
    tff.utils.assign(model_weights, initial_weights)

    num_examples = tf.constant(0, dtype=tf.int32)
    for batch in dataset:
      with tf.GradientTape() as tape:
        output = model.forward_pass(batch)
      grads = tape.gradient(output.loss, model_weights.trainable)
      #grads = tf.nest.map_structure(lambda g: clip_ops.clip_by_norm(g,5.0), grads)
      grads, _ = tf.clip_by_global_norm(grads, 1.0)
      grads_and_vars = zip(grads, model_weights.trainable)
      client_optimizer.apply_gradients(grads_and_vars)
      num_examples += tf.shape(output.predictions)[0]

    aggregated_outputs = model.report_local_outputs()
    weights_delta = tf.nest.map_structure(lambda a, b: a - b,
                                          model_weights.trainable,
                                          initial_weights.trainable)
    weights_delta, has_non_finite_weight = (
        tensor_utils.zero_all_if_any_non_finite(weights_delta))

    weights_delta, client_transmitted, client_delta_norm = estimation_utils.threshold_weights_global_norm(
      structure=weights_delta, 
      replace_structure=predicted_delta, 
      thresh=threshold, 
    )
    real_client_weight = tf.cast(num_examples, dtype=tf.float32)
    if has_non_finite_weight > 0:
      client_weight = tf.constant(0, dtype=tf.float32)
    elif client_weight_fn is None:
      client_weight = tf.cast(num_examples, dtype=tf.float32)
    else:
      client_weight = client_weight_fn(aggregated_outputs)

    #weights_delta_encoded = tf.nest.map_structure(mean_encoder_fn, weights_delta)

    return ClientOutput(
        weights_delta, client_weight, real_client_weight, aggregated_outputs,
        collections.OrderedDict([('num_examples', num_examples)]), client_transmitted, client_delta_norm)

  return client_update


def build_server_init_fn(
  *,
  model_fn: ModelBuilder,
  server_optimizer_fn: Callable[[], tf.keras.optimizers.Optimizer],
  aggregation_process: Optional[measured_process.MeasuredProcess])-> computation_base.Computation:
  """Builds a `tff.tf_computation` that returns the initial `ServerState`.

  The attributes `ServerState.model` and `ServerState.optimizer_state` are
  initialized via their constructor functions. The attribute
  `ServerState.round_num` is set to 0.0.

  Args:
    model_fn: A no-arg function that returns a `tff.learning.Model`.
    server_optimizer_fn: A no-arg function that returns a
      `tf.keras.optimizers.Optimizer`.

  Returns:
    A `tff.tf_computation` that returns initial `ServerState`.
  """

  @computations.tf_computation()
  def server_init_tf():
    server_optimizer = server_optimizer_fn()
    model = model_fn()
    _initialize_optimizer_vars(model, server_optimizer)
    return _get_weights(model), server_optimizer.variables(),

  @computations.tf_computation()
  def wrap_zeroed_weights():
    model = model_fn()
    return tf.nest.map_structure(tf.zeros_like, model.weights.trainable)

  @computations.tf_computation()
  def get_int_0():
    return tf.constant(0, dtype=tf.int32)
  @computations.tf_computation()
  def get_float_0():
    return tf.constant(0.0, dtype=tf.float32)
    


  @computations.federated_computation()
  def initialize_computation():
    model = model_fn()
    initial_global_model, initial_global_optimizer_state = intrinsics.federated_eval(
        server_init_tf, placements.SERVER)
    return intrinsics.federated_zip(ServerState(
        model=initial_global_model,
        optimizer_state=initial_global_optimizer_state,
        round_num=tff.federated_value(0.0, tff.SERVER),
        predicted_delta = intrinsics.federated_eval(wrap_zeroed_weights, placements.SERVER),
        Sy = intrinsics.federated_eval(wrap_zeroed_weights, placements.SERVER), 
        Syy = intrinsics.federated_eval(wrap_zeroed_weights, placements.SERVER),
        Sx = intrinsics.federated_eval(wrap_zeroed_weights, placements.SERVER),
        Sxx = intrinsics.federated_eval(wrap_zeroed_weights, placements.SERVER),
        Sxy = intrinsics.federated_eval(wrap_zeroed_weights, placements.SERVER),
        num_participants= intrinsics.federated_eval(get_int_0, placements.SERVER),
        global_norm_mean = intrinsics.federated_eval(get_float_0, placements.SERVER),
        global_norm_std = intrinsics.federated_eval(get_float_0, placements.SERVER),
        threshold = intrinsics.federated_eval(get_float_0, placements.SERVER),
        delta_aggregate_state=aggregation_process.initialize(),
        ))

  return initialize_computation


def build_fed_avg_process(
    model_fn: ModelBuilder,
    client_optimizer_fn: OptimizerBuilder,
    client_lr: Union[float, LRScheduleFn] = 0.1,
    server_optimizer_fn: OptimizerBuilder = tf.keras.optimizers.SGD,
    server_lr: Union[float, LRScheduleFn] = 1.0,
    client_weight_fn: Optional[ClientWeightFn] = None,
    aggregation_process: Optional[measured_process.MeasuredProcess] = None,
) -> tff.templates.IterativeProcess:
  """Builds the TFF computations for optimization using federated averaging.

  Args:
    model_fn: A no-arg function that returns a `tff.learning.Model`.
    client_optimizer_fn: A function that accepts a `learning_rate` keyword
      argument and returns a `tf.keras.optimizers.Optimizer` instance.
    client_lr: A scalar learning rate or a function that accepts a float
      `round_num` argument and returns a learning rate.
    server_optimizer_fn: A function that accepts a `learning_rate` argument and
      returns a `tf.keras.optimizers.Optimizer` instance.
    server_lr: A scalar learning rate or a function that accepts a float
      `round_num` argument and returns a learning rate.
    client_weight_fn: Optional function that takes the output of
      `model.report_local_outputs` and returns a tensor that provides the weight
      in the federated average of model deltas. If not provided, the default is
      the total number of examples processed on device.

  Returns:
    A `tff.templates.IterativeProcess`.
  """

  client_lr_schedule = client_lr
  if not callable(client_lr_schedule):
    client_lr_schedule = lambda round_num: client_lr

  server_lr_schedule = server_lr
  if not callable(server_lr_schedule):
    server_lr_schedule = lambda round_num: server_lr

  with tf.Graph().as_default():
    dummy_model = model_fn()
    dummy_optimizer = server_optimizer_fn()
    _initialize_optimizer_vars(dummy_model, dummy_optimizer)
    optimizer_variable_type = type_conversions.type_from_tensors(
        dummy_optimizer.variables())    

  
  initialize_computation = build_server_init_fn(
        model_fn = model_fn,
        # Initialize with the learning rate for round zero.
        server_optimizer_fn = lambda: server_optimizer_fn(server_lr_schedule(0)), 
        aggregation_process = aggregation_process)

  server_state_type = server_init_tf.type_signature.result
  model_weights_type = server_state_type.model
  predicted_delta_type = server_state_type.predicted_delta
  round_num_type = server_state_type.round_num
  threshold_type = server_state_type.global_norm_mean
  num_participants_type = server_state_type.num_participants
  norm_mean_type = server_state_type.global_norm_mean
  norm_std_type = server_state_type.global_norm_std

  tf_dataset_type = tff.SequenceType(dummy_model.input_spec)
  model_input_type = tff.SequenceType(dummy_model.input_spec)

  aggregation_state = aggregation_process.initialize.type_signature.result.member

  server_state_type = ServerState(
        model=model_weights_type,
        optimizer_state=optimizer_variable_type,
        round_num=round_num_type,
        predicted_delta = predicted_delta_type,
        Sy = predicted_delta_type, 
        Syy = predicted_delta_type,
        Sx = predicted_delta_type,
        Sxx = predicted_delta_type,
        Sxy = predicted_delta_type,
        num_participants= num_participants_type,
        global_norm_mean = norm_mean_type,
        global_norm_std = norm_std_type,
        threshold = threshold_type,
        delta_aggregate_state=aggregation_state,
        )



  @tff.tf_computation(model_input_type, model_weights_type, round_num_type, predicted_delta_type, threshold_type)
  def client_update_fn(tf_dataset, initial_model_weights, round_num, predicted_delta, threshold):
    client_lr = client_lr_schedule(round_num)
    client_optimizer = client_optimizer_fn(client_lr)
    client_update = create_client_update_fn()
    return client_update(model_fn(), tf_dataset, initial_model_weights,
                         client_optimizer, predicted_delta, threshold,client_weight_fn,)

  @tff.tf_computation(server_state_type, model_weights_type.trainable, num_participants_type, norm_mean_type, norm_std_type)
  def server_update_fn(server_state, model_delta, num_participants, norm_mean, norm_std):
    model = model_fn()
    server_lr = server_lr_schedule(server_state.round_num)
    server_optimizer = server_optimizer_fn(server_lr)
    # We initialize the server optimizer variables to avoid creating them
    # within the scope of the tf.function server_update.
    _initialize_optimizer_vars(model, server_optimizer)
    return server_update(model, server_optimizer, server_state, model_delta, num_participants, norm_mean, norm_std)


  @tff.tf_computation(tf.float32, tf.float32)
  def local_variance_fn(grad, mu):
    return client_center_square_grad(grad, mu)

  @tff.federated_computation(
      tff.FederatedType(server_state_type, tff.SERVER),
      tff.FederatedType(tf_dataset_type, tff.CLIENTS))
  def run_one_round(server_state, federated_dataset):
    """Orchestration logic for one round of computation.

    Args:
      server_state: A `ServerState`.
      federated_dataset: A federated `tf.Dataset` with placement `tff.CLIENTS`.

    Returns:
      A tuple of updated `ServerState` and the result of
      `tff.learning.Model.federated_output_computation`.
    """
    client_model = tff.federated_broadcast(server_state.model)
    client_round_num = tff.federated_broadcast(server_state.round_num)
    client_predicted_delta = tff.federated_broadcast(server_state.predicted_delta)
    client_threshold = tff.federated_broadcast(server_state.threshold)

    client_outputs = tff.federated_map(
        client_update_fn,
        (federated_dataset, client_model, client_round_num,client_predicted_delta, client_threshold ))

    client_weight = client_outputs.client_weight
    num_participants = tff.federated_sum(client_outputs.client_transmitted)
    new_global_norm_mean = tff.federated_mean(client_outputs.client_delta_norm, 
      weight=client_outputs.real_client_weight)

    var_at_clients = tff.federated_map( local_variance_fn, 
          (client_outputs.client_delta_norm, 
            tff.federated_broadcast(new_global_norm_mean)) )

    new_global_norm_var = tff.federated_mean(var_at_clients,
          weight=client_outputs.real_client_weight)
    logging.info(f'new var: {new_global_norm_var}')
    logging.info(f'new mean: {new_global_norm_mean}')

    aggregation_output = aggregation_process.next(
        server_state.delta_aggregate_state, client_outputs.weights_delta,
        client_weight)

    # model_delta = tff.federated_mean(
    #     client_outputs.weights_delta, weight=client_weight)

    server_state = tff.federated_map(server_update_fn,
                                     (server_state, 
                                      aggregation_output.result,
                                      num_participants,
                                      new_global_norm_mean,
                                      new_global_norm_var))

    aggregated_outputs = dummy_model.federated_output_computation(
        client_outputs.model_output)
    if aggregated_outputs.type_signature.is_struct():
      aggregated_outputs = tff.federated_zip(aggregated_outputs)

    return server_state, aggregated_outputs

  # @tff.federated_computation
  # def initialize_fn():
  #   return tff.federated_value(server_init_tf(), tff.SERVER)

  return tff.templates.IterativeProcess(
      initialize_fn=initialize_computation, next_fn=run_one_round)
